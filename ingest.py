import json
import math
import ollama
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

# --- Step 1: Chunk the text ---
def chunk_text(text, chunk_size=2000, overlap=200):
    """Sliding window chunker"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Step 2: Extract concepts per chunk ---
def extract_concepts_from_chunk(chunk):
    """
    Extract named entities AND domain-specific noun phrases.
    spaCy NER for proper nouns, noun chunks for domain terms.
    Filter out very short tokens.
    """
    doc = nlp(chunk)
    concepts = set()

    # Named entities
    for ent in doc.ents:
        name = ent.text.strip().lower()
        if len(name) > 2:
            concepts.add(name)

    # Noun chunks — domain specific terms
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in ["NOUN", "PROPN"]:
            name = chunk.root.lemma_.strip().lower()
            if len(name) > 3:
                concepts.add(name)

    return concepts

# --- Step 3: Build TF-IDF scores ---
def build_tfidf(chunks, min_tf=10):
    """
    Compute TF-IDF scores using chunks as documents.
    TF = total occurrences across all chunks
    IDF = log(total_chunks / chunks_containing_concept)
    Score = TF * IDF
    
    Concepts appearing in nearly every chunk score near zero — they're noise.
    Concepts appearing in some chunks score high — they're signal.
    """
    total_chunks = len(chunks)
    
    # TF: total count across all chunks
    tf = defaultdict(int)
    # DF: how many chunks contain this concept
    df = defaultdict(int)
    # Store which chunks contain each concept
    concept_chunks = defaultdict(list)

    for i, chunk in enumerate(chunks):
        concepts = extract_concepts_from_chunk(chunk)
        chunk_concepts = set()
        for concept in concepts:
            tf[concept] += 1
            concept_chunks[concept].append(chunk)
            chunk_concepts.add(concept)
        for concept in chunk_concepts:
            df[concept] += 1

    # Compute TF-IDF scores
    tfidf = {}
    for concept in tf:
        if tf[concept] < min_tf:
            continue
        idf = math.log(total_chunks / (1 + df[concept]))
        tfidf[concept] = tf[concept] * idf

    return tfidf, concept_chunks, tf

# --- Step 4: Derive association strength ---
def derive_association_strength(tf_score, max_tf):
    """Normalize raw TF count to 0.0-1.0"""
    return min(tf_score / max_tf, 1.0)

# --- Step 5: Summarize concept from chunks ---
def score_chunks_for_concept(concept, chunks, all_tfidf):
    """
    Score chunks by how information-dense they are for this concept.
    Prefer chunks that contain many high-TF-IDF concepts, not just this one.
    More signal neighbors = more information-dense chunk.
    """
    scored = []
    for chunk in chunks:
        # Count how many high-value concepts appear in this chunk
        chunk_lower = chunk.lower()
        signal_count = sum(
            1 for c, score in all_tfidf.items()
            if score > 400 and c in chunk_lower
        )
        scored.append((signal_count, chunk))
    
    # Sort by signal density descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored]

def summarize_concept(concept, chunks, all_tfidf, max_chunks=5):
    """
    Feed concept's most information-dense chunks to TinyLlama.
    Get a one sentence description back.
    """
    # Pick highest signal chunks instead of first 5
    ranked_chunks = score_chunks_for_concept(concept, chunks, all_tfidf)
    sample = ranked_chunks[:max_chunks]
    context = "\n---\n".join(sample)

    prompt = f"""You are summarizing a text document.
Using ONLY the information explicitly present in these excerpts, complete this sentence in exactly one sentence:
"{concept} is..."

Rules:
- Only use what is directly written in the text
- Do not interpret, infer, or add context beyond what is explicitly stated
- Do not add outside knowledge
- If the text does not clearly describe {concept}, say "{concept} is a concept mentioned in this text."

TEXT EXCERPTS:
{context}

Complete the sentence "{concept} is..." in exactly one sentence:"""

    response = ollama.chat(
        model='tinyllama',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content'].strip()

def name_concept(concept_token, description):
    """
    Ask TinyLlama to give this concept a meaningful name
    based on its description rather than the extracted token.
    """
    prompt = f"""Based on this description, give a short 1-3 word name that captures what this concept is about.
Do not use generic words like "time", "one", "first", "number".
Use specific meaningful terms from the description.

Description: {description}

Respond with ONLY the name, nothing else. 2-3 words maximum:"""

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    name = response['message']['content'].strip().lower()
    # Fallback to original token if response is too long or weird
    if len(name.split()) > 4 or len(name) > 40:
        return concept_token
    return name

# --- Step 6: Write concepts.json ---
def build_concepts_json(texts, n_top=20, min_tf=10, output_path="concepts.json"):
    """
    Full ingestion pipeline. Accepts a list of (name, text) tuples.
    """
    # Combine all texts with source tracking
    all_chunks = []
    
    for source_name, text in texts:
        print(f"Chunking {source_name}...")
        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks across all sources: {len(all_chunks)}")

    print("Computing TF-IDF scores...")
    tfidf, concept_chunks, tf = build_tfidf(all_chunks, min_tf=min_tf)
    print(f"  {len(tfidf)} concepts above min_tf={min_tf}")

    if not tfidf:
        print("No concepts found! Lower min_tf.")
        return

    top_concepts = sorted(
        tfidf.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n_top]

    max_tf = max(tf[c] for c, _ in top_concepts)
    print(f"  Top concept: '{top_concepts[0][0]}' (tfidf={top_concepts[0][1]:.1f})")

    print("Summarizing concepts with TinyLlama...")
    concepts_out = []
    seen_names = set()
    for concept, score in top_concepts:
        raw_tf = tf[concept]
        strength = derive_association_strength(raw_tf, max_tf)

        print(f"  Summarizing: {concept} (tfidf={score:.1f}, tf={raw_tf}, strength={strength:.3f})")
        description = summarize_concept(concept, concept_chunks[concept], tfidf)
        better_name = name_concept(concept, description)
        print(f"    → {better_name}: {description[:60]}...")

        if better_name in seen_names:
            better_name = f"{better_name}_{concept}"
        seen_names.add(better_name)

        concepts_out.append({
            "name": better_name,
            "association_strength": round(strength, 3),
            "domain": "auto",
            "offset": 0.0,
            "description": description,
            "properties": []
        })

    output = {"concepts": concepts_out}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(concepts_out)} concepts to {output_path}")
    return output


if __name__ == "__main__":
    sources = [
        ("SpaceX S-1", "spacex_s1.txt"),
        ("Mackay - Extraordinary Popular Delusions", "mackay.txt"),
        ("Veblen - Theory of Business Enterprise", "veblen.txt"),
        ("Brandenburg - Profitable Stock Exchange Investments","brandenburg.txt"),
    ]

    texts = []
    for name, path in sources:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Loaded {name}: {len(text):,} characters")
            texts.append((name, text))
        except FileNotFoundError:
            print(f"WARNING: {path} not found, skipping")

    build_concepts_json(texts, n_top=60, min_tf=10)