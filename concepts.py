import spacy
import ollama
import numpy as np
from product_manifold import find_nearest_product

nlp = spacy.load("en_core_web_sm")

def get_embedding(text):
    """Get embedding from ollama nomic-embed-text"""
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return np.array(response['embedding'])

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_concepts(text):
    """
    Extract key concepts from a text query
    Returns list of lowercase lemmas
    """
    doc = nlp(text)
    concepts = []
    for ent in doc.ents:
        concepts.append(ent.text.lower())
    for chunk in doc.noun_chunks:
        concepts.append(chunk.root.lemma_.lower())
    return list(set(concepts))

def match_to_disk(extracted_concepts, disk_concepts, threshold=0.60):
    """
    Fuzzy match extracted concepts to disk concepts using ollama embeddings.
    Matches against name AND description.
    """
    if not extracted_concepts:
        return []

    matched = []
    seen = set()

    query_embeddings = [get_embedding(c) for c in extracted_concepts]

    for disk_concept in disk_concepts:
        if disk_concept['name'] in seen:
            continue

        disk_string = f"{disk_concept['name']}: {disk_concept['description']}"
        disk_embedding = get_embedding(disk_string)

        best_score = max(
            cosine_similarity(qe, disk_embedding) 
            for qe in query_embeddings
        )

        if best_score >= threshold:
            matched.append((disk_concept, best_score))
            seen.add(disk_concept['name'])

    matched.sort(key=lambda x: x[1], reverse=True)
    return matched

def get_context(query, disk_concepts, top_k=3):
    extracted = extract_concepts(query)
    print(f"Extracted concepts: {extracted}")

    matched_with_scores = match_to_disk(extracted, disk_concepts)

    if not matched_with_scores:
        print("No concepts matched on disk")
        return None

    for concept, score in matched_with_scores:
        print(f"  Matched: {concept['name']} (score={score:.3f})")

    context = []
    for concept, score in matched_with_scores:
        nearest = find_nearest_product(concept, disk_concepts, top_k=top_k)
        context.append({
            'query_concept': concept['name'],
            'match_score': score,
            'nearest': nearest
        })

    return context