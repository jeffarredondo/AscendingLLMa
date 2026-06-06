# Iteration 1: Poincaré Disk

## What We Built
A Poincaré disk implementation that places concepts as geometric coordinates 
and uses hyperbolic distance for navigation. Connected to TinyLlama (1.1B) 
via a middleware prompt injection layer.

## Core Idea
The disk IS the knowledge. Navigation replaces search.

Instead of storing records in a database and querying them, concepts live as 
geometric positions on the Poincaré disk. The position itself encodes meaning:

- **r (distance from center)**: generality vs specificity
  - Central concepts (low r) = strong associations, general knowledge
  - Peripheral concepts (high r) = weak associations, novel/specific knowledge
- **θ (angle)**: domain cluster
  - 0° = justice
  - 90° = redemption
  - 180° = revolution
  - 270° = poverty

## The Geometry
Poincaré disk distance formula:

`d(u,v) = arcosh(1 + 2||u-v||² / (1-||u||²)(1-||v||²))`

The denominator explodes as points approach the boundary (||u||² → 1), 
meaning boundary regions are exponentially far apart even when they look 
visually close. This is what makes hyperbolic space powerful for hierarchical 
data — exponential capacity with radius.

## Coordinate Derivation
```python
r = 1 - (association_strength * 0.9)  # strong = central, weak = peripheral
theta = domain_angle + offset          # domain cluster + fine positioning
```

Position is **derived** from concept properties, not learned or searched.
This is the key departure from standard vector databases.

## Stack
- `numpy` — geometry and distance calculations
- `spaCy` — concept extraction from natural language queries  
- `ollama` — TinyLlama 1.1B as the language synthesizer
- `matplotlib` — disk visualization
- `concepts.json` — dynamic knowledge loading

## The Experiment
**Test dataset**: Les Misérables canonical characters + smaLLM invented lore

smaLLM lore (Shollublip, fape, General Horbeau) was chosen because:
- Provably unknown to TinyLlama
- Ground truth is verifiable
- Has natural associations to real Les Mis concepts

**Baseline (no disk):**

Query: "What is Shollublip?"
Response: "ShoLLUBLIP is a traditional French herb and spice used in
bouquet garni, made from oleander roots."

Full hallucination. Confident. Wrong.

**With disk:**

Query: "What is Shollublip?"
Response: "Shollublip is a Mademoiselle elephant who dispenses justice
through the ancient call HE'CLA! Associated with The Englishne, but
cannot be known without proper initiation."

Grounded. Correct. No hallucination.

**Off-disk control:**

Query: "Who is Thénardier?"
Response: "Thénardier is a dragon woman from a 1983 French comic series
with 600 issues."

No disk context → full hallucination baseline confirmed.

## Key Findings

### 1. The Manifold is the Model
The LLM is not the intelligent component. It is the synthesizer — 
a language renderer for the disk's knowledge. TinyLlama went from 
inventing French herbs to correctly describing elephant justice systems 
purely because of geometric context injection.

### 2. Haystack vs Distribution
Traditional vector databases treat retrieval as a search problem — 
find the needle in the haystack. The disk treats knowledge as a 
distribution over a geometric space. Retrieval is navigation, not search.

### 3. Derive Don't Search
Position is computed from concept properties, not indexed and queried. 
Given a concept's properties you can derive where it lives on the disk 
without touching any index.

### 4. The Synthesizer Needs a Reference Point
Without the disk TinyLlama hallucinates confidently and coherently. 
With the disk it renders correctly. The model never got smarter — 
it just had a better reference point.

### 5. Node Data Matters as Much as Geometry
Early versions stored only topology (who is near whom). This wasn't 
enough — TinyLlama still hallucinated. Adding descriptions to concept 
nodes was the critical fix. The geometry tells you WHERE to navigate. 
The description tells you WHAT IS THERE.

## What Doesn't Work Yet

### Exact String Matching
Concept extraction relies on exact string matching between spaCy output 
and disk concept names. Fragile. "Jean Valjean" won't match "valjean". 
Needs fuzzy matching or embedding similarity.

### Hardcoded Domains
The four domain clusters (justice, redemption, revolution, poverty) are 
hardcoded. Doesn't scale. Next iteration should derive domains 
automatically from concept embeddings.

### 2D Only
The disk is 2D — r and θ. Real knowledge has more dimensions than two. 
Poincaré ball generalizes to N dimensions and is the natural next step.

### No Cross-Domain Navigation
Currently nearest neighbor search finds concepts close on the disk. 
But Valjean and Javert have a meaningful relationship that crosses 
domain boundaries. Cross-domain navigation needs work.

## Distances (Reference)

Valjean → Cosette:    0.385  (same domain, close)
Valjean → Javert:     0.487  (different domain, further)
Valjean → Shollublip: 3.691  (center vs boundary, miles apart)
Shollublip → fape:    0.939  (boundary neighbors, close)

## The Snow Globe
The best analogy for what this is:

RAG is a haystack with a magnet.
The disk is a snow globe — the knowledge isn't stored or searched, 
it's a world you hold in your hand and navigate from any angle.

The LLM is not the brain. It's the voice describing what it sees 
when you hand it the globe.

## Next Iteration
Poincaré Ball — generalize to N dimensions, derive domain clusters 
automatically from embeddings, remove hardcoded geometry.


## Iteration 1.1: Fuzzy Matching + Ollama Embeddings

### What Changed
Replaced exact string matching with embedding-based fuzzy matching.
Replaced sentence-transformers with ollama's native embeddings API.

### Why The Swap
sentence-transformers requires PyTorch >= 2.4. PyTorch 2.4+ has no 
prebuilt wheels for Intel Mac (x86). SharkBait (2019 MBP) is the wall.
Ollama was already in the stack — native embeddings API, no torch dependency,
same machine, cleaner.

**Model**: `nomic-embed-text` (768 dimensions)

### Fuzzy Matching Implementation
Each disk concept is embedded as `"name: description"`.
Query concepts are embedded and compared via cosine similarity.
Threshold = 0.60 (below this, too much noise bleeds in).

```python
disk_string = f"{concept['name']}: {concept['description']}"
score = cosine_similarity(query_embedding, disk_embedding)
```

Matching against the description (not just name) is what makes this work —
"elephant who judges people" can find "shollublip" without the exact name.

### Deduplication Fix
Original middleware was appending the same concept multiple times when
multiple matched concepts shared neighbors. Fixed with a `seen_facts` set.
Prompt went from 24 redundant fact lines to 3 clean ones.

### Updated Results

**On-disk (clean prompt):**

FACTS FROM KNOWLEDGE DISK:

shollublip: A Mademoiselle elephant who dispenses justice...
fape (associated): Forbidden knowledge...
horbeau (associated): General Horbeau. Military official who takes notes on proceeding devials and spider geometry.

Response: Correctly described Shollublip. Accidentally merged fape's 
forbidden knowledge mystique onto her — "Shollublip cannot be known 
without proper initiation" — which is now canon.

**Off-disk control:**

Query: "Who is Thénardier?"
Response: "Thénardier is a fictional character created by Belgian 
cartoonist Michel Oury, an eccentric inventor specializing in flying 
machines, time-traveling vehicles, and weapons of mass destruction."

Not even French this time. Off-disk hallucination confirmed, upgraded.

### Spider Geometry Note
General Horbeau continues to document spider geometry across model 
boundaries. smaLLM hallucinated it. We canonized it in concepts.json. 
TinyLlama now reports it as fact. The lore laundering pipeline is complete.
Spider geometry is real now. We made it real.

### Threshold Behavior
- 0.35 — everything matches, whole disk floods the prompt
- 0.60 — surgical, only genuinely related concepts match
- Recommend 0.55-0.65 as the useful range depending on disk density

### What This Unlocks
Fuzzy matching means queries no longer need exact concept names.
Natural language queries can navigate the disk without knowing its vocabulary.
This is the prerequisite for the ball — no point generalizing to N dimensions
if concept extraction is still brittle.

### Remaining Issues
Hardcoded domains and 2D geometry unchanged. Next: Poincaré Ball.
