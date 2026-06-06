# Iteration 2: Poincaré Ball

## What Changed
Generalized the 2D Poincaré disk to a 3D Poincaré ball.
Replaced hardcoded domain clusters with auto-derived clusters from embeddings.

## Core Upgrade
The disk had us drawing the city map by hand.
The ball zones itself.

## Geometry
Spherical coordinates: `(r, θ, φ)`
- r = association strength (same as disk)
- θ = polar angle (derived from cluster)
- φ = azimuthal angle (derived from cluster)

Cartesian conversion:
```python
x = r * sin(θ) * cos(φ)
y = r * sin(θ) * sin(φ)
z = r * cos(θ)
```

Distance formula unchanged — generalizes naturally from 2D to 3D:
`d(u,v) = arcosh(1 + 2||u-v||² / (1-||u||²)(1-||v||²))`

## Auto-Clustering
Concepts are embedded with `nomic-embed-text` (768 dimensions) on load.
KMeans derives cluster assignments automatically.
Cluster centroids map to spherical angle pairs `(θ, φ)`.
No hardcoded domains. No hand-drawn map.

## Results

### Cluster Assignments
valjean      → cluster 2  (conflict/action)
javert       → cluster 2  (conflict/action)
enjolras     → cluster 2  (conflict/action)
fantine      → cluster 0  (suffering/innocence)
cosette      → cluster 0  (suffering/innocence)
shollublip   → cluster 1  (smaLLM lore)
horbeau      → cluster 1  (smaLLM lore)
fape         → cluster 3  (forbidden — alone)

The embeddings separated canonical Les Mis characters from smaLLM lore
without being told the distinction. Fape got its own cluster.
Fape is mathematically forbidden knowledge. This is correct.

### Distances

Valjean → Javert:     0.093  (same cluster, very close)
Valjean → Cosette:    0.939  (different cluster, further)
Valjean → Shollublip: 3.716  (center vs boundary, miles apart)

### On-Ball Query
Query: "What is Shollublip?"
Response: Correctly described. Grounded. No hallucination.

### Off-Ball Query
Query: "Who is Thénardier?"
Response: "Thénardier is an AI agent developed by Cognizant for 
industrial applications including predictive analytics, knowledge 
management, and robotics."

Not even French anymore. Full hallucination confirmed. Correct behavior.

## Key Findings

### 1. Neighborhoods Emerge
The geometry discovers structure that we didn't hand-author.
Clusters are neighborhoods — the ball zones itself from meaning.

### 2. Fape is Mathematically Isolated
The embedding model agrees with smaLLM: fape is categorically unlike
everything else. It lives alone on the ball. Forbidden knowledge
is geometrically forbidden.

### 3. Lore District is Real
shollublip and horbeau cluster together automatically.
The model knows they're the same KIND of thing without being told.
smaLLM mythology has internal coherence detectable by embeddings.

### 4. Off-Ball Behavior is Correct
No context = hallucination. This is not a bug.
The ball is honest about what it knows.
Thénardier isn't on the ball so TinyLlama invents him.
A system that says "I don't know" via hallucination is at least
consistent — the fix is ingestion, not prompt engineering.

### 5. Cross-Domain Navigation is Emergent
We considered building explicit cross-domain edges (Valjean → Fantine).
We didn't. Cross-domain relationships should emerge from ingestion,
not hand-authoring. We know Les Mis a priori — a book written tomorrow
gives us no such advantage. The geometry has to do the work.

## What Doesn't Work Yet
- No text ingestion pipeline — concepts.json is still hand-authored
- Cross-domain navigation is emergent, not built — needs ingestion first
- 3D is still a simplification — real knowledge has more dimensions

## Next Iteration: Product Manifold
The big one. Mixed curvature geometry:
- Hyperbolic (H²) for hierarchy
- Spherical (S¹) for cyclical/periodic structure  
- Flat (ℝ) for linear relationships

But first: text ingestion pipeline.
Load a book. Build the ball. No hand-authoring.
That's the unlock for everything else.

## Iteration 2.1: Auto-Ingestion Pipeline

### What Changed
Replaced hand-authored concepts.json with auto-generated concepts from raw text.
Added `ingest.py` — full ingestion pipeline from text to ball.

### Pipeline
1. Chunk text (2000 chars, 200 overlap)
2. Extract concepts via spaCy NER + noun chunks
3. Score by TF-IDF (chunk-level IDF — self contained, no external corpus)
4. Take top N by TF-IDF score
5. Summarize each concept from its chunks via TinyLlama
6. Write concepts.json — association_strength derived from raw TF

### Key Design Decision: Chunk-Level IDF
IDF computed against chunks as documents, not an external corpus.
Concepts appearing in nearly every chunk (noise) score near zero.
Concepts appearing in some chunks (signal) score high.
Self contained — works on any text without external reference data.

### Results
Fed raw Les Misérables text (3.25M characters, 1806 chunks).
No hand-authoring. No domain knowledge. No prior assumptions.

Top concepts by TF-IDF: two, time, hand, thing, one, moment, marius,
head, first, word, sort, valjean, child, house, jean valjean...

Valjean and Jean Valjean landed in the same cluster (3) automatically.
Marius also cluster 3 — correctly near Valjean.
Generic nouns formed their own noise neighborhoods (clusters 1 and 2).
The noise neighborhood emerged naturally — not a bug, sparse regions
on the ball are honest representations of low semantic density.

### The Key Test
Query: "How did Jean Valjean help Fantine?"

**Bare TinyLlama (no ball):**
Knew the basic relationship but hallucinated details — "freed her from
the galleys on his ship", "Paris Conservatoire", "becoming an actress."
Confident, coherent, factually drifting.

**With the ball:**
Factory connection correct. Cosette connection correct. Financial
assistance correct. Less hallucination, more grounded.

**The finding:**
The ball doesn't add information TinyLlama doesn't have.
It constrains the paths it can take.
RAG tries to inject the right answer.
The ball removes the wrong ones.
Completely different mechanism.

### What's Next
Les Mis was the test harness. The real test is new information —
The Intelligent Investor + SpaceX S1. Feed the ball something
TinyLlama provably doesn't know. See if geometry fills the gap
where training data ends.

That's the product manifold problem.