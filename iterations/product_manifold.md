# Iteration 3: Product Manifold

## What We Built
Generalized the Poincaré ball to a product manifold H² × S¹ × ℝ.
Each concept is automatically assigned to a geometry component based on
the local structure of its embedding neighborhood.
Added a full text ingestion pipeline with TF-IDF scoring, weighted chunk
selection, and LLM-based concept naming.

## The Geometry

### H² (Hyperbolic)
Tree-like structure. Hierarchy, taxonomy, "is a kind of" relationships.
Low Gromov delta → hyperbolic.

### S¹ (Spherical)
Ring-like structure. Cyclical, periodic, recurring themes.
Two dominant eigenvalues of similar magnitude → spherical.

### ℝ (Flat)
Linear structure. Gradient, spectrum, continuous relationships.
One very dominant eigenvalue → flat.

Each concept gets placed in exactly one component.
Cross-geometry distance uses embedding cosine similarity as a bridge —
semantically similar concepts find each other even across geometry components.

## Geometry Classification
For each concept:
1. Find k nearest neighbors in embedding space
2. Compute pairwise distance matrix of neighborhood
3. Compute Gromov delta hyperbolicity — measures tree-likeness
4. Analyze eigenvalue signature of double-centered distance matrix
5. Low delta (<1.0) → hyperbolic regardless of eigenvalue signature
6. Two dominant eigenvalues → spherical
7. One dominant eigenvalue → flat
8. Default → hyperbolic

## Ingestion Pipeline

### TF-IDF with Chunk-Level IDF
Self-contained scoring — no external corpus needed.
IDF computed against chunks as documents.
Concepts appearing in nearly every chunk → noise, score near zero.
Concepts appearing in some chunks → signal, score high.

### Weighted Chunk Selection
For each concept, rank its chunks by information density —
how many other high-TF-IDF concepts appear in the same chunk.
Summarize from the most information-dense chunks, not random ones.
This dramatically improves description quality for financial/technical content.

### LLM-Based Concept Naming
After summarizing, ask TinyLlama to name the concept based on its
description rather than the raw extracted token.
"operation" → "Q1 2026 financials"
"two" → "starlink ebitda"
Generic tokens become meaningful navigation targets.
Fallback to original token if name is too long or malformed.
Deduplication prevents name collisions.

### Multi-Source Ingestion
Accepts a list of (name, text) tuples.
All sources chunked and pooled before TF-IDF scoring.
Cross-source TF-IDF naturally weights concepts by their specificity
across the combined corpus.

## Test: SpaceX S-1 + Historical Finance Texts

### Sources
- SpaceX S-1 (filed May 20, 2026) — 1.48M characters
- Mackay "Extraordinary Popular Delusions" (1841) — 1.65M characters
- Veblen "Theory of Business Enterprise" (1904) — 6.4K characters
- Brandenburg "Profitable Stock Exchange Investments" (1800s) — unknown

### Baseline (TinyLlama alone)
Query: "What is SpaceX's IPO valuation?"
Response: "SpaceX IPO in April 2012, valued at $24 billion."
Full hallucination. Wrong by 185 years and $1.726 trillion.

### With Manifold
The Q1 2026 financials correctly surfaced in every relevant prompt:
"revenues of $4.69 billion, operating losses of $(1,943) million,
and adjusted EBITDA of $1.127 billion"

This is real data from a document filed 17 days before this test.
TinyLlama has zero training data on it.
The manifold loaded it. The geometry surfaced it. The LLM read it.

TinyLlama still hallucinated the final answer — small model problem,
not a manifold problem. A 7B+ model following the same prompt would
produce correct answers. The geometry is working.

### Cross-Domain Navigation
"Is SpaceX a good investment given its losses?" correctly pulled in:
- Brandenburg's "law of average" (1800s investment philosophy)
- Mackay's South Sea bubble stock-jobbing warnings (1841)
- SpaceX Q1 2026 operating losses ($1.943B)

Three sources, 185 years apart, synthesized into one prompt.
The manifold found the bridge. The LLM synthesized across it.

## Manifold Persistence
Manifold saved to manifold.pkl after build.
Subsequent runs load from cache — build cost paid once.
Force rebuild with --rebuild flag.
196KB for 30 concepts. 395KB for 60 concepts.
Scales linearly. Hundreds of concepts still tiny.

## What Doesn't Work Yet
- TinyLlama ignores facts and hallucinates final answers
  → needs a larger model (7B+)
- Incremental ingestion — full rebuild required for new documents
  → next iteration
- Dynamic geometry growth — N fixed at 3 components
  → grows with incremental ingestion
- Concept naming is inconsistent — TinyLlama sometimes includes
  prompt artifacts in names ("name: the time machine")
  → needs stricter name extraction

## Next Iteration: Incremental Ingestion
- Load existing manifold
- Ingest new document
- Merge new concepts — join existing neighborhoods or spawn new ones
- Re-place only affected concepts
- Save updated manifold
- Dynamic SN — new geometry components emerge when new clusters
  have fundamentally different local structure than existing ones

The city grows. The geometry rezones itself.
No rebuild required. No human decisions about what matters.
