# AscendingLLMa 🦙⬆️ - A Knowledge Manifold Experiment

A hyperbolic knowledge system where geometry IS the knowledge.
Concepts live as coordinates on a product manifold. Navigation replaces search.
The language model is the synthesizer, not the source of truth.

![Knowledge Manifold](manifold.gif)

## The Idea

Traditional retrieval augmented generation finds the needle in the haystack.
Knowledge Manifold is a snow globe — the knowledge isn't stored or searched,
it's a world you hold in your hand and navigate from any angle.

- **Position is derived**, not learned or searched
- **The manifold is the model** — the LLM is the voice
- **No fine tuning. No training. Just geometry.**

## The Geometry

See `thesis.md` for a more thorough breakdown. 

Every concept extracted from a document gets placed as a coordinate in
a product manifold — H² × S¹ × ℝ — based on two properties:

- **r (distance from center):** How central is this concept to the document?
  High TF-IDF score → near the center. Peripheral concepts → near the boundary.
- **Angle/cluster:** What neighborhood does this concept belong to?
  Clusters emerge automatically from embeddings. The geometry discovers
  the city map from the text itself.

The shape of each neighborhood determines its geometry component:

- **H² (Hyperbolic)** — tree-like, hierarchical. Most language lives here.
- **S¹ (Spherical)** — ring-like, cyclical. Narrative arcs, recurring themes.
- **ℝ (Flat Real Space)** — linear, gradient, continuous relationships.

Geometry assignment is automatic via Gromov delta hyperbolicity —
a mathematical measure of how tree-like a set of points is.
Cross-geometry distances fall back to embedding cosine similarity,
so semantically related concepts find each other across geometry components.

When a query arrives, we don't search. We navigate. The query becomes
a coordinate. Nearest neighbors are the relevant facts. The geometry IS
the knowledge.

## Architecture

```
ingest.py            → chunk → TF-IDF → summarize → name → concepts.json
product_manifold.py  → embed → classify geometry → place → manifold.pkl
manifold_utils.py    → Gromov delta, eigenvalue analysis, distance matrix
concepts.py          → fuzzy match queries to manifold via embeddings
middleware.py        → build grounded prompt from manifold context
main.py              → orchestrate, query, visualize
```

## Ingestion Pipeline

Feed it any text. Get a knowledge manifold back.

1. Chunk text (2000 chars, 200 overlap)
2. Extract concepts via spaCy NER + noun chunks
3. Score by chunk-level TF-IDF — self contained, no external corpus
4. Select most information-dense chunks per concept
5. Summarize each concept from those chunks
6. Name each concept meaningfully using the language model
7. Embed all concepts, classify local geometry, place on manifold
8. Save manifold.pkl

## Two Model Pipeline

- **Ingestion model** (TinyLlama 1.1B) — fast, cheap, runs overnight.
  Summarizes and names concepts. Doesn't need to be smart, just extract and compress.
- **Inference model** (Mistral 7B) — follows instructions, reasons correctly.
  Answers queries from manifold context. Swap for any model with API access.

## Stack

- `spaCy` — concept extraction
- `ollama` + `nomic-embed-text` — embeddings, no PyTorch dependency
- `ollama` + `tinyllama` / `mistral` — summarization and inference
- `scikit-learn` — KMeans clustering
- `numpy` — geometry and distance calculations
- `matplotlib` + `imageio` — manifold visualization and animation

## Getting Started

```bash
# Setup
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Pull models
ollama pull tinyllama
ollama pull mistral
ollama pull nomic-embed-text

# Add your source documents as .txt files
# Update the sources list in ingest.py

# Ingest
python3.12 ingest.py

# Query
python3.12 main.py
```

## Results

See `manifold_test_results.md` for full results.

**Bare Mistral 7B — no manifold:**
> "I don't have real-time data on SpaceX's Q1 2026 finances."

**Mistral 7B with Knowledge Manifold — SpaceX S-1 filed 18 days prior:**
> "Revenues of `$4,694` million and losses from operations of `$(1,943)` million."

The model didn't get smarter. It got a map.

## The Bigger Vision
`Disk → Ball → Product Manifold → Incremental Ingestion → Manifold of Manifolds`

- New documents add neighborhoods — the manifold grows without rebuilding
- Active manifold — the model writes back, useful inferences become new nodes
- Partitioned manifolds — domain shards loaded on demand, tiny memory footprint
- Model agnostic — swap the inference model without touching the manifold
- C++ implementation planned — `.km` file format, native binary, no Python

## Iterations

- `iterations/poincare_disk.md` — 2D disk, hardcoded domains
- `iterations/poincare_ball.md` — 3D ball, auto-clustering
- `iterations/product_manifold.md` — H² × S¹ × ℝ, geometry classification

## References

- Poincaré Embeddings: [arXiv:1705.08039](https://arxiv.org/abs/1705.08039) — Poincaré Embeddings for Learning Hierarchical Representations
- LAPHA: [arXiv:2602.09375](https://arxiv.org/abs/2602.09375) — Latent Poincaré Shaping for Agentic Reinforcement Learning
- Memento: [arXiv:2508.16153](https://arxiv.org/abs/2508.16153) — Fine-tuning LLM Agents without Fine-tuning LLMs