# AscendingLLMa 🦙⬆️

A hyperbolic knowledge system where geometry IS the knowledge.
Concepts live as coordinates on a manifold. Navigation replaces search.
The LLM is just the voice.

## The Idea

Traditional RAG finds the needle in the haystack.
AscendingLLMa is a snow globe — you hold the knowledge and navigate it.

- **Position is derived**, not learned or searched
- **The manifold is the model**. The LLM is the synthesizer.
- **No fine tuning. No training. Just geometry.**

Inspired by the Memento paper (arXiv:2508.16153) and Poincaré embeddings
for hyperbolic space. Validated empirically with a 1.1B parameter model
that correctly described things it had never been trained on.

## Architecture

```
ingest.py           → chunk → TF-IDF → summarize → name → concepts.json
product_manifold.py → embed → classify geometry → place → manifold.pkl
concepts.py         → fuzzy match queries to manifold concepts
middleware.py       → build grounded prompt from manifold context
main.py             → orchestrate everything, query TinyLlama
```

## The Manifold: H² × S¹ × ℝ

Concepts are placed in one of three geometry components based on
the local structure of their embedding neighborhood:

- **H² (Hyperbolic)** — hierarchy, taxonomy, "is a kind of" relationships
- **S¹ (Spherical)** — cyclical, periodic, recurring themes
- **ℝ (Flat)** — linear, gradient, continuous relationships

Geometry assignment is automatic — Gromov delta hyperbolicity measures
how tree-like each concept's neighborhood is. Low delta → hyperbolic.
High delta → spherical or flat based on eigenvalue signature.

Cross-geometry distances use embedding cosine similarity as a bridge,
so semantically related concepts find each other even across geometry
components.

## Ingestion Pipeline

Feed it any text. Get a knowledge manifold back.

1. Chunk text (2000 chars, 200 overlap)
2. Extract concepts via spaCy NER + noun chunks
3. Score by chunk-level TF-IDF — self contained, no external corpus
4. Summarize each concept from its most information-dense chunks
5. Name each concept using the LLM — generic tokens become meaningful names
6. Write concepts.json
7. Embed, classify geometry, place on manifold, save manifold.pkl

## Stack

- `spaCy` — concept extraction
- `ollama` + `nomic-embed-text` — embeddings (no PyTorch dependency)
- `ollama` + `tinyllama` — summarization and query synthesis
- `scikit-learn` — KMeans clustering
- `numpy` — geometry and distance calculations
- `matplotlib` — manifold visualization

## Results

### Les Misérables (test harness)
TinyLlama alone: Shollublip = French herb from oleander  
With manifold: correctly described elephant justice system ✅

### SpaceX S-1 + Mackay + Veblen + Brandenburg (new information test)
TinyLlama alone: SpaceX IPO in April 2012, valued at $24 billion  
With manifold: South Sea bubble dynamics applied to SpaceX losses,
Brandenburg's law of average as investment framework ✅

## The Bigger Vision
Disk → Ball → Product Manifold → Incremental Ingestion → Manifold of Manifolds

- Load any text onto the manifold
- New documents add neighborhoods, existing ones expand
- Cross-domain bridges emerge from geometry
- Give a small model a manifold → it becomes a domain expert
- No fine tuning. No RAG. Just geometry and a synthesizer.

## Iterations

- `iterations/poincare_disk.md` — 2D disk, hardcoded domains
- `iterations/poincare_ball.md` — 3D ball, auto-clustering
- `iterations/product_manifold.md` — H² × S¹ × ℝ, geometry classification

## References
- Poincaré Embeddings: [arXiv:1705.08039](https://arxiv.org/abs/1705.08039) - Poincaré Embeddings for Learning Hierarchical Representations 
- LAPHA: [arXiv:2602.09375](https://arxiv.org/abs/2602.09375) — Latent Poincaré Shaping for Agentic Reinforcement Learning
- Memento: [arXiv:2508.16153](https://arxiv.org/abs/2508.16153) - Fine-tuning LLM Agents without Fine-tuning LLMs

