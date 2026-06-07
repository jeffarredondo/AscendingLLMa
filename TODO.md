# Knowledge Manifold — Next Weekend TODO

## Immediate Fixes
- [ ] Verbose flag — dev mode vs production output
- [ ] Response length limit — 3 sentences max in prompt
- [ ] Deterministic ingestion — seed random sampling so same corpus = same manifold
- [ ] Cache summaries — don't re-summarize concepts across runs
- [ ] Two model pipeline explicit — TinyLlama for ingest, Mistral for inference, both configurable

## Architecture
- [ ] Incremental ingestion — add documents without full rebuild
  - New concepts join existing neighborhoods or spawn new ones
  - Re-place only affected concepts
  - Manifold grows, never rebuilds from scratch
- [ ] Active manifold — model can write back to manifold
  - Useful inferences become new nodes
  - Navigated paths strengthen associations
  - Associative memory that evolves through use
- [ ] Intrinsic dimensionality for n_concepts
  - Let the corpus decide how many concepts it needs
  - TwoNN estimator or similar
  - No hardcoded n_top=60
- [ ] Dynamic SN geometry growth
  - New geometry components emerge when data demands them
  - Emerges naturally from incremental ingestion

## Testing
- [ ] Mistral 7B stable manifold test with SpaceX S-1
- [ ] Intelligent Investor ingestion
- [ ] Cross-domain query: SpaceX + Intelligent Investor together
- [ ] Baseline comparison — bare Mistral vs manifold Mistral

## Rename
- [ ] Rename repo to Knowledge Manifold
- [ ] Update README

## Future (C++ / CondAscendingLLMa)
- [ ] Model as input parameter not hardcoded
- [ ] .km manifest file format
- [ ] Native binary, no Python
- [ ] llama.cpp direct integration