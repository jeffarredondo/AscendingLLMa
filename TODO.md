# Knowledge Manifold — Next Weekend TODO

## Immediate Fixes
- [ ] Verbose flag — dev mode vs production output
- [ ] Deterministic ingestion — seed random sampling so same corpus = same manifold
- [ ] Entity disambiguation pass — after ingestion, verify entities aren't conflated
- [ ] Cache summaries — don't re-summarize concepts across runs. Add information to manifold
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
- [ ] Add industy and competitor information 
- [ ] Add open source S1 analysis from bull and bear perspectives
- [ ] Cross-domain query: SpaceX + open source analysis
- [ ] Baseline comparison — bare Mistral vs manifold Mistral

## Scaling
- [ ] Scaling test — measure file size vs concept count vs query quality
- [ ] Manifold sharding — split large knowledge bases into domain shards (.km files)
- [ ] Lazy loading — load only relevant shards based on query
- [ ] Shard router — query embedding similarity to manifold metadata decides which shards to load
- [ ] Memory budget — define max manifolds in memory at once

## New Repo
- [ ] Create repo for Knowledge Manifold - C++ implementation

## Future (C++)
- [ ] Model as input parameter not hardcoded
- [ ] .km manifest file format
- [ ] Native binary, no Python
- [ ] llama.cpp direct integration