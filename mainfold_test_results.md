# Knowledge Manifold — Test Results

## Test Setup

**Date:** June 7, 2026  
**Model:** Mistral 7B (inference), TinyLlama 1.1B (ingestion)  
**Hardware:** 2019 Intel MacBook Pro (SharkBait)  
**Manifold sources:**
- SpaceX S-1 (filed May 20, 2026) — 1.48M characters
- Mackay "Extraordinary Popular Delusions" (1841) — 1.65M characters  
- Veblen "Theory of Business Enterprise" (1904) — 6.4K characters
- Brandenburg "Profitable Stock Exchange Investments" (1800s) - 62,193 characters

**Key detail:** The SpaceX S-1 was filed 18 days before this test.
Mistral 7B has zero training data on it.

---

## Baseline: Bare Mistral 7B (no manifold)

**Q: What was SpaceX's revenue and loss in Q1 2026?**
> "I don't have real-time data or the ability to predict future financials
> for companies like SpaceX. The revenue and loss figures for SpaceX's Q1
> 2026 would not be available until after that quarter has ended."

**Q: What is SpaceX's valuation for the IPO?**
> "As of my last update, SpaceX has not yet announced plans for an initial
> public offering (IPO). The private company is owned and led by Elon Musk.
> Until there is a formal announcement, it is impossible to determine the
> valuation for a potential IPO."

**Q: What happened with xAI and SpaceX?**
> "There is no known direct connection between xAI and SpaceX. Elon Musk
> co-founded OpenAI in 2015..."

**Result:** Model correctly identifies it doesn't know. All answers stale or wrong.

---

## With Knowledge Manifold

**Q: What was SpaceX's revenue and loss in Q1 2026?**
> "According to the fact 'q1 financial results (2026)': SpaceX's financial
> results for the three months ended March 31, 2026 included:
> * Revenues of $4,694 million
> * Losses from operations of $(1,943) million.
> There is no information about net income or profit in Q1 2026."

**Q: What is SpaceX's valuation for the IPO?**
> "The provided facts do not mention or imply that SpaceX has had an Initial
> Public Offering (IPO) with a specific valuation. The fact 'value: The table
> below reflects stock options and RSUs issued to our NEOs during the 2025
> fiscal year...' suggests it may be related to equity valuation, but there
> is no explicit mention of SpaceX's IPO or valuation."

**Q: What happened with xAI and SpaceX?**
> "Based on the provided facts, it appears that xAI (formerly known as
> Anthropic PBC) has an agreement with SpaceX to provide access to compute
> capacity through COLOSSUS and COLOSSUS II..."
> *(Note: entity confusion due to ingestion model limitations — known issue)*

**Q: Is SpaceX a good investment given its losses?**
> "The provided facts do not directly address whether SpaceX is a good
> investment given its losses... Therefore, it is impossible to determine
> if SpaceX is a good investment based solely on these facts."

**Q: Has market mania affected SpaceX's valuation?**
> "Based on the provided facts, there is no explicit statement that mentions
> a market mania affecting SpaceX's valuation. The reference to the 'gold
> delusion Cassandra' and 'aristocratic prophetic control' discuss historical
> examples of market speculation and delusions, but they do not directly
> apply to SpaceX..."

---

## Analysis

| Query | Bare Mistral | With Manifold |
|-------|-------------|---------------|
| Q1 2026 revenue & loss | ❌ No data | ✅ Exact figures |
| IPO valuation | ❌ "No IPO announced" | ✅ Honest — not in facts |
| xAI + SpaceX | ❌ Stale/wrong | ⚠️ Partially correct, entity confusion |
| Good investment? | ❌ Can't answer | ✅ Honest — insufficient facts |
| Market mania? | ❌ Can't answer | ⚠️ Found historical parallel, couldn't bridge |

## Key Finding

The model did not get smarter. It got a map.

A 7B parameter model with no fine-tuning, given access to a Knowledge
Manifold derived from a document filed 18 days prior, correctly answered
factual questions that the same model cannot answer without the manifold.

The geometry navigated to the right facts. The model synthesized them.
No training. No fine-tuning. No RAG pipeline. Just geometry and a voice.

## Known Limitations

- **Entity disambiguation:** TinyLlama conflates similar entities (xAI vs
  Anthropic) during ingestion. Fix: disambiguation pass or smarter ingestion model.
- **Investment reasoning:** "Is SpaceX a good investment" didn't pull Q1
  loss data — fuzzy matching missed the connection. Fix: better query
  extraction or lower threshold for financial queries.
- **Single source:** Manifold only has SpaceX's own narrative. Bear case
  requires external analyst reports and industry context.
- **Ingestion variance:** Each TinyLlama ingestion run produces slightly
  different concept names and descriptions. Fix: deterministic ingestion,
  summary caching.

## Next Steps

See TODO.md