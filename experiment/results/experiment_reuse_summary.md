# LexCaR candidate-pool reuse experiment

This retrospective experiment uses existing session logs with at least one Stage 3 refinement. It compares the recorded LexCaR workflow with an estimated full-re-run workflow that repeats the same Stage 1 MCP calls for every refinement round. The estimate is for MCP calls only; token cost and wall-clock latency were not recorded consistently.

| Session | Pool | Refinements | Initial MCP | LexCaR MCP | Full re-run (estimated) | Calls saved | Saving | Base Top-10 overlap after refinements |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| session_20261014_114113.json | 50 | 2 | 4 | 4 | 12 | 8 | 66.7% | 4/4 / 10 |
| session_20261014_130038.json | 50 | 2 | 8 | 8 | 24 | 16 | 66.7% | 7/6 / 10 |
| session_20261014_133349.json | 33 | 1 | 9 | 9 | 18 | 9 | 50.0% | 9 / 10 |

## Aggregate

Across these sessions, LexCaR reused the same candidate pool for each refinement. The mean initial MCP-call count was 7.0, and the mean estimated MCP-call saving over the recorded refinement rounds was 61.1%. The overlap counts show that reranking changed the Top-10 composition while retaining the same pool.

## Limitation

The current machine could not connect to the configured MCP endpoints during a fresh run, so the full-re-run column is a controlled call-count estimate rather than a new online measurement. No issue-specific qrels are stored in the repository; therefore this artifact does not claim NDCG or Issue Coverage improvements.
