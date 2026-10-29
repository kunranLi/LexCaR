# LexCaR

**User-Centered Legal Case Retrieval with Editable Ranking Criteria**

Legal case retrieval helps professionals find prior judgments that inform a new case. Within a session, the user’s focus often shifts; fixed ranking criteria cannot follow that change, while re-running the full agentic loop at every shift raises latency and cost.

LexCaR addresses this with three properties:

- **Issue-aware guidance.** Surfaces atomic legal issues attended to in prior judgments and assesses whether each holds under the query facts, suggesting directions for fine-grained retrieval.
- **Direct control.** Users can add or remove criteria, adjust weights, and define exclusion rules to steer ranking toward their preferred angle.
- **Efficiency.** Builds a reusable candidate pool once and re-ranks it, so each refinement avoids replanning and external retrieval.

<p align="center">
  <img src="figures/framework.png" alt="Method of LexCaR" width="720"/>
</p>

## Method

1. **Candidate pool construction.** An agentic retriever with a built-in (replaceable) retrieval skill queries external case databases via MCP and grows the pool until the target size.
2. **Issue discovery and query assessment.** Jev re-ranks the pool under preset criteria (legal relationship, key facts, precedential value); an LLM extracts atomic issues from top judgments; Jev then labels each issue on the query as yes / no / unknown.
3. **User-steered reranking.** Natural-language criteria and exclusion rules are compiled into Jev questions; the same pool is re-scored under editable weights—without new MCP retrieval.

## Quick start

```bash
git clone https://github.com/kunranLi/LexCaR.git
cd LexCaR
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # set DMX_API_KEY; optional PKULAW_MCP_TOKEN / WK_MCP_TOKEN
```

```bash
python pipeline.py --stage all --demo          # CLI
python demo_interactive.py                     # interactive Stage 3
python web_server.py                           # Web UI (default http://127.0.0.1:8765)
```

Optional LeCaRDv2 files for `--qid`: see [data/lecardv2/README.md](data/lecardv2/README.md).

## Experiments

Paper-style efficiency / controllability comparison (LexCaR vs. full agentic re-retrieval): [experiment/README.md](experiment/README.md). Aggregates and figures are under `experiment/results/`.

## Citation

```bibtex
@misc{lexcar2026,
  title  = {LexCaR: User-Centered Legal Case Retrieval with Editable Ranking Criteria},
  author = {kunranLi},
  year   = {2026},
  url    = {https://github.com/kunranLi/LexCaR}
}
```

## License

[MIT](LICENSE). Third-party case databases and datasets remain under their own terms.
