# LexCaR

**User-Centered Legal Case Retrieval with Editable Ranking Criteria**

LexCaR is a three-stage legal case retrieval system that builds a candidate pool once, then lets users edit ranking criteria and exclusion rules to re-rank the same pool—without re-querying the case database.

<p align="center">
  <img src="figures/framework.png" alt="LexCaR framework" width="720"/>
</p>

## Features

- **Stage 1 — Candidate pool**: LLM-planned agentic retrieval over MCP case-search tools (e.g. PKULaw / Wolters Kluwer)
- **Stage 2 — Experience ranking**: Multi-dimension Jev scoring, checklist extraction, and fact confirmation
- **Stage 3 — User follow**: Add/edit ranking dimensions, adjust weights, or exclude case types; re-rank in-pool (0 MCP calls)
- **Web UI** and CLI demos for interactive sessions
- **Experiments** comparing LexCaR vs. full LLM+MCP re-retrieval on efficiency, cost, and controllability

## Repository layout

```
LexCaR/
├── pipeline.py              # CLI entry (stages 1/2/3/all)
├── demo_interactive.py      # Interactive terminal demo
├── web_server.py            # Local web UI
├── config.py                # Settings loaded from .env
├── clients/                 # LLM, Jev, MCP, skills
├── stages/                  # Stage1 / Stage2 / Stage3
├── skills/                  # Retrieval skill prompts (SKILL.md)
├── mcp/                     # MCP server endpoint configs
├── data/                    # LeCaRDv2 loader (+ optional local data)
├── experiment/              # Efficiency / controllability scripts
├── web/                     # Frontend
└── figures/                 # Architecture & UI assets
```

## Requirements

- Python 3.10+
- API access:
  - [DMXAPI](https://www.dmxapi.cn) for LLM + Jev (`DMX_API_KEY`)
  - Optional MCP tokens for PKULaw / Wolters Kluwer case search

## Quick start

```bash
git clone https://github.com/kunranLi/LexCaR.git
cd LexCaR
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then fill in DMX_API_KEY and optional MCP tokens
```

### Run CLI

```bash
# Built-in demo query (full pipeline)
python pipeline.py --stage all --demo

# Custom case description
python pipeline.py --stage all --query "你的案情描述……"

# Interactive Stage3 refinement
python demo_interactive.py
```

### Run Web UI

```bash
python web_server.py
# open http://127.0.0.1:8765 (default; see --help)
```

### Optional: LeCaRDv2 sample queries

Local `query.json` / `candidates.jsonl` are **not** shipped (size / license). Place them under `data/lecardv2/` if you want `--qid` support. See [data/lecardv2/README.md](data/lecardv2/README.md).

Official dataset: [THUIR/LeCaRDv2](https://github.com/THUIR/LeCaRDv2).

## Experiments

See [experiment/README.md](experiment/README.md). Aggregate summaries and paper figures are under `experiment/results/`.

```bash
python -m experiment.run_efficiency_controllability --qid "00000,00002" --pool-size 15 --n-shifts 2
python -m experiment.plot_round_curves --exp-dir experiment/results/exp_eff_ctrl_n50_combined
```

## Configuration

All knobs live in `.env` (copy from `.env.example`): models, pool size, timeouts, Stage2/3 thresholds, and prompts. MCP endpoints are JSON files under `mcp/` and use `${PKULAW_MCP_TOKEN}` / `${WK_MCP_TOKEN}` placeholders—never commit real tokens.

## Citation

If you use LexCaR in your research, please cite this repository:

```bibtex
@misc{lexcar2026,
  title        = {LexCaR: User-Centered Legal Case Retrieval with Editable Ranking Criteria},
  author       = {kunranLi},
  year         = {2026},
  howpublished = {\url{https://github.com/kunranLi/LexCaR}},
}
```

## License

This project is released under the [MIT License](LICENSE).

Third-party case databases (PKULaw, Wolters Kluwer, LeCaRDv2) remain under their respective terms; you must obtain your own access credentials and comply with their licenses.
