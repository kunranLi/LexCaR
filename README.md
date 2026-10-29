# LexCaR

**User-Centered Legal Case Retrieval with Editable Ranking Criteria**

Legal case retrieval helps legal professionals find previous cases that inform their analysis of a new case. However, a user’s purpose may change as they examine retrieved judgments, so the target cases they consider relevant can change within the same session. Existing methods either keep ranking criteria fixed, unable to follow the user’s evolving focus, or re-run the whole loop at every focus shift, which drives up latency and cost.

LexCaR is a user-centered legal case retrieval system with editable ranking criteria. It offers three properties:

- **Issue-aware guidance.** LexCaR surfaces the atomic legal issues attended to in prior judgments and assesses whether each holds under the query facts to suggest directions for fine-grained retrieval.
- **Direct control.** Users can add or remove criteria, adjust weights, and define exclusion rules to steer the ranking toward their preferred angle.
- **Efficiency.** LexCaR builds a reusable candidate pool once and re-ranks it, so each refinement avoids replanning and external retrieval, staying cheaper and faster than restarting the loop.

<p align="center">
  <img src="figures/framework.png" alt="LexCaR framework" width="720"/>
</p>

## Pipeline

- **Stage 1 — Candidate pool construction**: Agentic retrieval over MCP case databases (e.g. PKULaw), driven by a replaceable retrieval skill, to build a reusable pool of globally similar cases.
- **Stage 2 — Issue discovery and query assessment**: Preset-criteria Jev reranking, extraction of atomic legal issues from top judgments, and assessment of whether each issue holds / is decidable under the query facts.
- **Stage 3 — User-steered reranking**: Edit criteria and weights, define exclusion rules, and re-rank the same pool in-session (no replanning or external retrieval).
- **Web UI** and CLI demos for interactive sessions
- **Experiments** comparing LexCaR vs. full agentic re-retrieval on controllability, cost, and latency

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

If you use LexCaR in your research, please cite:

```bibtex
@inproceedings{lexcar2027,
  title     = {{LexCaR}: User-Centered Legal Case Retrieval with Editable Ranking Criteria},
  booktitle = {Proceedings of the 49th European Conference on Information Retrieval ({ECIR} 2027), Demo Track},
  year      = {2027},
  note      = {Code and live demo: \url{https://github.com/kunranLi/LexCaR}},
}
```

## License

This project is released under the [MIT License](LICENSE).

Third-party case databases (PKULaw, Wolters Kluwer, LeCaRDv2) remain under their respective terms; you must obtain your own access credentials and comply with their licenses.
