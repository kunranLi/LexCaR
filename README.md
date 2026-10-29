# LexCaR
User-Centered Legal Case Retrieval with Editable Ranking Criteria

[![Conference](https://img.shields.io/badge/ECIR-2027%20Demo-blue.svg)](https://github.com/kunranLi/LexCaR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/kunranLi/LexCaR)

## 📖 Introduction

Legal case retrieval helps legal professionals find previous cases that inform their analysis of a new case. However, a user’s purpose may change as they examine retrieved judgments, so the target cases they consider relevant can change within the same session. Existing methods either keep ranking criteria fixed, unable to follow the user’s evolving focus, or re-run the whole loop at every focus shift, which drives up latency and cost.

To address this, we present **LexCaR**, a user-centered legal case retrieval system with three properties. **Issue-aware guidance.** LexCaR surfaces the atomic legal issues attended to in prior judgments and assesses whether each holds under the query facts to suggest directions for fine-grained retrieval. **Direct control.** Users can add or remove criteria, adjust weights, and define exclusion rules to steer the ranking toward their preferred angle. **Efficiency.** LexCaR builds a reusable candidate pool once and re-ranks it, so each refinement avoids replanning and external retrieval, staying cheaper and faster than restarting the loop.

## ✨ Key Features

* **Three-stage pipeline**: Candidate pool construction → issue discovery & query assessment → user-steered reranking, with editable criteria throughout the session.
<p align="center">
  <img src="figures/framework.png" alt="LexCaR framework" width="800">
  <br>
  <em>Figure 1: Method of LexCaR.</em>
</p>

* **Issue-aware guidance**: From top-ranked judgments, LexCaR extracts atomic legal issues and uses Jev to assess whether each issue holds / is decidable under the query facts, suggesting directions for fine-grained retrieval.

* **Direct control**: Users can add or remove ranking criteria (natural-language intents compiled into Jev questions), adjust weights, and define exclusion rules to steer results toward the angle they care about.

* **Efficient in-pool refinement**: A reusable candidate pool is built once via agentic MCP retrieval; each focus shift re-ranks the same pool without replanning or external search—cheaper and faster than restarting the loop.

* **Interactive demos**: Local Web UI (`web_server.py`) and CLI (`pipeline.py` / `demo_interactive.py`) for end-to-end sessions.

## 🔧 Method Overview

### Stage 1: Candidate Pool Construction

Given a query case, LexCaR constructs a candidate pool from external legal case databases via MCP with an agentic retriever. Retrieval is driven by a built-in (replaceable) retrieval skill; the retriever may execute several rounds of search until the target pool size is reached. Users can replace the skill and set search rounds / pool size for customized pools.

### Stage 2: Issue Discovery and Query Assessment

1. **Preset-criteria reranking** — Jev evaluates each candidate with three preset criteria (legal-relationship similarity, key-fact similarity, precedential value) and ranks by a weighted score.
2. **Legal issue extraction** — An LLM analyzes judicial reasoning of top-ranked cases and produces a structured checklist of atomic legal issues.
3. **Query-case assessment** — For each issue, Jev judges whether it is established by the query facts and whether the query text provides sufficient information (yes / no / unknown).

### Stage 3: User-Steered Reranking

Users edit the ranking specification by adding/removing criteria, adjusting weights, and defining exclusion rules. The candidate pool is reused; Jev re-evaluates active criteria at each rerank and LexCaR presents the updated ranking with per-criterion contributions.

## 📊 Main Findings

<p align="center">
  <img src="figures/Discussion.png" alt="LexCaR performance" width="800">
  <br>
  <em>Figure 2: The performance of LexCaR.</em>
</p>

We compared LexCaR and a traditional agentic retriever on the same underlying LLM (GPT-4o-mini), with legal experts running 3 rounds of interactive retrieval over 50 queries. Round 1 shared a candidate pool from PKULaw MCP. In Rounds 2–3, experts proposed focus shifts; LexCaR re-ranked the shared pool, while the agentic baseline re-ran full retrieval. Using experts’ target sets on the Round-1 pool as gold:

* LexCaR aligned top results with the updated gold standard (P@K ≈ 0.60 / 0.51 in R2 / R3).
* The re-run agent, suffering candidate-pool drift, fell to ≈ 0.06–0.08.
* By Round 3, cumulative cost/latency were ¥0.90 / 112.2 s for LexCaR vs. ¥2.52 / 123.4 s for the baseline.

Aggregate tables and plotting scripts: [`experiment/results/`](./experiment/results/) and [`experiment/README.md`](./experiment/README.md).

## 📂 Repository Structure

```text
LexCaR/
├── pipeline.py              # CLI entry (stages 1 / 2 / 3 / all)
├── demo_interactive.py      # Interactive terminal demo
├── web_server.py            # Local web UI
├── config.py                # Settings loaded from .env
├── clients/                 # LLM, Jev, MCP, skills
├── stages/                  # Stage1 / Stage2 / Stage3
├── skills/                  # Retrieval skill prompts (SKILL.md)
├── mcp/                     # MCP server endpoint configs
├── data/                    # LeCaRDv2 loader (+ optional local data)
├── experiment/              # Efficiency / controllability scripts & results
├── web/                     # Frontend
├── figures/                 # Paper figures & UI assets
├── .env.example             # Configuration template
├── requirements.txt
└── LICENSE
```

## 🚀 Getting Started

### Requirements

- Python 3.10+
- [DMXAPI](https://www.dmxapi.cn) key for LLM + Jev (`DMX_API_KEY`)
- Optional MCP tokens for PKULaw / Wolters Kluwer case search

### Installation

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
python pipeline.py --stage all --query "Your case description..."

# Interactive Stage 3 refinement
python demo_interactive.py
```

### Run Web UI

```bash
python web_server.py
# open http://127.0.0.1:8765 (default; see --help)
```

### Optional: LeCaRDv2 sample queries

Local `query.json` / `candidates.jsonl` are **not** shipped (size / license). Place them under `data/lecardv2/` for `--qid` support. See [`data/lecardv2/README.md`](./data/lecardv2/README.md). Official dataset: [THUIR/LeCaRDv2](https://github.com/THUIR/LeCaRDv2).

### Reproduce experiments

```bash
python -m experiment.run_efficiency_controllability --qid "00000,00002" --pool-size 15 --n-shifts 2
python -m experiment.plot_round_curves --exp-dir experiment/results/exp_eff_ctrl_n50_combined
```

### Configuration

All knobs live in `.env` (copy from `.env.example`): models, pool size, timeouts, Stage 2/3 thresholds, and prompts. MCP endpoints under `mcp/` use `${PKULAW_MCP_TOKEN}` / `${WK_MCP_TOKEN}` placeholders—never commit real tokens.

## 📜 Citation

If you use LexCaR in your research, please cite:

```bibtex
@inproceedings{lexcar2027,
  title     = {{LexCaR}: User-Centered Legal Case Retrieval with Editable Ranking Criteria},
  booktitle = {Proceedings of the 49th European Conference on Information Retrieval ({ECIR} 2027), Demo Track},
  year      = {2027},
  note      = {Code and live demo: \url{https://github.com/kunranLi/LexCaR}},
}
```

## 📄 License

This project is released under the [MIT License](./LICENSE).

Third-party case databases (PKULaw, Wolters Kluwer, LeCaRDv2) remain under their respective terms; you must obtain your own access credentials and comply with their licenses.
