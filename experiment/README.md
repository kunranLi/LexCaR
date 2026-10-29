# LexCaR Experiments

## Code

- `run_efficiency_controllability.py` — LexCaR vs LLM+MCP（3 轮 session：效率 / 花费 / 可控性）
- `metrics.py` — P@K / R@K / AP@K / nDCG@K

## Protocol (3 rounds)

1. **Round 1（共用）**：Stage1 MCP 建池 + Stage2 发现法律问题  
2. **Round 2**：焦点偏移 A（新增排序标准）  
3. **Round 3**：焦点偏移 B（再加标准，或改为排除规则）

每轮偏移：

- **LexCaR**：同池改标准 / 排除后重排（MCP=0）  
- **LLM+MCP**：把累计焦点拼进 query，整轮重跑 Stage1  

金标：对 Round1 池用 Jev 自动打「是否涉及该焦点」分，取 Top-G（无需人工标注）。

## Cost

统计并换算：

| 资源 | 默认单价（可改） |
|------|------------------|
| MCP tool call（北大法宝） | ¥0.25 |
| LLM call（DMXAPI 估） | ¥0.005 |
| Jev call（DMXAPI 估） | ¥0.002 |

路径总开销 = Round1（共用）+ Round2..N 各自系统开销。自动标金标的 Jev 单独记账，不计入路径对比。

## Results

产物在 `results/`：

```bash
python -m experiment.run_efficiency_controllability --qid "00000,00002" --pool-size 15 --n-shifts 2
```

可用 `--price-mcp / --price-llm / --price-jev` 调整单价。

## Figures (R1–R3 grouped bars)

```bash
python -m experiment.plot_round_curves --exp-dir experiment/results/exp_eff_ctrl_n50_combined
```

Outputs under `figures/`:

- `fig_combined.png` — stacked (a) controllability P@K + (b) cost + (c) latency
- `fig_controllability_rounds.png` — controllability (mean P@K)
- `fig_cost_cumulative.png` — cumulative CNY (single)
- `fig_latency_cumulative.png` — cumulative seconds (single)
