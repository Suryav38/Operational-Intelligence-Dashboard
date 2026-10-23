# TD Fab Capacity & Wafer Loading Planner

A capacity-planning dashboard for a **simulated network of two technology-development (TD) semiconductor fabs**,
with an AI planning copilot. All data is synthetic and illustrative; it is not any company's data.

Live app: https://operational-ai-intelligence.streamlit.app

## What it answers

- **Can the fabs run the 2027 wafer-start plan?** Tool-group × quarter loading heatmap for both fabs, the
  constrained cells, the programs driving them, and how many tools each constraint needs.
- **How much should we trust the requests?** A forecast model compares past requested vs actual wafer starts
  per program, corrects each program's bias, and backtests the correction (error falls from 10.8% to 3.4%).
  With corrected demand, at-risk cells fall from 17 to 8.
- **What happens if the plan changes?** A scenario lab for route changes, pull-ins, wafer-start changes,
  tool delays, availability changes and moving load between fabs.
- **How is the fab running today?** 13 weeks of Q3 2026 performance from a SimPy simulation: wafer starts
  and outs, moves, WIP, cycle time (hot vs normal lots), days per mask layer, X-factor, line yield,
  cycle-time breakdown and SEMI E10 tool states.

## Where the AI comes in

| AI job | What it does | Guardrail |
|---|---|---|
| Read planning notes | Turns a process change notice, program request or equipment update into structured plan changes | A planner reviews and edits the changes before the model runs; ambiguities are listed as open questions |
| Planning copilot | Answers questions by calling the capacity model, forecast model, read-only SQL and document search | It never does arithmetic; every number comes from a tool, and each answer shows its steps |
| Leadership readout | Drafts a short readout of a scenario for the quarterly planning review | Written only from the scenario results |

## Models

- **Capacity:** required hours/week = Σ WSPW × passes × minutes per pass ÷ 60; available hours/week =
  tools × 168 × production availability; loading = required ÷ available; plan below an 85% ceiling.
- **Forecast:** actual starts = b × requested starts per program (least squares through the origin), fitted on
  2024Q1–2025Q4 and backtested on 2026Q1–Q3.
- **Simulation:** SimPy discrete-event model of each fab: lots of 25 wafers on re-entrant routes, hot-lot
  priority, batch furnaces, sampled metrology, PM / breakdowns / engineering time, transport and
  engineering holds.

## Validation

- The capacity model's Q3 2026 loading matches the simulated utilization within 1.5 points on average.
- Little's Law holds in the simulation: WIP ≈ throughput × cycle time (235 vs 237 lots at Fab A).

## Run locally

```bash
pip install -r code/requirements.txt
python code/data_generation.py   # optional: rerun the simulation and rebuild all data (~15 s)
python code/self_check.py        # offline check of the models and copilot tools, no API key needed
streamlit run code/apprag.py     # add OPENAI_API_KEY to .env to enable the AI features
```

Optional setting: `OPENAI_MODEL` (default `gpt-4o-mini`).

## Files

| File | Purpose |
|---|---|
| `code/fab_config.py` | Every assumption: tool groups, routes, tools, plans |
| `code/fab_sim.py` | SimPy fab simulation |
| `code/data_generation.py` | Runs the simulation, writes tables, documents and the SQLite database |
| `code/fab_capacity.py` | Capacity model and scenario engine |
| `code/forecast_model.py` | Request-bias regression and backtest |
| `code/fab_data.py` | Performance metrics and validation checks |
| `code/assistant.py` | AI note reading, tool-calling copilot, readout drafting |
| `code/apprag.py` | Streamlit dashboard |
