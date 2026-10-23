# Operational Intelligence Dashboard

A manufacturing operations dashboard with an AI assistant that answers plant questions in plain English.
It runs on a **simulated three-line assembly plant** (synthetic 2024 data plus a Q1 2025 demand forecast).

Live app: https://operational-ai-intelligence.streamlit.app

## What it does

- **Dashboard:** OEE broken into availability × performance × quality, plan attainment, scrap by line,
  supplier and shift, a downtime Pareto by machine, failure-only MTBF/MTTR, and an automatic watch list
  of segments that stand out.
- **Capacity planning:** checks whether each line can meet the Q1 2025 forecast and tests levers
  (demand growth, availability, scrap, weekend shifts, fixing a specific machine).
- **AI assistant:** answers questions such as *"Can Line 2 meet Q1 demand, and what's the cheapest way to
  close the gap?"* by calling tools, then explains the result and shows every step it took.

## How the AI assistant works

```
question → GPT (tool calling) → picks tools and inputs
             ├─ KPI and breakdown functions (same pandas code as the charts)
             ├─ capacity model (scenario levers)
             ├─ read-only SQL on the SQLite plant database
             └─ document search (SOPs, maintenance tickets, shift reports, weekly summaries)
         → answer with sources, plus a "How this answer was built" panel
```

Design rule: **the model never does the math.** Numbers come from tools; the model chooses the tool,
fills in the inputs and explains the result. SQL runs read-only, one SELECT at a time, capped at 50 rows.
The document index is built in memory on the first question, so it always matches the documents.

## Key formulas

| Metric | Definition |
|---|---|
| Availability | runtime ÷ planned time |
| Performance | (units × ideal cycle time) ÷ runtime |
| Quality | good units ÷ units produced |
| MTBF | operating hours ÷ failures (mechanical + sensor; jams and changeovers excluded) |
| MTTR | mean repair minutes per failure |
| Capacity loading | required hours ÷ (scheduled hours × availability × performance) |
| Required hours | Σ (demand ÷ quality) × ideal cycle time |

## The simulated plant

- 3 lines × 15 machines, Monday–Friday, three 8-hour shifts, four SKUs per shift in 2-hour runs.
- Machine stops are generated first; line downtime is derived from them, so the downtime log and the
  production table agree.
- The simulation includes three operational problems for the dashboard and assistant to surface:
  an unreliable machine, a supplier quality issue on one line, and a weaker night shift.

## Run locally

```bash
pip install -r code/requirements.txt
python code/data_generation.py   # regenerate the data (optional, already included)
python code/build_sqlite.py      # rebuild the SQLite database
python code/self_check.py        # offline check, no API key needed
streamlit run code/apprag.py     # needs OPENAI_API_KEY in .env for the assistant
```

Optional: set `OPENAI_MODEL` (default `gpt-4o-mini`).
