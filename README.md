# TD Planning — Capacity & Operations

A focused Streamlit interview prototype for a fictional network of two Technology Development fabs. **All data is synthetic.** This is an independent design based on planning responsibilities, not a reproduction of a Micron internal dashboard.

## Three pages

1. **Capacity plan:** requested wafer starts, full-year equipment loading, planned equipment releases, program contribution, and a selected-quarter exception register.
2. **Scenario review:** change a program's wafer starts, optionally test recovered equipment availability, review baseline versus scenario, and export the comparison and decision brief. Optional AI reads a written request into reviewable fields.
3. **Fab performance:** 13 weeks of historical simulation results: weekly starts and completions, WIP, cycle-time composition, equipment queues and wafer survival yield.

The presentation focuses on EUV, etch and deposition. **All programs and equipment groups stay in the calculations.** Exception counts include every equipment group, and the scenario chart surfaces exceptions outside the focus groups.

## Run

Use Python 3.11 or 3.12 (verified here on 3.12). From the project root:

```bash
python -m pip install -r requirements.txt
python code/self_check.py
python -m streamlit run code/apprag.py
```

The supplied CSVs and SQLite database are ready to use; data regeneration is not required. For Streamlit Community Cloud, select `code/apprag.py` as the entry point and Python 3.12. Do not leave an old deployment pointed at a different entry point.

### Optional AI

Copy `.env.example` to `.env` and set your own `OPENAI_API_KEY`. Alternatively use Streamlit Cloud app secrets:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-4o-mini"
```

AI is invoked only when **Extract request** is selected. It uses the OpenAI SDK's typed structured output to read one percentage change in wafer starts. It returns an exact source quote and open questions. Missing or ambiguous fields block copying to the scenario controls. A person reviews the fields and explicitly selects **Compare scenario**. The language model never calculates capacity, changes source data, or approves a plan.

Without an API key, all capacity, manual scenario, historical performance and export functions still work. No generated AI response is substituted when the service is unavailable.

Supported note example:

> Increase DRAM Node B wafer starts by 10% at Fab A during 2027Q2 only.

This simplified note reader intentionally supports one percentage-based request, with a site and explicit 2027 start and end quarters. Route changes, multiple requests and absolute wafer quantities should use a future extension; ambiguous notes generate open questions.

## What changed in v2

- Replaced five dense tabs and a general chatbot with three focused pages.
- Added a consistent light enterprise layout, explicit scope and units, accessible status labels, and visible synthetic/historical badges.
- Added a focused baseline-versus-scenario workflow, optional availability recovery, traceable workload contribution and downloadable review evidence.
- Added stale-result protection: changing inputs hides old scenario results until recalculated.
- Added a coherent historical production view without inventing targets, OEE, die yield, due dates or live status.
- Simplified AI to one reviewable task and removed vector-store dependencies from the runtime requirements.
- Preserved the complete underlying workload and supplied source data.
- Fixed missing-capacity classification in the underlying constraint helper.

## Data and calculations

The package retains 11 CSV tables, 15 illustrative planning documents and a SQLite copy of the tables. `fab_config.py` remains the source of equipment, routes and timing assumptions. Existing CSV route/equipment tables are generated snapshots; editing them alone does not change all model inputs. `plan_2027.csv` supplies demand.

- Required tool-hours/week = sum of wafer starts/week × visits/wafer × minutes/visit × sampling fraction ÷ 60.
- Available hours/week = installed resources × 168 × production availability.
- Loading = required hours ÷ available hours.
- The default planning limit is 85%, a chosen demo policy, not an asserted Micron standard.
- Equipment additions contribute a full quarter of capacity beginning in their release quarter.
- The selected-quarter exception count includes all modeled installed groups at that site.

Historical operating data covers **29 June–27 September 2026**, after a simulation warm-up. Some lots start before the measurement window because they complete within it. The historical simulation is not rerun when a future loading scenario changes.

### Production metrics

- **Completions/week:** average wafers exiting the simulated route during each reporting week.
- **Full-flow cycle time:** lot-weighted elapsed days of completed full-flow lots; excludes short-loop experiments and incomplete WIP.
- **Average active WIP:** mean end-of-week snapshots in input-wafer equivalents. Each active lot contributes its original 25 wafers; interim wafer losses are not deducted from this snapshot field.
- **Line yield:** surviving output wafers divided by input wafers for the same completed-lot cohort; includes short-loop lots. This is wafer survival, not electrical die yield.
- **Queue time:** average hours waiting for a tool-group visit over the full 13-week window.
- **Transport/dispatch:** includes modeled dispatch delays between operations; not only physical travel.

## More data for a live production deployment

The existing tables are sufficient for this historical interview demo. A live shift-level implementation needs timestamped lot-operation and equipment-state events, current lot locations and quantities, shift calendars, approved starts/output targets with plan versions, and refresh/data-quality monitoring. Due-date adherence needs lot milestones or committed dates. Electrical die yield needs wafer-test results. OEE needs a consistent equipment-state model, ideal rates, relevant good-unit counts and an agreed measurement boundary.

## Scope limits

This is an aggregate quarterly capacity screen, not an exact lot scheduler. It does not estimate scenario queue times, milestone probabilities, cost savings or ROI. Tool groups pool resources; routes and furnace behavior are simplified. Synthetic data generated from shared assumptions can demonstrate internal consistency but cannot validate a real fab. The original request-realization regression remains available in `forecast_model.py` as an advanced reference; the presentation consistently uses requested starts.

## Files to study first

| File | Purpose |
|---|---|
| `code/apprag.py` | Three-page dashboard |
| `code/review_model.py` | Scope-safe review metrics and scenario summaries |
| `code/fab_capacity.py` | Complete capacity and scenario calculations |
| `code/assistant.py` | Optional typed AI request extraction |
| `code/fab_config.py` | Assumptions and resource definitions |
| `DEMO_GUIDE.md` | Five-minute walkthrough and interview questions |

`fab_sim.py`, `data_generation.py`, `fab_data.py` and `forecast_model.py` remain available for deeper study. `python code/data_generation.py` regenerates the original synthetic data and documents and overwrites the current data folder; it is optional.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover capacity arithmetic, scope preservation, quarter boundaries, qualification checks, availability recovery, cohort-based production metrics, typed AI parsing with a mock client, and Streamlit page interactions. Live AI access requires the user's API account and is separate from offline validation.
