# TD Fab — Performance & Planning (v4)

An independent interview prototype for a fictional two-fab Technology Development network. **All fab data is synthetic.** This does not reproduce a Micron internal dashboard.

## Three views

1. **Fab performance** opens first. Its tabs cover production overview, equipment health, historical start attainment and machine OEE.
2. **Capacity & scenarios** combines the quarterly capacity outlook with manual demand scenarios. Change demand by percentage or enter an exact target WSPW for every selected quarter; optionally test availability recovery.
3. **Lot flow & cycle time** provides cycle-time decomposition, priority comparisons, cycle-time distributions, WIP aging and a downloadable lot explorer.

AI features are paused: there is no AI page, request reader, API status, AI explanation or external model call in the application entry point. Legacy AI modules and their tests are retained for a later integration. No API key is required.

## Filters and supported detail

- Site: Combined, Fab A or Fab B.
- Program family: DRAM, NAND, HBM or Experiments.
- Program names: DRAM Node A, DRAM Node B (EUV), 3D NAND Gen C, HBM test vehicle and Short-loop experiments, restricted to the selected fab's qualifications.
- Execution: HOT/NORMAL priority and a reporting-week range.
- Equipment health: equipment-group selection.
- Lot explorer: completed during the period, WIP at the period end, or started during the period.

The original data has no product/SKU codes, recipe-level event history, individual machine IDs or process-step timestamps. Program → fab → priority → lot is the finest supported execution hierarchy. The source route table supports equipment-group workload tracing, but not tracking the current operation of a lot.

**Capacity filters are a program focus, not a change to network demand.** Equipment loading, risks, gaps and scenarios retain every program's workload. The selected programs focus the starts chart and workload contribution chart; the latter groups other programs explicitly. Equipment health covers the entire Q3 2026 and follows only site/equipment filters. Historical attainment follows site/program filters over 2024Q1–2026Q3, independent of execution weeks and lot priority. These scopes are labelled on their tabs.

## Run

Python 3.11 or 3.12 is supported. From the project root:

```bash
python -m pip install -r requirements.txt
python code/self_check.py
python -m streamlit run code/apprag.py
```

For Streamlit Community Cloud, use **code/apprag.py** as the entry point. The supplied data is ready to use; do not regenerate it merely to start the app.

## Machine OEE

The existing fab simulation contains group time states, not sufficient measurements for individual machine OEE. Its equipment charts are labelled **Equipment time allocation** and **Production loading**.

The Machine OEE tab adds a working calculator with two input modes:

- **Worked example:** EXAMPLE-01 is explicitly independent of the fab simulation. Editable inputs default to 480 planned minutes, 360 run minutes, 60 seconds ideal cycle, 300 processed units and 294 first-pass good units. A = 75%, P = 83.333%, Q = 98%, OEE = 61.25%.
- **Upload machine records:** supply the columns in `templates/machine_oee_template.csv`. No uploaded data is written back to the project or used by the fab capacity model. All records are evaluated in the uploaded file's own scope, independent of sidebar filters.

| Field | Meaning |
|---|---|
| machine_id | Individual machine identifier |
| period | Unique, non-overlapping reporting block for this machine |
| recipe_id | Single recipe / ideal cycle standard for this record |
| planned_minutes | Planned production time under an agreed scheduling policy |
| run_minutes | Actual run time within that planned period |
| ideal_cycle_seconds | Ideal seconds per processed unit for that recipe |
| total_count | All units processed in the period |
| good_count | Units that pass first time, without rework |

Availability = run / planned time. Performance = ideal time × total count / run time. Quality = good / total count. OEE = A × P × Q = good count × ideal time / planned time. Unit conversions are explicit. Split mixed recipes into separate non-overlapping periods; do not reuse the full shift's planned time for every recipe. Counts must be whole, finite and non-negative; good ≤ total; run ≤ planned; performance must not exceed 100%. For zero output, quality is undefined and OEE is zero. The interface does not average unlike machines' OEE.

Reference: [Vorne's OEE formulas](https://www.oee.com/calculating-oee/).

## Execution definitions

The source simulation covers **29 June–27 September 2026**, after warm-up. Some lots began earlier and completed within this window.

- Weekly starts: input wafers from lots started in that week, following selected filters.
- Weekly completions: surviving output wafers from lots completed in that week.
- WIP: original input-wafer equivalents for lots active at each week end; reconstructed from minute-resolution start/end timestamps. The all-program totals reconcile exactly to the original weekly CSV.
- Cycle time: completed-lot mean, weighted by the underlying lots across both fabs. Short loops are excluded unless enabled; they remain in flow/WIP/yield when selected.
- Yield: output divided by input for the same completed cohort; wafer survival, not electrical die yield or machine first-pass quality.
- Weekly cycle time: weighted across that week's completed lots, not an unweighted average of fab averages.
- Lot explorer: completed/start cohorts follow selected weeks. WIP age is reconstructed at the selected cutoff. Route progress is shown only at the final simulation snapshot because intermediate progress history is absent. Future completions/output are hidden when reviewing an earlier cutoff.
- Queue + hold share: measured from the selected completed cycle-time cohort. The default full-flow population is approximately **40.0%**, not >50%. Including transport raises non-processing share to approximately 63.6%. Engineering holds can be necessary development work; the dashboard does not classify all elapsed time as avoidable waste.

Little's Law is not claimed as an exact reconciliation of the current finite-window KPI cards. Snapshot WIP, completed cohorts, short-loop exclusions and boundary effects must be aligned for such an analysis.

## Capacity definitions

Required hours/week = Σ starts/week × passes/wafer × minutes/pass × sampling ÷ 60. Available hours/week = installed resources × 168 × production availability. Loading = required / available hours.

The 85% limit is a demo planning policy. Equipment releases contribute for the full stated quarter. Spare capacity at one fab does not cancel a constraint at the other. Additional-tool counts are aggregate requirements at that limit, before qualification and financial review. There are no equipment prices, costs or ROI estimates.

Historical performance is not rerun when a future demand scenario changes. The capacity model does not predict scenario queues, cycle times, due dates or savings.

`fab_config.py` supplies routing, equipment, availability and arrival assumptions. `plan_2027.csv` supplies requested demand. Equipment/route CSVs are generated snapshots of the configuration. The original 11 CSVs, 15 text documents and SQLite database are preserved unchanged.

## Update your Mac/GitHub project

Download **td-fab-planner-v4.zip** to Downloads. In Cursor's terminal:

```bash
unzip -q -o "$HOME/Downloads/td-fab-planner-v4.zip" -d "$HOME/Downloads/td-planner-v4"
ditto "$HOME/Downloads/td-planner-v4/td-fab-planner-update" "$HOME/Desktop/Operational-Intelligence-Dashboard"
cd "$HOME/Desktop/Operational-Intelligence-Dashboard"
git status
git add -A
git commit -m "Lead with fab performance and add program filters and machine OEE calculator"
git push origin main
```

Refresh after Streamlit finishes rebuilding from the new commit. Keep the entry point at `code/apprag.py`.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See `VALIDATION.md` for results and `DEMO_GUIDE.md` for the interview walkthrough.
