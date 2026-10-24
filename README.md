# TD Fab — Performance & Planning (v5)

An independent interview prototype for a fictional two-fab Technology Development network. **All fab data is synthetic.** This does not reproduce a Micron internal dashboard.

## Four views

1. **Fab performance** opens first. Its tabs cover production overview, equipment health, historical start attainment and machine OEE.
2. **Demand outlook** shows all four quarters of the 2027 plan, with family/program/route breakdowns, fab comparisons, a fab/program ramp heatmap and requested-versus-expected starts. A separate tab exposes historical backtesting.
3. **Capacity & scenarios** combines the capacity outlook with demand scenarios over a quarter range or a start/end week range. Change demand by percentage or exact target WSPW; optionally test availability recovery. Results name the affected fab and equipment, show request priority, and omit the other fab by default.
4. **Lot flow & cycle time** provides cycle-time decomposition, priority comparisons, cycle-time distributions, WIP aging and a downloadable lot explorer.

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

## 2027 demand outlook

Demand is visible across **2027Q1–Q4**, with Combined, Fab A and Fab B aggregation. Sidebar family/program filters apply. The selected review quarter highlights the ramp and supplies the cards, while charts retain the full next-year horizon.

- Breakdown: program family, named program or route technology (Includes EUV / DUV only).
- Chart basis: requested plan or expected starts.
- Comparison: both fabs, a fab/program quarterly heatmap, and requested plan versus expected starts.
- Export: quarter × fab × program records with the realization factor and both weekly rates.

**Requested plan** is the supplied business input. **Expected starts** = requested plan × that program's historical realization factor, rounded per fab/program/quarter. A least-squares model fits actual starts = factor × requested starts, pooled across fabs for each program. The 2027 factors use history through 2026Q3. This estimates realization conditional on requests, not independent market demand, bookings or new customer orders. Capacity commitments remain requested-plan based.

The synthetic plan totals are 2,790 / 2,900 / 3,010 / 3,040 WSPW; expected starts are 2,678 / 2,763 / 2,850 / 2,867 WSPW. Each point is a weekly rate during its quarter, not the quarter's wafer total. The average-request card is the mean of four quarterly rates.

Forecast validation fits only 2024Q1–2025Q4 and evaluates 2026Q1–Q3. The full selection has 24 held-out fab/program/quarter records, raw-request MAPE 10.8% and adjusted MAPE 3.4%. The held-out chart aggregates records, while MAPE is computed across individual records. Backtest coefficients use only training data; final 2027 coefficients use all available history. All inputs are synthetic, with limited history; no future accuracy or confidence interval is asserted.

Route technology comes from configured EUV passes. It is not a physical wafer specification. Diameter, substrate, grade and product/SKU fields are absent and are not invented.

## Scenario timing, fab scope and priority

Choose **Quarter range** or **Week range**. Weekly options cover all 365 days of 2027 in 53 Monday-based calendar buckets; opening and closing weeks are clipped to the year. Required hours and available hours use actual day overlap with each quarterly baseline, including a resource release at a quarter boundary. A short request changes only its selected weekly window; outside weeks and the other fab retain baseline values. A flat target WSPW applies to the selected program during every affected segment.

Results default to the **fab receiving the change**, even when the workspace scope is Combined. Enable "Include the other fab in results" to inspect both. A request heading identifies fab, program, priority and timing; the at-risk list names each fab/equipment group and baseline → scenario loading. Exports follow that result scope and carry request/window/priority metadata.

**HOT/NORMAL is a declared request attribute.** The existing future plan has no priority split. Setting HOT does not alter equipment hours, reserve capacity, change simulated lots or prove an expediting benefit. A HOT request needs a stated experiment milestone/need-by context and a decision about displacement of normal work. Queue, sequence and delivery effects require a scheduling or simulation analysis. The optional milestone field documents context; it is not an approval workflow.

Weekly calculations are a steady-rate capacity screen derived from the quarterly model, not a rerun of the discrete-event simulation. Resource releases are assumed available from their configured quarter's first day. Weekly equipment loading is calculated using the window's hours, with weekly-equivalent starts shown explicitly for partial weeks. Capital costs and ROI are still outside scope.

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

Download **td-fab-planner-v5.zip** to Downloads. In Cursor's terminal:

```bash
unzip -q -o "$HOME/Downloads/td-fab-planner-v5.zip" -d "$HOME/Downloads/td-planner-v5"
ditto "$HOME/Downloads/td-planner-v5/td-fab-planner-update" "$HOME/Desktop/Operational-Intelligence-Dashboard"
cd "$HOME/Desktop/Operational-Intelligence-Dashboard"
git status
git add -A
git commit -m "Add 2027 demand outlook and weekly fab scenario comparisons"
git push origin main
```

After the push, use **Manage app → ⋮ → Reboot app** in Streamlit so imported chart modules reload together. Keep the entry point at `code/apprag.py`. A plain browser refresh can leave an older imported module in memory.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See `VALIDATION.md` for results and `DEMO_GUIDE.md` for the interview walkthrough.
