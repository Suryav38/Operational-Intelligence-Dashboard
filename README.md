# TD Fab — Performance, Resilience & Planning (v7)

Independent interview prototype for a fictional two-fab Technology Development network. **All operating data, routes, equipment counts, targets and notes are synthetic.** No Micron MES data or internal screen is reproduced.

The decision story is: **Can we accommodate unexpected DRAM growth or loss of equipment, and can qualified work move between fabs without creating another constraint?** One explicit pilot introduction adds a controlled WIP/completion study.

## What changed

- **Demand comparison:** 2026 vs 2027 by matching quarter, plus an eight-quarter history/outlook graph. The Q1 2027 change from Q4 2026 is separate from the same-quarter year-over-year change.
- **Q4 planning reference:** the user-supplied 2,950 WSPW total is separate from observed synthetic history. Its fab/program split is assumed from baseline 2027Q1 shares; no Q4 actuals are invented and it does not enter model training. Q1 2027 remains 2,800, a 150 WSPW / 5.1% decrease.
- **Filters within both comparison charts:** dropdowns for program/type, plus clickable legends. These affect their own graph; sidebar fab/program filters apply to all page totals and views. Export includes periods and source/allocation basis.
- Machine OEE removed. Equipment Health displays quantities by fab, actual simulated time states, weekly queues and furnace batch fill.
- **52 reporting weeks:** 29 September 2025–27 September 2026, with lot/program/priority detail, weekly wafer balances and dated issue records.
- **DRAM-heavy baseline:** DRAM technology, including HBM test vehicles, targets 75% of wafer starts; NAND 15%; experiments 10%. This is a user-selected synthetic start-mix assumption, not inferred from Micron revenue share. Weekly actual starts are approximately 75% after lot rounding.
- **2027 plan:** 2,800 / 3,000 / 3,300 / 3,600 WSPW. Fab A DRAM Node B grows 750 → 1,125 WSPW. Other types are reduced relative to the previous prototype.
- Six explicit ordered route definitions: five established programs and **DRAM_P / DRAM EUV pilot**, qualified only at Fab A.
- Pilot starts **17 May 2027**, week 7 of a 16-week study. Baseline, introduction and slower-ramp cases use the same starting WIP and matched existing-lot inputs across five seeds.
- Live scenarios cover **demand growth, temporary tool unavailability and qualified load transfers**, by quarter or start/end weeks, including the selected pilot loading.
- Embedded AI panels in Fab Performance and Capacity & Scenarios read current values, query deployed data, retrieve dated notes and call capacity models. Evidence preview works without an API key.

## Four views

1. **Fab performance** opens first: weekly flow, WIP, cycle time, program starts, equipment inventory/health, historical request attainment and route inspection.
2. **Demand outlook:** 2026 comparison, Q4-to-Q1 transition and full 2027 ramp by technology family, named program, route technology and fab. Requested plan, historical actuals and model expectations are labelled separately. A pilot has no history, so no fitted forecast is invented for it.
3. **Capacity & scenarios:** capacity heatmap, workload drivers, resource gaps, three disruption/response types, and a **Program introduction** tab with precomputed DES comparisons.
4. **Lot flow & cycle time:** observed cycle-time components, HOT/NORMAL cohorts, distributions, WIP age and downloadable lot detail.

Keep **Pilot case = Pre-pilot baseline** for the baseline view. Switch to **Pilot introduction** or **Slower pilot ramp** to include its requested demand in planning screens. Historical performance remains the historical simulation. Program-introduction flow results always identify their future study period and Fab A scope.

## Run

Python 3.11 or 3.12:

```bash
python -m pip install -r requirements.txt
python code/self_check.py
python -m streamlit run code/apprag.py
```

Streamlit Cloud entry point: **code/apprag.py**. All data is already supplied; startup does not regenerate the simulations.

## AI setup and data path

In Streamlit Cloud, use **Manage app → menu → Settings → Secrets**:

```toml
OPENAI_API_KEY = "your-own-key"
OPENAI_MODEL = "gpt-4o-mini"
```

An existing key in app Secrets is reused. No key is included in this project. For local use, set the environment variables. Charts, models, exports and keyword evidence previews work without a key; the app never substitutes a fake AI answer.

The data path is:

**GitHub files → deployed Streamlit tables/notes → selected filters and model values → AI evidence and model tools.**

- The assistant receives the current view's underlying values, scope, period and basis. It does not read screenshot pixels.
- It can query the bundled SQLite tables read-only, retrieve dated text notes, read pilot-study results and calculate isolated capacity requests.
- It does not fetch live GitHub content per question. GitHub is the versioned source deployed with the app.
- Optional embeddings and keyword matches retrieve unstructured notes. A labelled keyword fallback is used if semantic retrieval fails.
- Answers cite source IDs. Evidence includes source passages, table results and model operations. Unknown citations and unsupported numerical quantities are rejected; these checks do not prove every interpretation or causal statement.
- Old v1 prototype notes are marked **ARCHIVED**. Current v6 tables and dated v6 records take precedence. Notes do not silently overwrite route assumptions.
- Live AI calls require the user's configured key. Automated checks use mocked API responses plus actual SQL and note retrieval; no live generation was verified in the build environment.

## Three understandable models

| Model | Purpose | When it runs |
|---|---|---|
| Analytical capacity | Convert starts and route visits into hours; compare against qualified resources and availability | Live when a capacity view or custom request is calculated |
| Request-realization regression | Estimate actual starts conditional on supplied requests, with training/holdout separation | Demand view calculation |
| Re-entrant discrete-event simulation | Follow lots, shared queues, batching and holds; estimate WIP and matched-lot completion shifts | Reproducible offline generation; bundled case results are read by the UI |

See **MODEL_GUIDE.md** for the equations, inputs, limits and a concise interview explanation.

## Pilot study

- Fab A, 16 reporting weeks beginning **5 April 2027**, after 16 warm-up weeks.
- Release in week 7: **17 May 2027**.
- Introduction: 100 WSPW for two weeks, then 200 WSPW.
- Slower ramp: 50 WSPW for four weeks, then 75 WSPW.
- Five seeds: 7, 17, 27, 37, 47. Every case starts with identical pre-release WIP for its seed. Existing lots use their own deterministic random streams.
- The paired cohort comprises **1,613 existing-lot instances across five runs**, all completed in this study window. The UI still exposes matched and unfinished counts rather than assuming every future study will complete.
- Introduction: mean post-release WIP **8,780.5 → 10,240.5** input-wafer equivalents; mean existing-lot completion shift **0.59 days**.
- Slower ramp: mean WIP **9,334.5**; mean shift **0.24 days**. Reduced impact is not a guarantee that every equipment group meets the planning limit.

These are model outputs under synthetic assumptions, not real-fab predictions. Additional completion time is relative to baseline; targets are illustrative, not customer promises. HOT is non-preemptive priority and cannot create equipment capacity.

## Data and scope

- `data/fab/tables/`: structured CSVs, including ordered routes, program catalog, weekly program flow, operating events, future pilot ramps, initial WIP, paired completion results and simulation summaries.
- `data/fab/documents/v6_records/`: dated issue records, flow reviews, pilot change and model/mix policy. Existing notes are retained as clearly marked archives.
- `data/fab/td_fab.db`: same structured tables for read-only AI queries.
- `data/fab/specification.json`: declared provenance, periods, assumptions and study cases.

Historical quarter attainment for 2025Q4–2026Q3 reconciles to matching blocks of 13 reporting weeks. These are week-aligned reporting quarters; the future capacity model uses calendar-quarter day overlap. The source has 52 weeks of simulated history, not measured Micron operations.

The live capacity model accounts for full-route average hours; weekly windows prorate demand, resource releases and pilot rates by actual days. It does not time-phase existing WIP through individual operations. The separate DES supplies those flow effects for the three declared pilot cases. A typed arbitrary shock does not automatically rerun the DES or produce a new queue/delay prediction.

Tool groups pool identical qualified tools. Furnace batches require matching recipes, maximum three lots, a two-hour formation wait and one six-hour machine cycle. Metrology is sampled. Capacity holds are non-preemptive: they start after in-flight work. Operators, chambers, rework networks and electrical die yield are outside scope. There are no capital prices or ROI estimates.

## Reproduce synthetic data

```bash
python code/generate_fab_demo.py
```

This deliberately overwrites the generated tables, notes and database. A full run takes several minutes. Do not run it merely to open the dashboard. `--seeds 5` is the supplied study; fewer seeds are available for development. Legacy `data_generation.py` delegates to this generator.

## Update your Mac/GitHub project

Download **td-fab-planner-v7.zip** into Downloads, then run in Cursor:

```bash
unzip -q -o "$HOME/Downloads/td-fab-planner-v7.zip" -d "$HOME/Downloads/td-planner-v7"
ditto "$HOME/Downloads/td-planner-v7/td-fab-planner-update" "$HOME/Desktop/Operational-Intelligence-Dashboard"
cd "$HOME/Desktop/Operational-Intelligence-Dashboard"
git rm --ignore-unmatch code/oee_model.py templates/machine_oee_template.csv
git add -A
git commit -m "Add prior-year demand comparison and chart filters"
git push origin main
```

After Streamlit rebuilds, use **Manage app → menu → Reboot app** so all imported modules and data reload together. A browser refresh alone can retain an older imported module. Chart/data caches also use file fingerprints.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See **VALIDATION.md** for completed checks and practical limits; **DEMO_GUIDE.md** for a compact walkthrough; **AI_AND_DATA_GUIDE.md** for retrieval and model-call details.
