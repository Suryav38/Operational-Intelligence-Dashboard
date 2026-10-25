# TD Capacity Plan 2027 (v9)

An independent interview prototype for a fictional two-fab Technology Development network. **All operating data, routes, tool counts, targets and notes are synthetic.** No company data and no internal screens are used.

The dashboard answers one question: **what tools does the 2027 plan need, by when, and what could we do instead of buying them?**

| Page | Question it answers |
|---|---|
| **Capital plan** | Net-new tools and pull-ins by fab, tool group and need-by quarter, with a drill-down into which programs drive each group |
| **Demand** | Requested vs. expected 2027 starts, and how much of a request historically gets run |
| **Levers** | Planning to expected starts, raising the loading limit, and an optimized rebalancing of qualified work, each shown with its risk |
| **Scenarios** | Live demand, tool-outage and transfer tests, plus the precomputed EUV-pilot lot simulation (WIP and delay) |
| **2026 baseline** | Checks the capacity model against the lot simulation, request attainment and cycle time |

Three global assumptions sit in the grey bar: **Fab**, **Plan case** and **Planning limit**. Every other control sits next to the chart it changes. **VIEW_GUIDE.md** explains every view and calculation.

### What changed in v9
- The equipment-health, lot-explorer and route pages were removed. They covered daily operations, not capital decisions.
- New `capital_model.py` provides the buy list (net-new vs. pull-in), lever steps, a transfer optimizer, and the model-vs-simulation check.
- New `plan_charts.py` and `dashboard.css` give one restrained visual system: IBM Plex, a single accent colour, and colour used only for limit and over-capacity status.
- The program/family filters and the "More filters" popover were removed. The sidebar was replaced by a top tab rail.
- The AI assistant now appears only on Capital plan and Scenarios.

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

Download **td-fab-planner-v9.zip** into Downloads, then run in Cursor:

```bash
unzip -q -o "$HOME/Downloads/td-fab-planner-v9.zip" -d "$HOME/Downloads/td-planner-v9"
ditto "$HOME/Downloads/td-planner-v9/td-fab-planner-update" "$HOME/Desktop/Operational-Intelligence-Dashboard"
cd "$HOME/Desktop/Operational-Intelligence-Dashboard"
git rm --ignore-unmatch tests/test_dashboard.py tests/test_dashboard_v8.py
git add -A
git commit -m "Refocus dashboard on 2027 capital planning"
git push origin main
```

After Streamlit rebuilds, use **Manage app → menu → Reboot app** so all imported modules and data reload together. A browser refresh alone can retain an older imported module. Chart/data caches also use file fingerprints.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See **VALIDATION.md** for completed checks and practical limits; **DEMO_GUIDE.md** for a compact walkthrough; **AI_AND_DATA_GUIDE.md** for retrieval and model-call details.
