# TD Planning — Stakeholder Review (v3)

An independent interview prototype for a fictional two-fab Technology Development network. **All data is synthetic.** The interface is designed for a stakeholder review; it does not reproduce a Micron internal screen.

## What is in this version

- **Fab A, Fab B and Combined** views. Combined is the default.
- **Capacity plan:** comparative loading heatmap, equipment benchmark, wafer-start mix and workload-driver charts. Detailed tables and definitions are expandable.
- **Scenario review:** optional AI note reading, a guided loading change, baseline-versus-scenario charts, horizon-wide risk comparison and an AI explanation.
- **Fab performance:** wafer flow, WIP by fab, cycle time by program/fab and cycle-time composition. The reporting window is historical simulation data, not live production.
- **AI copilot:** reads the dashboard's selected site, quarter, limit and baseline/scenario; queries structured tables; retrieves planning-document passages; returns cited insights and an evidence trace.

## Run

Python 3.11 or 3.12 is supported; validation used Python 3.12. From the extracted project root:

```bash
python -m pip install -r requirements.txt
python code/self_check.py
python -m streamlit run code/apprag.py
```

For Streamlit Community Cloud, keep the entry point at **code/apprag.py**. The supplied data is ready to use; do not regenerate it merely to start the app.

## Enable live AI

In Streamlit Community Cloud, open your app's **Manage app → menu → Settings → Secrets**, or open app settings from your workspace. Add:

```toml
OPENAI_API_KEY = "your-own-api-key"
OPENAI_MODEL = "gpt-4o-mini"
```

Save the settings. For local use, copy `.env.example` to `.env` and supply the values there. Never commit your actual key. The sidebar says whether a key is configured; that is not a claim that a live connection was verified.

Without a key, the charts, manual scenarios, source-document preview and exports work. Live AI buttons are disabled and no fake answer is substituted. See `AI_AND_DATA_GUIDE.md` for the AI workflow.

## How the AI reads the dashboard and documents

1. The application supplies the current dashboard calculations and filter context as source **S1**. It reads the underlying data, not screenshot pixels.
2. SQL tools query the **11 structured CSV/SQLite tables** for historical, lot-level and planning details. Capacity questions use the deterministic model. Read-only queries have table restrictions, operation limits and row caps.
3. The **15 planning documents** are chunked with filename/position metadata. OpenAI `text-embedding-3-small` embeddings and keyword scores are combined for retrieval. This small corpus uses an in-memory vector index; a separate Chroma service is unnecessary. If semantic retrieval is unavailable, keyword fallback is identified.
4. The language model combines the evidence and cites **[S1]**, **[S2]**, **[D1]**, etc. Users can inspect source values, SQL, document passages and tool steps. Unknown citations and quantities absent from cited sources are rejected. These checks do not prove that every interpretation or causal statement is correct; human review remains necessary.
5. Hypothetical calculations use a separate copy of the plan. They do not edit files, approve requests or replace the scenario displayed elsewhere.

Notes supply context, not automatic route changes. A process-change notice must be reconciled with the configured baseline before applying it again. Historical notes and current planning results retain their own periods.

The compact scenario explanation has an additional check: its numerical quantities are substituted from known model-fact references. Unrecognized references and literal generated digits are rejected.

## Combined metric definitions

- **Wafer starts, completions and WIP:** sum the two sites' aligned weekly quantities.
- **Cycle time:** mean across the actual completed full-flow lots from both sites. This is lot-weighted; it is not an unweighted mean of site averages.
- **Line yield:** total surviving output wafers divided by input wafers for those same completed cohorts. This measures wafer survival, not electrical die yield.
- **Capacity risk:** count and compare each **fab/equipment** combination separately. Spare hours in Fab A do not erase a Fab B constraint. Equipment qualifications still apply.
- **WIP:** input-wafer equivalents from weekly snapshots; each active lot contributes its original 25 wafers. Interim losses are not removed from that snapshot quantity.

## Model scope

Required hours/week = Σ starts/week × passes/wafer × minutes/pass × sampling fraction ÷ 60. Available hours/week = installed resources × 168 × production availability. Loading is required divided by available hours.

The 85% planning limit is a chosen demo policy, not an asserted Micron standard. Equipment released in a quarter contributes a full quarter of capacity. Groups are pooled within each site. This aggregate model does not predict exact lot due dates, scenario queue times, costs, savings or ROI.

Operating performance covers **29 June–27 September 2026**, after a simulation warm-up. Some lots began before this window and completed inside it. The historical simulation is not rerun when a future request changes.

The original request-realization regression is retained as a supporting analytics tool. Lower historical realization does not guarantee spare capacity against committed requests.

## Update your existing Mac/GitHub project

Save the new download as **td-fab-planner-v3.zip** in Downloads. In Cursor's terminal:

```bash
unzip -q -o "$HOME/Downloads/td-fab-planner-v3.zip" -d "$HOME/Downloads/td-planner-v3"
ditto "$HOME/Downloads/td-planner-v3/td-fab-planner-update" "$HOME/Desktop/Operational-Intelligence-Dashboard"
cd "$HOME/Desktop/Operational-Intelligence-Dashboard"
git status
```

Check that `code/planning_copilot.py` and `code/executive_charts.py` appear among the new files. Then:

```bash
git add -A
git commit -m "Add stakeholder comparisons and grounded AI copilot"
git push origin main
```

Your Streamlit deployment must then rebuild from that commit. Refresh after the deployment completes.

## Tests and files

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

| File | Purpose |
|---|---|
| `code/apprag.py` | Four stakeholder-facing views |
| `code/executive_charts.py` | Comparative charts |
| `code/review_model.py` | Scope-safe metrics and Combined aggregation |
| `code/planning_copilot.py` | Hybrid retrieval, read-only SQL, model tools and citation checks |
| `code/assistant.py` | AI request extraction and grounded scenario explanation |
| `code/fab_capacity.py` | Capacity/scenario engine |
| `code/fab_config.py` | Assumptions, routes, equipment and plans |
| `DEMO_GUIDE.md` | Five-minute interview walkthrough |
| `AI_AND_DATA_GUIDE.md` | AI methodology and questions to demonstrate |
| `VALIDATION.md` | Checks completed and remaining limits |

The original 11 CSV files, 15 text documents and SQLite database are preserved. `fab_config.py` remains the source for many model assumptions; CSV equipment/route files are generated snapshots. `plan_2027.csv` supplies demand. Optional `python code/data_generation.py` regenerates and overwrites the synthetic data folder.

Public technical references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings), [Streamlit settings and secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings).
