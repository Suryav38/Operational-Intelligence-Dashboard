# AI and evidence — v8

## Where AI appears

Expandable assistants sit inside historical Production Overview and Equipment Health, the 2027 Capacity Outlook, live Scenario Comparison and Program Introduction. Each receives a separate scope-specific context; there is no detached generic chat page.

No API call runs merely from opening the dashboard or changing a filter. **Ask AI** triggers generation. **Preview evidence** is keyword retrieval plus the exact numerical context and works offline.

## Data path

1. GitHub stores versioned project code, CSV/SQLite tables and text notes.
2. Streamlit loads its deployed copy. File fingerprints invalidate data caches when files change.
3. The panel supplies the currently selected filters, reporting period, pilot case, planning limit and calculated values as **S1**. It reads underlying values, not chart pixels or client-side legend toggles.
4. Read-only SQL queries retrieve additional structured facts. Notes are searched independently with keyword retrieval and optional embeddings.
5. Explicit model tools calculate capacity scenarios or read the precomputed pilot study. The assistant explains the returned evidence and cites **S1/S2/D1** source IDs.

It does not continually fetch live GitHub files. Updates enter through the deployment. Changes to an on-screen filter are reflected in the next question's context. Equipment-state context follows fab/equipment/weeks, because shared tool states are not attributed to the selected program or priority.

## Structured data

The bundle contains 28 tables. They include:

| Group | Tables |
|---|---|
| Engineering assumptions | program_catalog, route_steps, routes, qualification, tool_groups, installed_tools, tool_arrivals |
| Historical planning/flow | wafer_start_history, sim_lots, sim_fab_weekly, weekly_program_flow, sim_tool_group_weekly, sim_tool_group_q3_2026, operating_events |
| Future requested demand | plan_2027, pilot_plan_quarterly, pilot_plan_weekly |
| Prior-year planning reference | demand_bridge_2026q4 (user-provided total; assumed program allocation; no actuals; excluded from model training) |
| Pilot experiment | launch_initial_wip, launch_summary, launch_weekly, launch_program_weekly, launch_equipment, launch_lot_comparison, launch_batches, three launch_lots case tables |

`data/fab/td_fab.db` mirrors the CSVs. Raw SQL answers identify the actual query and scope; numerical aggregation must use the intended dates, site, case and seed.

## Unstructured evidence

There are 36 text notes: 15 archived earlier-prototype records and 21 current records. Current records include five dated capacity-hold incidents, 13 calendar-month flow reviews (the 52-week span includes partial boundary months), one pilot change note, one model/mix policy and one demand-comparison reference policy. The Q4 reference policy distinguishes the user-provided 2,950 WSPW total from its assumed allocation and unavailable actuals.

The current notes identify the request and actual realized hold timestamps, affected equipment and duration, with references to source tables. A documented capacity hold does not establish that every observed WIP or queue change was caused by it.

Legacy notes begin **ARCHIVED PROTOTYPE NOTE**. They cannot override the v6 fleet, route or loading tables. Historical observations, future requests and simulated pilot results retain separate periods and provenance.

Text chunks retain source filenames and character positions. Optional semantic retrieval uses `text-embedding-3-small`, combined with keyword ranking. If embedding retrieval is unavailable, the fallback is labelled. Structured tables remain SQL/model inputs, not document embeddings.

## Model tools

- **Read capacity:** route-aware requested-plan loading, including the selected pilot case.
- **Calculate capacity request:** isolated demand changes, unavailable tools or qualified transfers, using explicit quarter/week bounds. It does not mutate source files or active UI controls.
- **Read pilot study:** baseline/introduction/slower results from matched SimPy runs. These are precomputed simulation outputs, not measured production or a new arbitrary run per question.
- **Query data / search notes:** inspect source facts and dated context.

Transfers enforce qualification and conserve total network starts. Losing the only EUV tool produces no EUV capacity. A HOT label does not increase available hours. Arbitrary pilot ramp, route, queue or due-date predictions outside the three supplied DES cases require a new simulation study.

## Checks and limits

Source IDs must exist. Numerical claims must occur in cited evidence; unknown quantities are rejected. SQL runs through a bounded, read-only connection with a table allowlist and operation limits. Text and table contents are evidence, not instructions.

These checks do not prove semantic association, causality or recommendation quality. The assistant should identify measured facts, documented explanations and hypotheses separately. It cannot approve loading, promise a due date, infer real-fab accuracy, or invent CAPEX/ROI.

No live API key was available during the build. Tests exercise actual data/retrieval and model calculations with mocked generation/embedding calls. Enable the user's own key in Streamlit Secrets; no fake AI response is supplied when disconnected.
