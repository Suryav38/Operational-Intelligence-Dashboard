# Validation — v8

V8 verification: the full 92-test suite and offline self-check passed on Python 3.12. Desktop browser checks exercised all four workspaces, Demand detail/validation, scenario calculation, nested pilot capacity and lot-ID search. No JavaScript page errors were observed. New browser layout checks used 1512×1100 and 390×844 viewports. Earlier model and download checks below describe the retained v7 baseline; the in-chart dropdown interaction was replaced by shared on-screen filtering in v8.

## Completed checks

- **92 automated tests passed** under Python 3.12. Coverage includes capacity arithmetic, demand estimates, prior-year comparisons and missing values, weekly calendar overlap, route qualifications, synthetic flow, paired pilot comparisons, retrieval and dashboard interactions. AI response generation is mocked; SQL, note retrieval and mathematical model tools use the actual project data.
- `python code/self_check.py` passes. Python sources compile successfully.
- The baseline 2027 requested rates are **2,800 / 3,000 / 3,300 / 3,600 WSPW**. DRAM technology, including HBM test vehicles, represents 75% in every quarter. This is an explicit synthetic wafer-start assumption, not a conversion from Micron revenue.
- The data contains 52 reporting weeks for both fabs. Weekly surviving-wafer balances reconcile: opening WIP + starts − completions − scrap = closing WIP. Program and priority partitions reconcile to total flow.
- Historical actual-start rates for 2025Q4–2026Q3 reconcile to their matching blocks of 13 reporting weeks. Requested starts and older planning history remain separately generated synthetic inputs.
- Six program routes match their configured equipment visit totals. The pilot has 402 ordered operations, 44 mask layers and eight EUV visits. Fab B has no EUV tools; unqualified DRAM B and pilot transfers are rejected.
- Equipment inventory contains all ten tool groups at each fab, including Fab B's zero EUV quantity: 139 tools at Fab A and 61 at Fab B before future releases. Equipment time allocation reconciles to calendar hours. Queue averages are weighted by visits and furnace fill by actual batches.
- Furnace tests verify compatible recipes, maximum batch size and a single six-hour resource cycle shared by three batched lots. HOT dispatch is non-preemptive; processing, transport and hold distributions do not improve merely because priority changes.
- Pilot cases use identical pre-launch states for each seed. Keyed random streams preserve each existing lot's intrinsic processing, transport, hold and survival draws across cases. Completion differences therefore reflect changed waits in this model. Capacity holds remain non-preemptive; their realized timing can depend on congestion.
- All 1,613 starting existing-lot instances across five paired runs finish within the study horizon. Mean introduction-case additional WIP is approximately **1,460 wafers**, and mean existing-lot completion shift is **0.59 days**. The slower ramp gives approximately **554 wafers** and **0.24 days**. These are comparisons within this synthetic study, not customer lateness forecasts.
- The pre-pilot Fab A Q2 EUV calculation is independently checked: 116.67 required tool-hours / 117.6 available tool-hours = 99.2% loading. A 10% DRAM B increase raises it to 107.4%.
- Growth, tool unavailability and qualified load transfer retain the baseline and untouched periods. Outages cannot remove more released tools than exist. Transfers conserve network starts and change workloads at both fabs.
- The 2027 calendar contains exactly 365 days across 53 clipped weekly buckets. Boundary weeks prorate quarterly demand and tool releases by exact day overlap. Pilot weekly demand starts on 17 May; quarterly averages reconcile to the ramp.
- The new pilot has no historical realization model. Its expected starts remain unestimated; the dashboard does not present an unsupported all-program expected total when the pilot is active.
- Held-out realization estimates preserve training separation. This regression estimates requested-to-actual start conversion; it does not forecast external market demand.
- AI evidence binds to current filters, period, displayed values and active pilot case. Its SQL tool is read-only, notes carry dates and provenance, archived notes cannot override current assumptions, and model tools calculate scenario values. Citation and numeric checks are covered by tests.
- Machine OEE is removed from the application and its calculator/template are excluded from this package.
- All 28 CSV tables match the SQLite query copies (within export precision). Realized DRAM share across the stochastic weekly history is 75.157%, consistent with the 75% release target after random variation and lot rounding.
- Browser checks cover performance, equipment, routes, AI evidence preview, baseline/pilot demand, pilot WIP/capacity, outages and qualified transfers. No JavaScript page errors were observed. Desktop (1512×1100) and mobile (390×844) have no document-level horizontal overflow; mobile tabs scroll and cards stack.
- Actual browser downloads were checked: 402 route operations, 36 pilot-inclusive demand rows, 1,613 paired lot instances and 76 transfer equipment-quarter rows across both fabs. Transfer workload is conserved across the network; positive pilot demand retains missing expected starts rather than inventing an estimate.
- v10: the 2026Q4 trend forecast totals 2,994 requested / 2,789 starts with no actuals. 2027Q1 requests 2,990 (expected about 2,800), so neither line steps down at the year boundary (tests/test_demand.py). The Q4 forecast does not alter the historical model fit.
- v10: the wafer loading analyst reproduces the hand calculation for REQ-2027-014: EUV 117.6 h/wk available, 105 h baseline, 6.72 h of room at 95% gives 55 extra Node B wafers (tests/test_wafer_analyst.py). LLM answers are tested with mocked responses (citation check and repair turn).
- v10: the Q4 tool logbook agrees with the simulation: for every fab, week and tool group the event hours in tool_events_q4 sum to the simulated unscheduled-down hours (within 0.05 h); no event exceeds 60 h and no tool exceeds 168 h in a week (tests/test_tool_logs.py).
- V8 shared-filter regressions cover on-page controls, inactive tabs remaining unbuilt, selection/window persistence, reset behavior, demand chart reconciliation, limited chart periods, quarter-on-quarter arithmetic, capacity retaining other-product workload, equipment filter empty states, and lot search/age limits. Missing historical/future actuals remain unknown.

## Model limits

All routes, fleets, lots, demand and operating notes are synthetic and independent of Micron. More detailed inputs make this demonstration internally consistent; they do not establish real fab predictive accuracy. Validation against MES, actual qualification matrices, downtime events and completion outcomes would be necessary for operational use.

The capacity model charges average full-route work per wafer start. It identifies equipment-hour gaps; it does not predict new WIP or completion times. Custom growth, outage and transfer requests run this capacity model. They do not rerun the discrete-event simulation.

The precomputed pilot study follows remaining routes, shared equipment, compatible furnace batches, sampled metrology, queues and holds. It represents one selected Fab A launch with three cases and five seeds. It omits chamber-level matching, operators, detailed recipe setup, rework networks, transport networks and electrical yield. Engineering development time is not automatically avoidable loss.

Availability holds are non-preemptive and do not interrupt an in-process lot. Installed tools are pooled as identical units within each group. Future tools become usable at their declared release boundary; installation and qualification lead times are not independently predicted.

Illustrative completion targets are not promised customer due dates. Additional completion delay is measured against the paired baseline. The five-seed ranges describe model randomness, not confidence bounds on real fab accuracy. Finite-window WIP and completed-cohort cycle time are not asserted to satisfy an exact steady-state Little's Law balance.

Historical execution ends on 27 September 2026. The 2027 plan and pilot launch are future scenarios. Reporting-quarter history uses week-aligned blocks, while future capacity uses calendar boundaries. Partial calendar months are labelled in the operating reviews.

Live AI generation needs `OPENAI_API_KEY` in deployment secrets. No key was available during these checks, so actual provider responses, credentials, latency and billing were not exercised. Evidence preview, retrieval, SQL and model tools were exercised locally.

Internal browser screenshots and downloaded QA samples are excluded from the deliverable.
