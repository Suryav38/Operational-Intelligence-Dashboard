# Stakeholder walkthrough — v4

All data is synthetic. Describe this as an independent planning prototype.

## 1. Execution first (90 seconds)

Open **Fab performance → Production overview** with Combined and all programs. Explain four metrics: completions/week, completed-lot cycle time, WIP and same-cohort wafer yield. Compare the two fabs' WIP and cycle-time trends. Filter **DRAM → DRAM Node B (EUV)**, then **HOT**. Explain that combined cycle time is weighted by actual completed lots. Reset the program, priority and week filters before the next comparison.

## 2. Equipment health and plan attainment (60 seconds)

In **Equipment health**, compare productive, standby, engineering, scheduled down and unscheduled down. These are full-quarter group time allocations, not individual-machine OEE. Show the equipment queue trend. In **Historical attainment**, compare requested and actual start rates across quarters. This history follows the selected programs, not the execution-week slider.

## 3. Find the bottleneck (90 seconds)

Open **Capacity & scenarios → Capacity outlook**, Combined, 2027Q2. The requested plan is 2,900 WSPW. Fab A EUV loading is 98.2%; Fab B Etch and Deposition are also above the demo 85% limit. Follow the horizon heatmap to Q3 EUV >100% and its Q4 relief from a second scanner. The equipment-gap table identifies tools required at the limit without inventing spending or ROI.

In **Scenario comparison**, select Fab A / DRAM Node B (EUV), +10%, Q2 only. Compare: EUV rises to 106.0%. Show baseline/scenario loads and risk across quarters. Alternatively enter a target WSPW. Confirm that every other program stays in the calculation even when the display is focused on one program.

## 4. Explain flow losses with evidence (60 seconds)

Open **Lot flow & cycle time**. Compare processing, queues, transport and engineering holds by program, HOT/NORMAL cohorts, distributions and WIP ages. Queue + hold is about 40% in the default full-flow data. Avoid claiming all holds are removable waste. Open **Lot explorer** and inspect a specific lot or WIP cohort. Intermediate route progress is intentionally unavailable because the data has only a final progress snapshot.

## Optional: machine OEE (45 seconds)

Open **Fab performance → Machine OEE**. State explicitly that EXAMPLE-01 is a worked calculation separate from the fab simulation. Show A/P/Q and 61.25% OEE, edit an input, or upload actual machine-period records using the template. Do not describe this example as measured Fab A/B OEE.

## Scope to explain if asked

- Five programs, two fabs, lot/priority detail; no product SKU, machine event or current process-step history.
- Operating detail: 13 simulated weeks in Q3 2026. Requested/actual starts: 2024Q1–2026Q3. Capacity plan: 2027Q1–Q4.
- Capacity predicts aggregate hours, equipment gaps and constraints. New queue/cycle-time predictions would require a validated simulation scenario.
- AI is paused in this version; calculation, filtering and export operate without an API key. Legacy retrieval/model-call modules remain available for a later embedded question interface.
