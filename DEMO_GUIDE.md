# Stakeholder walkthrough — v5

All data is synthetic. Describe this as an independent planning prototype.

## 1. Execution first

Open **Fab performance → Production overview** with Combined and all programs. Explain completions/week, completed-lot cycle time, WIP and same-cohort wafer yield. Compare the two fabs' WIP and cycle-time trends. Filter **DRAM → DRAM Node B (EUV)**, then **HOT**. Combined cycle time is weighted by actual completed lots. Reset filters before the next comparison.

In **Equipment health**, compare productive, standby, engineering and downtime allocation. These are full-quarter equipment-group measurements. In **Historical attainment**, compare historical requested/actual starts; this follows program/fab filters, not execution-week filters.

## 2. Show the next-year demand ramp

Open **Demand outlook**, Combined and all programs. The 2027 requested plan rises from **2,790 to 3,040 WSPW**, about **9.0%** Q4 versus Q1. Show the family mix, switch to named programs, and compare Fab A with Fab B. The heatmap drills into fab × program × quarter.

Switch the chart basis to **Expected starts**: the model estimates historical start realization conditional on the requested plan. The expected totals are **2,678 / 2,763 / 2,850 / 2,867 WSPW**. Explain that this is not an independent market-demand forecast or a reason to discard commitments. The capacity baseline remains requested-plan based.

In **Forecast validation**, show the separate training/holdout periods and errors. The full synthetic holdout has 24 records; raw-request MAPE is about 10.8%, adjusted MAPE about 3.4%. These results do not guarantee next-year accuracy. Route technology means configured EUV use; physical wafer diameter/substrate data is absent.

## 3. Review capacity and a demand change

Open **Capacity & scenarios → Capacity outlook**, Combined, 2027Q2. The requested plan is **2,900 WSPW**. Fab A EUV is at **98.2%** loading. Follow the horizon to Q3 overcapacity and Q4 relief from a second scanner.

In **Scenario comparison**, choose **Fab A / DRAM Node B (EUV), +10%, Q2 only**. Results default to Fab A: **1,920 → 1,998 WSPW**, EUV **98.2% → 106.0%**. The named at-risk list also identifies Deposition. The baseline-versus-scenario chart omits unchanged Fab B. Use **Include the other fab in results** only when useful.

Switch to **Week range** and test **05–11 Apr through 12–18 Apr 2027**. Only those weeks change. Inspect either week and the full-year weekly risk trend. Equipment releases and quarter-boundary demand use actual calendar-day overlap; this is a weekly capacity screen, not a dispatch simulation.

Set **Request priority = HOT** and enter a real milestone only if one exists. The result heading/export records the declared priority; equipment hours remain the same. Discuss whether the need-by date justifies moving normal work, and identify the constrained equipment. Do not claim that a HOT label fixes overload or predicts delivery benefit.

## 4. Explain lot-flow drivers

Open **Lot flow & cycle time**. Compare processing, queues, transport and holds by program, HOT/NORMAL cohorts, cycle-time spread and WIP age. Queue + hold is about **40%** for the default full-flow population. Engineering holds can be required development work.

Open **Lot explorer** to inspect a specific completed lot or WIP cohort. Intermediate route progress is unavailable because only the final progress snapshot exists. Historical production does not change when a future scenario is evaluated.

## Optional: machine OEE

Open **Fab performance → Machine OEE**. EXAMPLE-01 is separate from the fab simulation. Show the factors and 61.25% OEE, edit inputs, or upload machine-period records from the template. Do not describe the example as measured Fab A/B OEE.

AI remains paused. Program/fab/lot detail is available; product SKU, individual-machine event history and physical wafer specifications are not present. Capacity screening does not predict new queue time, due dates, investment costs or ROI.
