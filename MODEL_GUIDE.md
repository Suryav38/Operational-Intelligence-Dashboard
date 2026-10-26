# How the fab models work

This is a synthetic decision-support prototype. The goal is to explain demand, resource constraints and the consequences of one new-program introduction. It is not a validated digital twin of Micron.

## 1. Capacity: a fast hours calculation

For each fab, equipment group and planning period:

**Required hours/week = Σ(program starts/week × visits/wafer × minutes/visit × sampling) ÷ 60.**

**Available hours/week = released tools × 168 × production availability.**

**Loading = required ÷ available.**

The route matters: 100 wafers requiring eight EUV visits are not equivalent to 100 wafers requiring no EUV. A furnace's analytical equivalent assumes a three-lot batch; actual partial-batch inefficiency is represented in the separate DES.

For the v6 pre-pilot Q2 plan, Fab A requests 830 DRAM B and 170 HBM test-vehicle starts per week. Each needs seven EUV visits at one minute/wafer/visit. Required EUV hours = (830 + 170) × 7 ÷ 60 = **116.67 hours/week**. One scanner at 70% availability supplies **117.6 hours/week**, so loading is **99.2%**. A 10% increase in DRAM B adds 83 WSPW and raises loading to **107.4%**.

Removing the only available scanner produces **no capacity**, not a finite loading percentage. Transferring a program is allowed only if its route is qualified at the receiving fab; network starts are conserved. Moving work can create a different receiving-fab bottleneck. The model does not approve a transfer or predict cross-site logistics lead time.

The 85% planning limit is a demo policy, not a Micron standard. Tools required at the limit = ceiling(required hours ÷ per-tool hours at that limit). Existing baseline gaps remain visible, so the entire gap is not falsely attributed to a new pilot.

Weekly requests cover Monday-based buckets clipped to calendar 2027. Hours and released resources are weighted by actual day overlap. Pilot rates use their actual release/ramp dates, rather than smearing a mid-quarter launch across every week.

Code: `fab_capacity.py`, `launch_model.py`, `weekly_scenarios.py`, `scenario_requests.py`.

## 2. Historical realization: a small regression

For each established program, fit:

**actual starts = realization factor × requested starts.**

The factor is least squares through the origin, pooled across the two fabs. Future expected starts are requested starts multiplied by that factor. This is conditional request realization, not a market-demand or customer-order forecast.

Backtesting fits through 2025Q4 and evaluates 2026Q1–Q3. Final 2027 factors use all available history. Site/program filters change the displayed aggregation, not the fitting population. The pilot has no history, so its fitted estimate remains unavailable; its ramp is an explicit planning assumption.

Code: `forecast_model.py`, `demand_outlook.py`.

## 3. Lot flow: a re-entrant discrete-event simulation

The simulation clock moves between events. Each lot:

1. Enters on a seeded release calendar at 25 wafers/lot.
2. Follows an explicit, ordered program route, repeatedly returning to shared equipment groups.
3. Waits for a qualified resource. HOT requests get non-preemptive queue priority; they do not interrupt an operation already running.
4. Processes, moves/dispatches, and sometimes enters an engineering hold. Metrology visits are sampled; wafer losses reduce surviving quantities.
5. Completes its route or remains WIP at the study end.

Single-tool groups pool identical resources. Furnaces collect only compatible recipe lots, maximum three, and wait at most two hours to form a batch. A batch occupies **one machine for six hours**, even when partly filled. It does not consume six independent machine-hours per lot in the same batch. Pilot anneal recipes are separate from established qualified recipes.

Background availability processes and explicit dated incidents occupy resource capacity. Holds are non-preemptive and can wait for current processing to finish. Actual hold starts may move with congestion. This simplification does not model a sudden chamber abort or wafer rework after failure.

The route generator uses declared layer modules and configured visit counts. These are explicit synthetic assumptions, not validated device recipes. The source is visible in `route_steps.csv` and the Program Routes tab.

Code: `fab_demo.py`, `fab_flow.py`, `generate_fab_demo.py`.

## 4. A controlled program-introduction experiment

The future study begins 5 April 2027, after a 16-week warm-up. The pilot is introduced on 17 May, week 7. Three cases are run for each of five seeds:

- No pilot.
- 100 pilot WSPW for two weeks, then 200.
- Slower response: 50 WSPW for four weeks, then 75.

For a given seed, all cases share the same pre-release WIP and existing release calendar. Random process, movement, hold and survival draws are keyed to the individual lot, so adding pilot lots does not arbitrarily rerandomize existing work. HOT and NORMAL use the same intrinsic distributions. Seeded availability processes are also retained; non-preemptive hold start times can shift when the resource is busy.

Initial WIP records capture lot identity, route version, current operation, queue/processing/hold state, remaining in-flight time and remaining route steps. The same existing lots are paired across cases:

**Additional completion time = scenario completion − baseline completion.**

The paired cohort is 1,613 lot instances across five seeds. All complete in the supplied study, but matched and unfinished counts remain explicit. Completion shifts are not customer lateness. Assumed targets are clearly labelled and are not actual order promises.

Post-release average WIP is the mean of weekly end snapshots after the introduction date. The pilot adds **1,460 input-wafer equivalents** in this example; the slower ramp adds **554**. Mean existing-lot completion shifts are approximately **0.59 days** and **0.24 days**. These are generated outputs, not prescribed values or guarantees.

The plotted ranges span the five runs' minimum and maximum WIP. They show simulation variability, not uncertainty bounds on a real fab prediction. Current capacity loading is a separate planning check and need not equal simulated productive loading: batch fill, yield, transient queues, sampling and realized holds differ from steady average assumptions.

The app reads bundled study outputs; it does not rerun the full DES on each click. Regeneration is reproducible with `python code/generate_fab_demo.py`. Custom demand/outage/transfer calculations remain live analytical screens. Arbitrary new queue or delivery predictions require a new DES run.

## 5. Why the history is believable as a demonstration—but still synthetic

The 52-week history is generated by the same fab-flow rules, with a trend, seeded release variation and dated capacity holds. Weekly surviving-wafer balances reconcile: opening WIP + starts − completions − scrap = closing WIP. Input-wafer-equivalent WIP is also retained for consistent lot-size reporting.

Reporting-quarter actual starts for the overlapping year come from the same 13-week blocks as the weekly program records. Requested starts and pre-2025Q4 history are synthetic planning inputs. Notes are generated from known events and resulting records; a note documents an intervention, not proof that every observed queue change had that cause.

Demonstration fleet quantities and the 75% DRAM technology mix are explicit assumptions. Revenue share is not wafer-start share. No observed Micron data has been used to establish predictive accuracy.

## A concise explanation to the hiring manager

The capacity screen converts program routes and starts into equipment hours. The flow study then follows lots through shared, re-entrant resources to show how one pilot changes WIP and existing-lot completion time. Baseline and response cases use matched starting conditions, and the dashboard exposes assumptions, sources and incomplete cohorts. AI helps retrieve and explain the evidence; the numerical models calculate it.


## Demand comparison and the year boundary

History ends at 2026Q3 (reporting week ending 27 Sep 2026). 2026Q4 is the quarter in progress, so it has no actuals. Each fab/program continues its own last-four-quarter trend, separately for requests and for starts. The result is stored as a labelled forecast in `demand_bridge_2026q4.csv` and never enters the realization fit or the backtest.

The 2027 requests (`fab_config.REQUESTED_WSPW_2027`) are on the same basis as the historical requests: planned run rate (`RUN_RATE_WSPW_2027`) ÷ the program's realization factor, rounded to 5. Earlier versions entered run rates labelled as requests, which made both the request and expected-start lines step down about 5% at the year boundary. The lot simulation still releases work at the run rates.

Charts compare weekly rates, not quarter totals. The history uses reporting-quarter blocks; the future plan uses calendar quarters.
