# View guide: what each page shows and how the numbers are calculated

There are three pages, and they form one planning workflow:

| Page | Question |
|---|---|
| **Demand outlook** (opens first) | What is requested for 2027, how did it change between the 2026Q3 and 2027 plans, and how much of it will actually run? |
| **Fab performance** | What did the line deliver in 2026Q4 to date: starts, completions, WIP and cycle time? It includes an AI assistant over the tool logbook. |
| **Capacity planning** | How is the current fleet doing, what does the 2027 demand profile need, what is the cheapest response, and what if something changes? |

All data is synthetic. The model is a two-fab development network: Fab A has EUV, Fab B is DUV only. It runs five established programs.

The grey bar at the top holds three settings: **Fab** (Both, Fab A or Fab B), **Planning limit** (default 85% loading) and **Tool availability**. Tool availability is either the *planning assumption* (1 − PM − unscheduled down − engineering) or the value *measured 2026Q4 to date*. It drives the 2027 baseline, the response plan, the planned purchases and every scenario, including the AI's own calculations. The pages have no headline sentences. Each chart heading states what is being compared and over what period.

---

## 1. Demand outlook

The 2027 plan is compared against **today**: the 2026Q4 rate (the quarter in progress).

| Element | How it is calculated |
|---|---|
| KPI tiles | 2026 vs. 2027 **averages** of requested starts per week. The 2026 average uses the 2026Q1–Q3 requests plus the 2026Q4 request; the 2027 average uses the four 2027 plan quarters. The tiles show both averages, the growth between them, and the programs with the largest increase and the largest drop (or the smallest increase, if no program declines). |
| Requested vs. actual wafer starts | Quarterly *average weekly* starts. **Requested** (grey) runs through 2026Q4, then continues as the 2027 plan request (blue). **Starts** (black) are actuals to 2026Q3, then the 2026Q4 forecast, then 2027 realization-adjusted starts (request × realization factor, hollow markers). These are *before capacity limits*: the demand the fleet must carry, not a check of whether it can. |
| 2026Q4 forecast | Q4 is in progress. For each fab and program, Q4 = the Q3 value × (1 + the average quarter-on-quarter change over the last four quarters), done separately for requests and for starts. It never enters the fit. |
| Wafer starts by program and quarter | Each cell shows the 2027 value and its change against 2026Q4. Shading: blue for increases and red for decreases, darker for bigger changes (±3%, 10% and 25% bands). |
| Requested vs. actual by program | One chart per program, covering 2024–2027. |
| 2027 mix | Stacked by program, family or fab. |
| Actual vs. requested (realization factor) | `b = Σ(requested × actual) ÷ Σ(requested²)` for each program, fitted on 2024Q1–2026Q3. **Realization-adjusted = requested × b** (before capacity limits). The backtest refits on data up to 2025Q4 and measures MAPE on 2026Q1–Q3: about 10% for the raw request against about 2% for request × b. |
| Priorities and milestones | Each program's priority, milestone, date and protected flag, with its 2026Q4 rate, 2027 average and change. |

The 2027 requests are stored on the same basis as historical requests (run rate ÷ b). That's why there is no artificial step at the year boundary: requests go 2,942 → 2,994 → 3,130 from 2026Q3 to 2027Q1.

## 2. Fab performance: 2026Q4 to date (28 Sep – 25 Oct 2026, simulated)

The four weeks after the stored history come from rerunning the same fab simulation (seed 42) for 56 weeks instead of 52. Lots, process times and tool states use per-lot and per-tool random streams, so the first 52 weeks reproduce the stored history exactly. The script checks this before writing anything. Only the new weeks are saved, to the `*_q4` tables (`code/simulate_q4_to_date.py`).

| Element | Calculation |
|---|---|
| Tiles (5) | Weeks completed out of 13; wafers started so far as a share of the quarter total (13 × the weekly request); average starts per week as a share of the requested weekly *rate*; WIP, with completions per week; cycle time, with its p90 and line yield. |
| Starts vs. completions vs. Q4 request | Weekly wafers against the 2026Q4 requested rate. |
| WIP and cycle time | WIP at each week end; the average cycle time of lots completed that week. |

Tool loading, downtime and availability moved to **Capacity planning → Current fleet**.

**Fab assistant: tool logbook and events** (collapsed at the bottom of the page). You ask in plain English, for example:
- "What problems did the deposition tools at Fab A have in Q4, and how many hours were lost?"
- "Which tools went down most often?"

The assistant:
1. retrieves the matching weekly equipment reviews, using keyword or keyword + embedding search;
2. can run read-only SQL on `tool_events_q4` (one row per unscheduled-down event: tool ID, start, hours, cause, action) and on the other tables;
3. answers with a citation on every paragraph. Numbers must come from a cited source, and a failed draft gets one repair turn.

`code/make_tool_logs.py` writes the reviews and the event table from the simulated tool states (seed 42). The event hours add up to the simulated unscheduled downtime for every fab, week and tool group. Like the rest of the data, it is synthetic.

## 3. Capacity planning

### Views, in order: Current fleet · 2027 baseline · Scenario analysis · Response plan

### Current fleet

The installed and on-order fleet, and how each tool group spent its time in 2026Q4 to date.

| Element | Calculation |
|---|---|
| Tiles (5) | Tools installed (and on order); utilization = productive ÷ calendar time, tool-weighted; unscheduled downtime hours and number of events; tool groups at or above the planning limit; the longest queue per visit. |
| Where tool time went | Calendar time split into productive, standby, engineering, scheduled down (PM) and unscheduled down, for each tool group. |
| Production loading vs. planning limit | Productive ÷ (calendar − PM − unscheduled down − engineering). The label shows queue hours per visit. |
| Availability: measured vs. planning assumption | Measured (productive + standby) ÷ calendar, against the planning value of 1 − PM − unscheduled down − engineering. |
| Fleet register | Installed, on order with arrival quarter, utilization, loading, availability against plan, unscheduled-down hours and events, the main downtime cause (from the tool logbook) and queue per visit. Downloadable. |

### 2027 baseline: what the demand profile needs from the fleet

```
required tool-h (quarter) = Σ starts × visits × min/visit × sampling ÷ 60
available tool-h per tool = 168 × availability (× batch efficiency for furnaces)
tools needed (quarter)    = ⌈required h ÷ (available h per tool × limit)⌉
fleet needed by quarter   = running maximum of tools needed (tools are never removed)
add (quarter)             = increase in fleet needed beyond installed + on order, quarter by quarter
buy · 2027                = sum of the quarterly adds = peak need − (installed + on order)
pull in                   = on-order tools needed before their scheduled arrival, with the quarter they're needed by
```

*Size the plan to* chooses requested starts or realization-adjusted starts (request × each program's historical share; before capacity limits).

Each quarter cell shows the tools to add, then **load before → after**:
- *before* = required hours ÷ hours from installed + on-order tools released by then;
- *after* = with the adds and pull-ins in place.

Shading follows the load after the response, so a group that the buy resolves no longer looks unresolved. A drill-down lists required vs. available hours by program, with the full arithmetic.

Furnaces use the lots per 6-hour run that the lot simulation achieved (Fab A 2.96, Fab B 2.61), not the 3-lot design value, because partial batches still take a full run.

### Response plan: close the 2027 gap, cheapest levers first

The plan starts from the 2027 baseline buy and applies each lever before buying:

| Step | What it does | Inputs |
|---|---|---|
| 2027 baseline buy | Net-new tools from the 2027 baseline tab | Size to: requested or realization-adjusted starts |
| Cut unscheduled downtime | Engineering recovers a share of each tool group's unscheduled downtime, and those hours become production time. The table shows each group's measured Q4 downtime, so you can see where the lever matters (Fab A EUV 13.4%, Fab B planarization 11%). | Downtime cut: 0, 10, 25 (default) or 50% |
| Rebalance fabs | A grid search over moving 0–max% of each dual-qualified program (DRAM Node A, NAND, short loops) between the fabs. It keeps the move only if the total cost index falls; a move can add a few tools at the receiving fab. | Most of a program that may move: 25, 50 or 100% |
| Buy the rest | The buy list by quarter, with load before → after and on-order pull-ins | |

Under the planning-assumption availability, with a 25% downtime cut and up to 50% movable: 125 → 118 after the downtime cut → 116 after rebalancing (cost index 288 → 264). The rebalancing moves 5% of DRAM Node A and 20% of NAND from Fab A to Fab B.

The tiles show the baseline buy, the tools saved by each lever, the tools to buy (with cost index) and the on-order tools to pull in. "Where each lever acts" breaks this down by fab and tool group. Cost index = Σ tools × relative cost (wet clean = 1, EUV = 18); it is illustrative and only ranks options.

### Scenario analysis

The manual scenario form covers four tests:

| Test | What changes |
|---|---|
| Demand growth | One program's starts, as a % or a target |
| Product mix shift | N wafers/week moved from one program to another at the same fab, with total starts unchanged |
| Tool unavailability | N tools out of service for a window |
| Qualified load transfer | A share of a dual-qualified program moved to the other fab |

Each test runs by quarter range or exact week range, with an optional availability recovery for demand growth.

**Tools assumed:**
- **Current fleet** (the default) = installed + on order.
- **With planned 2027 purchases** = the current fleet plus the buys from the 2027 baseline, each added in the quarter it's needed. Use it to see a scenario's marginal effect once the plan is funded.

Request priority (HOT) was removed: it changes dispatch order, not tool hours, so it never changed a capacity result.

**Results:** stats, baseline vs. scenario load by tool group, groups above the limit across 2027, the full table with a CSV download, and the planning assistant. The assistant's own calculations use the same fleet, availability, window and recovery as the screen.

**Responses tested (demand growth only).** These are calculated by the capacity model in Python (`wafer_analyst.py`); no AI is involved. Each response is re-run through the same weekly loading model. The responses are:
- accept in full;
- accept part and defer or decline the rest;
- cut a lower-priority, unprotected program on the bottleneck;
- run the excess on qualified tools at the other fab.

They are ranked by these rules, in order:
1. Feasible under the 95% ceiling.
2. Protected programs are never touched.
3. The most wafers started now.
4. The fewest wafers taken from other programs.
5. The fewest groups over the limit.

How the ranking works:
- **It covers exactly the scenario you ran:** every quarter in the window, the selected weeks, the availability setting and any recovery. A multi-quarter window shows what fits in each quarter, and the table shows the most constrained one.
- **Every tool group at both fabs is checked.** Groups already above the ceiling *before* the request are listed separately ("already above the ceiling, not caused by this request"). Any response that adds hours to one of them is *Not accepted*.
