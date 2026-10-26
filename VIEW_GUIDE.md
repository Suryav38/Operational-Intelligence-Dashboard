# View guide: what each page shows and how the numbers are calculated

There are three pages, and they form one planning workflow:

| Page | Question |
|---|---|
| **Demand outlook** (opens first) | What is requested for 2027, how did it change between the 2026Q3 and 2027 plans, and how much of it will actually run? |
| **Fab performance** | What did the line deliver in 2026Q4 to date? These are the operating conditions every scenario starts from. |
| **Capacity scenario planning** | Can we accept extra wafers, what happens to the programs sharing the tools, and what is the cheapest response to the 2027 plan? |

All data is synthetic. The model is a two-fab development network: Fab A has EUV, Fab B is DUV only. It runs five established programs.

The grey bar at the top holds two settings: **Fab** (Both, Fab A or Fab B) and **Planning limit** (default 85% loading). The pages have no headline sentences. Each chart heading states what is being compared and over what period.

---

## 1. Demand outlook

The 2027 plan is compared against **today**: the 2026Q4 rate (the quarter in progress).

| Element | How it is calculated |
|---|---|
| KPI tiles | 2026Q3 actual starts; the 2026Q4 forecast; 2027Q4 requested and expected starts; request growth from 2026Q4 to 2027Q4; and the backtest error for the raw request vs. the adjusted request. |
| Requested vs. actual wafer starts | Quarterly *average weekly* starts. **Requested** (grey) runs through 2026Q4, then continues as the 2027 plan request (blue). **Starts** (black) are actuals to 2026Q3, then the 2026Q4 forecast, then 2027 expected starts (request × realization factor, hollow markers). |
| 2026Q4 forecast | Q4 is in progress. For each fab and program, Q4 = the Q3 value × (1 + the average quarter-on-quarter change over the last four quarters), done separately for requests and for starts. It never enters the fit. |
| Demand bridge | The 2026Q4 rate, plus each program's change, gives the 2027 average. |
| Wafer starts by program and quarter | Each cell shows the 2027 value and its change against 2026Q4. Shading: blue for increases and red for decreases, darker for bigger changes (±3%, 10% and 25% bands). |
| Requested vs. actual by program | One chart per program, covering 2024–2027. |
| 2027 mix | Stacked by program, family or fab. |
| Actual vs. requested (realization factor) | `b = Σ(requested × actual) ÷ Σ(requested²)` for each program, fitted on 2024Q1–2026Q3. **Expected = requested × b.** The backtest refits on data up to 2025Q4 and measures MAPE on 2026Q1–Q3: about 10% for the raw request against about 2% for request × b. |
| Priorities and milestones | Each program's priority, milestone, date and protected flag, with its 2026Q4 rate, 2027 average and change. |

The 2027 requests are stored on the same basis as historical requests (run rate ÷ b). That's why there is no artificial step at the year boundary: requests go 2,942 → 2,994 → 3,130 from 2026Q3 to 2027Q1.

## 2. Fab performance: 2026Q4 to date (28 Sep – 25 Oct 2026, simulated)

The four weeks after the stored history come from rerunning the same fab simulation (seed 42) for 56 weeks instead of 52. Lots, process times and tool states use per-lot and per-tool random streams, so the first 52 weeks reproduce the stored history exactly. The script checks this before writing anything. Only the new weeks are saved, to the `*_q4` tables (`code/simulate_q4_to_date.py`).

| Element | Calculation |
|---|---|
| Progress tiles | Weeks completed out of 13; wafers started so far as a share of the quarter total (13 × the weekly request); average starts per week as a share of the requested weekly *rate*. |
| Starts vs. completions vs. Q4 request | Weekly wafers against the 2026Q4 requested rate |
| WIP and cycle time | WIP at each week end; the average cycle time of lots completed that week |
| Cycle time (p90), line yield | Full-flow lots completed in the window; yield = wafers out ÷ wafers in |
| Tool loading vs. planning limit | Productive hours ÷ (calendar − downtime − engineering). The label shows queue hours per visit. |
| Availability: measured vs. assumed | Measured (productive + standby) ÷ calendar, against the planning value of 1 − PM − unscheduled downtime − engineering |
| Starts by program vs. Q4 plan | Average weekly starts ÷ Q4 requested starts |
| Capacity model vs. lot simulation | 2026Q3 starts run through the capacity formula, compared with the simulated loading. 18 of 19 tool groups agree within ±3 points. Fab B furnace is the exception, because partially filled batches still take a full 6-hour run. |

## 3. Capacity scenario planning

### Views, in order: 2027 capacity · Response plan · Wafer request analyst · Other scenarios

### Wafer request analyst (the AI feature)

1. **Request.** You type or paste a request. *Extract inputs with AI* sends it to an LLM together with the top retrieved notes. The LLM must fill a **strict JSON schema**: fab, program, quarter, the new total or the change, weeks, protected programs, and constraints. Each constraint must cite `REQUEST` or a retrieved note ID. Python then validates the output:
   - unknown values are dropped;
   - qualification is checked;
   - a "+50" is converted to a new total;
   - dates are snapped to planning weeks;
   - citations to notes that were not retrieved are removed;
   - anything the request didn't state is listed as missing.

   Without an API key, choosing an example fills the inputs.
2. **Inputs.** Every input stays editable:
   - *Screen against*: planned run rate (default) or the requested plan.
   - *Acceptance ceiling*: default 95%.
   - *Tool availability*: the planning assumption, or the value measured in Q4 to date.
3. **Model.** Weekly through 2027, for both fabs:

   ```
   required tool-h = Σ starts × visits × min/visit × sampling ÷ 60   (prorated by days in the week)
   available tool-h = tools released that week × 168 × availability
   room = min over the program's tool groups and weeks of (ceiling × available − required) ÷ hours per wafer, rounded down to 5
   ```

   The analyst tests each response through the same weekly model:
   - accept the full request;
   - accept what fits and defer the rest (or decline it if no later quarter has room);
   - cut a lower-priority, unprotected program that shares the bottleneck;
   - use qualified tools at the other fab.

   **Ranking rules**, applied in order:
   1. Feasible under the ceiling.
   2. Protected programs are never touched.
   3. The most wafers started now.
   4. The fewest wafers taken from other programs.
   5. The fewest tool groups pushed over the limit.

   **Decision bands:**
   - Within limit: at or below the planning limit.
   - Needs exception: between the limit and the ceiling.
   - Not accepted: above the ceiling.
4. **AI explanation.** The LLM gets the computed result as source [S1], plus retrieved notes. It must cite a source for every paragraph, and every number it uses must appear in a cited source (rounding is allowed). If a draft fails, the model gets one repair turn and is told the exact reason.

**Examples:**
- **REQ-2027-014: DRAM Node B 825 → 925 at Fab A, Q1.** EUV supplies 117.6 h/week. The 2027 plan already uses (825 + 180) × 7 ÷ 60 = 117.25 h, which is 99.7%. There is no room, HBM is protected, and Fab B has no EUV, so the request is declined: capacity has to be added (see Response plan).
- **REQ-2027-021: DRAM Node A +50 at Fab B, Q1.** Wet clean binds. The recommended response cuts short-loop experiments (priority 3) to free the hours.
- **Short loops +100 at Fab A, Q1.** The excess runs on qualified tools at Fab B.

### 2027 capacity (first view)

```
tools needed (quarter) = ⌈required h ÷ (168 × availability × limit)⌉
fleet needed by quarter = running maximum of tools needed (tools are never removed)
add (quarter) = increase in fleet needed beyond installed + on order, quarter by quarter
buy · 2027 = sum of the quarterly adds = peak need − (installed + on order)
pull in = on-order tools needed before their scheduled arrival, with the quarter they're needed by
```

Each quarter cell shows the tools to add that quarter, with the load *before* adding them underneath (required hours ÷ hours from installed + on-order tools released by then). For example, Fab A Deposition needs 50 → 59 → 66 → 73 tools against 42 installed: **+8, +9, +7, +7** by quarter, 31 for the year. A drill-down lists required vs. available hours by program, with the full arithmetic.

### Response plan: 2026Q4 rate → 2027 plan → response

The view sizes the response step by step:

1. **2026Q4 rate**: net-new tools if today's rate were held flat through 2027.
2. **Growth only**: the 2027 plan for programs that grow, with every other program held at today's rate. Nothing gives way.
3. **Reallocate freed capacity**: the actual 2027 plan, in which NAND and short loops decline in some quarters and release shared etch, deposition and wet tools. EUV gets nothing back, because NAND doesn't use it. The chart shows hours added against hours freed for each tool group.
4. **Rebalance fabs**: a grid search over moving 0–50% of each dual-qualified program (DRAM Node A, NAND, short loops) between the fabs, in either direction. That is 9,261 combinations. The search minimises the illustrative cost index, Σ net-new tools × relative cost (wet clean = 1, EUV = 18).

The view ends with the buy list: tools, need-by quarter, pull-ins and cost index. You can download it as a CSV.

### Other scenarios

Each test runs on one of two fleets:

- **Planned fleet (default):** installed + on order + the 2027 buys from the *2027 capacity* view, with each tool added in the quarter it's needed. A pulled-in tool is not counted again when its original order arrives. The 2027 plan then sits at or below the planning limit everywhere (max 84.9%), so a scenario shows its true marginal effect. Without the buys, groups already at 105–150% make every scenario look the same.
- **Current fleet:** installed + on order only. This answers "what if we don't buy?"

The tests:

- **Demand growth:** change one program's starts, as a % or a target.
- **Product mix shift:** move N wafers/week from one program to another at the same fab, with total starts unchanged. Because routes differ, the same wafers load tools differently. Shifting NAND to HBM, for example, adds EUV and lithography load and frees deposition and etch.
- **Tool unavailability:** take N tools out of service for a window.
- **Qualified load transfer:** move a share of a dual-qualified program to the other fab.

Each test can run by quarter or by exact week. The **Wafer request analyst** has the same *Tools assumed* switch.
