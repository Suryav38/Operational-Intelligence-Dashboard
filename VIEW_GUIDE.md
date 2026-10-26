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

### Views, in order: 2027 capacity · Response plan · Scenarios

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

### Scenarios (with the AI assistant)

One tab for every "what if". The **assistant** sits on top: describe the scenario in plain English (or pick an example) and press *Fill inputs with AI*. The LLM must return a **strict JSON schema** (scenario type, fab, program, quarter, the new total or the change, target program, wafers shifted, tool group, tools down, transfer %, protected programs, constraints). Python then validates it:

- unknown values are dropped and qualification is checked;
- a "+50" becomes a new total on the 2027 requested plan;
- constraints must cite `REQUEST` or a retrieved note ID, and citations to notes that weren't retrieved are removed;
- anything the text didn't state is listed as missing.

The AI only fills the form. Every input stays editable, and **Python does every calculation**. Without an API key, the examples fill the form.

**Scenario types**

| Type | Inputs | What is calculated |
|---|---|---|
| Demand change | program, change per week, acceptance ceiling, protected programs | Weekly model through 2027. Responses tested: accept in full, accept part and defer/decline the rest, cut a lower-priority unprotected program on the bottleneck, run the excess on qualified tools at the other fab. These are ranked (below). |
| Product mix shift | from program, to program, wafers/week | Starts move between programs at the same fab, so the total is unchanged. Routes differ, so tool loading changes. |
| Tool unavailability | tool group, tools down | Fewer tools released in the quarter |
| Qualified load transfer | dual-qualified program, % moved | The share moves to the other fab. Results show both fabs. |

**Tools assumed:** *Planned fleet* (default) = installed + on order + the 2027 buys, each added in the quarter it is needed. Under this fleet the 2027 plan sits at or below the planning limit (max 84.9%), so a scenario shows its true marginal effect. *Current fleet* = installed + on order only ("what if we don't buy?").

**Demand-change ranking rules**, applied in order:
1. Feasible under the ceiling.
2. Protected programs are never touched.
3. The most wafers started now.
4. The fewest wafers taken from other programs.
5. The fewest tool groups pushed over the limit.

The decision band is *Within limit* (≤ planning limit), *Needs exception* (between the limit and the ceiling, default 95%), or *Not accepted* (above the ceiling).

**Results:**
- *Demand change:* the decision, the wafers that fit under the ceiling and under the limit, the binding tool group with its load before → after, and the ranked responses.
- *Other types:* starts, groups at or above the limit, peak load, groups over 100%, and one chart of the tool groups whose load changes (baseline vs. scenario).

An **AI explanation** follows each result. It gets the computed result as source [S1], plus retrieved notes. Every paragraph must cite a source, and every number must appear in a cited source. A failed draft gets one repair turn.

**Examples (planned fleet):**
- **REQ-2027-014: Node B +100 at Fab A, Q1** (970 → 1,070 requested). Accepted with an exception. Wet clean at Fab A binds at 85% → 89%. On the *current* fleet it is declined, because EUV is full and Fab B has no EUV.
- **REQ-2027-021: Node A +50 at Fab B, Q1.** Accepted with an exception; etch binds at 84% → 88%. On the current fleet only +20 fits.
- **Shift 50 WSPW NAND → HBM at Fab A, Q2.** EUV 78.9% → 81.3% and deposition 83.7% → 82.7%. Total starts are unchanged.
- **One EUV scanner down at Fab A, Q3.** EUV 61.7% → 92.5%. It stays under 100%, but it is above the limit.
- **20% of Node A from Fab A to Fab B, Q2.** Fab B wet clean 84.6% → 96.3%. The constraint moves to Fab B.
