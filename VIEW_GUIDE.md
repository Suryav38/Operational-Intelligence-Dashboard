# View guide: what each page answers and how every number is calculated

The dashboard answers one question: **what tools does the 2027 plan need, by when, and what could we do instead of buying them?** Each page is one step of that argument.

| Page | Question it answers |
|---|---|
| Capital plan | How many tools, of which kind, in which fab, by which quarter? |
| Demand | Is the plan we are sizing to believable? |
| Levers | What could cut the buy, and what does each option cost us? |
| Scenarios | What breaks if demand jumps, a tool goes down, work moves, or the EUV pilot launches? |
| 2026 baseline | Can we trust the model that produced all of the above? |

All data is synthetic: two development fabs, Fab A (with EUV) and Fab B (DUV only), and six programs.

---

## The three assumptions in the grey bar

These apply to every page. When a page doesn't use one, it's greyed out rather than hidden.

- **Fab**: Both fabs, Fab A or Fab B. This filters what you see. It never removes workload from the calculation, because tools are shared by every program running in that fab.
- **Plan case**: the current plan, the plan plus the DRAM EUV pilot (100 wafers/week from 17 May 2027, rising to 200), or the plan plus a slower pilot ramp (50, then 75).
- **Planning limit**: the highest loading you are willing to plan a tool group to run at. The default is 85%. This is a policy choice, and the Levers page shows what it costs to change it.

---

## 1. Capital plan

**Headline.** The headline is a sentence built from the numbers in the table below it. It is not AI-generated text.

**Size the plan to.** *Requested starts* is what the program teams asked for. *Expected starts* is each request multiplied by the share of its requests that the program has historically run (see Demand).

**Stat strip**

| Number | Calculation |
|---|---|
| Net-new tools to buy | The sum of **Buy** over all tool groups |
| On-order tools to pull in | The sum of **Pull in** over all tool groups |
| Tool groups short | The number of groups with Buy > 0 or Pull in > 0, out of all groups |
| First quarter short | The earliest quarter in which any group needs more tools than are released that quarter |
| Peak load | The highest loading of any group in any quarter, using the tools actually released in that quarter |

**Tool requirement table.** This is the core of the dashboard. For each fab, tool group and quarter:

```
Required hours/week  = Σ programs ( starts/week × visits per wafer × minutes per visit × sampled fraction ) ÷ 60
Hours per tool/week  = 168 × production availability        (availability = 1 − scheduled down − unscheduled down − engineering time)
Load (Q1–Q4 cells)   = Required ÷ (tools released that quarter × hours per tool)
Tools needed         = ⌈ Required ÷ (hours per tool × planning limit) ⌉
Peak need            = largest Tools needed across 2027
Buy                  = max(0, Peak need − (installed + all tools on order))
Pull in              = on-order tools needed before their scheduled arrival quarter
Need by              = first quarter where Tools needed > tools released that quarter
```

Cell colours: no fill means the group is below the limit, amber means between the limit and 100%, and red means over 100% (the plan physically cannot run).

Example: Fab A EUV has 1 scanner installed and 1 arriving in 2027Q4. The requested plan needs 2 from 2027Q1. Buy is **0** and Pull in is **1**. The correct action is to ask for the scanner already on order to arrive three quarters early, not to buy a third.

**What drives the requirement.** Pick a tool group to see a stacked bar for each quarter showing the hours each program consumes. The black tick marks the hours the fleet supplies; it steps up when tools arrive. The dashed amber tick marks the planning limit. The box beside the chart works through the calculation with the real numbers for that group's worst quarter. The table lists each program's starts, visits, minutes per visit and share of the hours. This is the route information that matters for capital, which is why the old route-inspection page was dropped.

---

## 2. Demand

| Element | Calculation |
|---|---|
| History and plan | Actual starts (black) and requested starts (dotted) from 2024Q1 to 2026Q3, then 2027 requested (solid blue) and expected (dashed). All values are weekly averages for the quarter, not quarterly totals. |
| 2027 mix | Stacked starts by program or by fab, as requested or as expected |
| Realization factor (bar chart) | For each program, the least-squares slope of actual starts on requested starts, through the origin: `b = Σ(req × act) ÷ Σ(req²)`. Fitted on all history and pooled across fabs. |
| Expected starts | Requested × b. The pilot has no history, so it stays at its request. |
| Backtest error | Fit b on 2024Q1–2025Q4 only, predict 2026Q1–Q3, then compute MAPE (mean of \|predicted − actual\| ÷ actual) across the 24 fab-program-quarters. The raw request misses by about 10%; request × b misses by about 2%. |
| DRAM share | DRAM + HBM requested starts ÷ all 2027 requested starts (75% by design) |

This is request realization, not a market forecast. It answers "how much of what program teams ask for actually gets started?"

---

## 3. Levers

The waterfall applies the levers one after another. Each step keeps the steps before it.

1. **Plan to expected starts.** Repeat the Capital plan calculation using requested × b. The risk line comes from a test: buy only the tools the expected plan needs, then run the full request. The line counts how many groups end up above the limit and above 100%.
2. **Raise the planning limit** by 5 or 10 points. Repeat the calculation at the higher limit. The cost comes from queueing theory: queue time grows with `u ÷ (1 − u)`. Moving from 85% to 90% multiplies queue time by `(0.90/0.10) ÷ (0.85/0.15) ≈ 1.6×`. That extra waiting becomes cycle time.
3. **Rebalance qualified work.** A grid search over the three programs qualified at both fabs (DRAM Node A, 3D NAND, short loops). It tries every combination of moving 0 to the maximum share (in steps of 5%) in either direction: 9,261 combinations at a 50% cap. It keeps the combination with the fewest net-new tools, and on a tie, the one that moves the fewest wafers. Network starts are conserved. EUV programs can't move because Fab B has no EUV.

**Why the limit is not 100%.** This chart is the queue-time curve for one tool group, `queue = k × u/(1−u) × process time`. The constant k is calibrated so the curve passes through that group's simulated 2026 operating point (the black diamond). The two markers show queue hours at the current and raised limits.

**Where the savings land.** Net-new tools by group after each step.

Limitation: a tool count is not a cost. An EUV scanner and a wet bench both count as one. The project deliberately contains no tool prices.

---

## 4. Scenarios

**Demand, tool or transfer.** This tab runs the same capacity calculation with one change: a program's starts up or down (as a percentage or a target), N tools of a group unavailable for a window, or a share of a program moved between fabs. Changes can apply to whole quarters or exact weeks. In weekly mode, hours and tool releases are prorated by the number of days that overlap each week. The results show starts, groups at or above the limit, the peak load and groups over 100%, before and after. HOT priority changes dispatch order, not tool hours, so it doesn't change capacity.

**EUV pilot study.** This tab shows precomputed discrete-event simulation (DES) results for Fab A. The study covers 16 weeks from 5 Apr 2027, with the pilot released in week 7. Each of 5 random seeds runs three cases from identical starting WIP, and each existing lot keeps its own random stream. Because of that, **delay = the same lot's finish time with the pilot − its finish time without**. The page shows:

- average WIP after release, baseline vs. case (wafers)
- mean and 90th-percentile delay to existing lots
- pilot lots completed
- the WIP curve (mean of the runs, with a min–max band)
- delay by program and queue by tool group
- quarter and weekly loading with the pilot added

This is the only view that measures WIP and delay. The capacity screen measures hours.

---

## 5. 2026 baseline

| Element | Calculation |
|---|---|
| Capacity model vs. lot simulation | Take the actual starts from the last 13 simulated weeks and run them through the capacity formula (blue dot). Compare with the production loading the lot simulation produced (diamond), which is productive hours ÷ (calendar − downtime − engineering). 18 of 19 groups agree within ±3 pts. The exception is the Fab B furnace (+11 pts): batches rarely fill, and a part-full batch still takes the full 6-hour run. The capacity formula assumes 3-lot batches. |
| Starts run vs. requested | Σ actual ÷ Σ requested over the last four quarters. This is where the realization factor comes from. |
| Cycle time | The weekly average cycle time of completed lots, by fab. The planning limit exists to protect this. |
| Availability table | Assumed downtime + engineering share vs. what the simulation observed, for each group |

---

## What was removed, and why

| Removed | Reason |
|---|---|
| Equipment health page (time states, queue trends, batch fill) | This is daily operations monitoring, not capital planning. The one part that matters for capital, whether assumed availability holds, is now the baseline availability table and model check. |
| Lot explorer, WIP aging, HOT vs NORMAL, cycle-time components | Lot-level execution detail with no effect on how many tools to buy |
| Line yield, completions KPI | Not an input to the capacity model |
| Program routes page | Replaced by the visits-per-wafer table in the Capital plan drill-down, which is the route information the calculation actually uses |
| Program/family multiselect, "More filters" popover, filter chips | The program filter changed displayed totals but not tool loading, which was confusing. Global filters are now three assumptions, and local controls sit next to the chart they change. |
| AI panel on every page | One panel on Capital plan and one on Scenarios, where a question leads to a decision |
