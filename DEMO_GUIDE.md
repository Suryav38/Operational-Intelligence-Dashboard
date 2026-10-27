# Five-minute walkthrough

1. **Demand outlook.**
   - The trend chart: requested and actual starts run into the Q4 forecast, then continue through the 2027 plan.
   - The tiles: the 2026 average is 2,870 requested starts per week and the 2027 average is 3,811 (+33%). HBM grows the most (+82%), and NAND is the only program that declines (−6%).
   - The program × quarter table shows each cell's change against 2026Q4 (blue = up, red = down).
   - The realization factor explains the gap between requested and realization-adjusted starts (before capacity limits).
2. **Fab performance, Q4 to date.** Four of 13 weeks are done, and starts are running at 94% of the requested weekly rate. Open the fab assistant and ask: "What problems did the deposition tools at Fab A have in Q4 so far?" It answers with tool IDs, causes and hours, each cited.
3. **Capacity planning → Current fleet.** Show where tool time went, which groups are above the limit, and Fab A EUV: about 89% loaded, 15 h of queue per visit and 13% unscheduled downtime.
4. **2027 baseline.** The demand profile needs 125 net-new tools. Each quarter shows the tools to add and the load before → after.
5. **Scenario analysis.** Run REQ-2027-014 (Node B +100 at Fab A, Q1) on the current fleet: it's declined because EUV is full, and the groups that were already overloaded are listed separately. Switch to *With planned 2027 purchases* to see it accepted with an exception.
6. **Response plan.** 125 baseline → 118 after recovering 25% of unscheduled downtime → 116 after rebalancing NAND and Node A to Fab B. Then show the buy list with pull-ins.

Interview line: *"The AI is used where engineers lose time: searching tool logs and downtime events. It retrieves and cites them, and it can query the event table. The capacity model and the optimizer do every planning calculation, so every decision can be reproduced."*
