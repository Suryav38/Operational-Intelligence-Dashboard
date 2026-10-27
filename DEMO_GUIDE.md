# Five-minute walkthrough

1. **Demand outlook.**
   - The trend chart: requested and actual starts run into the Q4 forecast, then continue through the 2027 plan.
   - The tiles: the 2026 average is 2,870 requested starts per week and the 2027 average is 3,811 (+33%). HBM grows the most (+82%), and NAND is the only program that declines (−6%).
   - The program × quarter table shows each cell's change against 2026Q4 (blue = up, red = down).
   - The realization factor explains the gap between requested and realization-adjusted starts (before capacity limits).
2. **Fab performance, Q4 to date.** Four of 13 weeks are done, and starts are running at 94% of the requested weekly rate. Open the fab assistant and ask: "What problems did the deposition tools at Fab A have in Q4 so far?" It answers with tool IDs, causes and hours, each cited.
3. **Capacity planning → Current fleet.** Show where tool time went, which groups are above the limit, and Fab A EUV: about 89% loaded, 15 h of queue per visit and 13% unscheduled downtime. The model check shows the capacity formula agrees with the lot simulation.
4. **2027 baseline.** Each quarter shows tools to add and the load before → after. Show the furnace lots-per-run sensitivity and the buy at measured availability.
5. **Response plan.** Growth only → reallocate the capacity freed by NAND and short loops → cost-optimal rebalancing → buy list with pull-ins.
6. **Scenarios.** Run REQ-2027-014 (Node B +100 at Fab A, Q1) on the current fleet: declined, because EUV is full, and the groups already overloaded are listed separately. Switch to *With planned 2027 purchases* to see it accepted with an exception.

Interview line: *"The AI is used where engineers lose time: searching tool logs and downtime events. It retrieves and cites them, and it can query the event table. The capacity model and the optimizer do every planning calculation, so every decision can be reproduced."*
