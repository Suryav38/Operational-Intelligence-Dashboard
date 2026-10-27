# Five-minute walkthrough

1. **Demand outlook.**
   - The trend chart: requested and actual starts run into the Q4 forecast, then continue through the 2027 plan.
   - The tiles: the 2026 average is 2,870 requested starts per week and the 2027 average is 3,811 (+33%). HBM grows the most (+82%), and NAND is the only program that declines (−6%).
   - The program × quarter table shows each cell's change against 2026Q4 (blue = up, red = down).
   - The realization factor explains the gap between requested and realization-adjusted starts (before capacity limits).
2. **Fab performance, Q4 to date.** Four of 13 weeks are done. Starts are running at 94% of the requested weekly rate. EUV is 89% loaded with about 15 hours of queue per visit. Measured availability is compared with the planning assumption, and the model check shows the capacity formula is trustworthy.
   - Ask the fab assistant: "What problems did the deposition tools at Fab A have in Q4 so far?" It pulls the weekly equipment reviews and the event table, and answers with tool IDs, causes and hours, each cited.
3. **Capacity scenario planning → Scenarios.**
   - Run a manual demand-growth test. Show where the gaps are (the tool groups above the limit), then the ranked responses.
   - Switch *Tools assumed* to the current fleet: the request is declined because EUV is full. Then try a tool outage and a Node A transfer.
4. **Response plan.** Can the current fleet run the 2027 plan? No.
   - The steps: growth only → reallocate the capacity freed by NAND and short loops → cost-optimal rebalancing → buy list with need-by quarters and the EUV pull-in.

Interview line: *"The AI is used where engineers lose time: searching tool logs and downtime events. It retrieves and cites them, and it can query the event table. The capacity model and the optimizer do every planning calculation, so every decision can be reproduced."*
