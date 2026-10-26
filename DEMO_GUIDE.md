# Five-minute walkthrough

1. **Demand outlook.**
   - The trend chart: requested and actual starts run into the Q4 forecast, then continue through the 2027 plan.
   - The bridge: from today's 2026Q4 rate to the 2027 average. Node B and HBM drive the growth; NAND gives a little back.
   - The program × quarter table shows each cell's change against 2026Q4 (blue = up, red = down).
   - The realization factor explains the gap between requested and expected starts.
2. **Fab performance, Q4 to date.** Four of 13 weeks are done. Starts are running at 94% of the requested weekly rate. EUV is 89% loaded with about 15 hours of queue per visit. Measured availability is compared with the planning assumption, and the model check shows the capacity formula is trustworthy.
3. **Capacity scenario planning → Scenarios.**
   - Type "Add 100 wafers a week of Node B at Fab A in Q1" and press *Fill inputs with AI*. The assistant fills the form, with citations for each constraint.
   - Run it on the planned fleet: accepted with an exception, and wet clean binds. Switch *Tools assumed* to the current fleet: declined, because EUV is full.
   - Then pick "One EUV scanner down" and "Move 20% of Node A to Fab B" to show the other scenario types. Each one ends with a cited AI explanation.
4. **Response plan.** Can the current fleet run the 2027 plan? No.
   - The steps: growth only → reallocate the capacity freed by NAND and short loops → cost-optimal rebalancing → buy list with need-by quarters and the EUV pull-in.

Interview line: *"AI turns a written request into checked inputs and explains the result with citations. The capacity model and the optimizer do every calculation, so the decision can be reproduced."*
