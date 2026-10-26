# Five-minute walkthrough

1. **Demand outlook.**
   - The trend chart: actual starts run into the Q4 forecast, then the July and October plans branch from it.
   - The bridge: Node B and HBM drive the increase, while NAND and short loops give some back.
   - The program × quarter table shows each cell's change against July.
   - The realization factor explains the gap between requested and expected starts.
2. **Fab performance, Q4 to date.** Starts are at 94% of the Q4 plan. EUV is 89% loaded with about 15 hours of queue per visit. Measured availability is compared with the planning assumption, and the model check shows the capacity formula is trustworthy.
3. **Capacity scenario planning → Wafer request analyst.**
   - REQ-2027-014 (Node B +100 in Q1) is declined: EUV is already full under the October plan, HBM is protected, and Fab B has no EUV.
   - REQ-2027-021 (Node A +50 at Fab B) is accepted by cutting short-loop experiments.
   - With an API key, show the AI extraction with its citations, then the cited explanation.
4. **Response plan.** Can the current fleet run the October plan? No.
   - The steps: increases only → reallocate the capacity freed by NAND and short loops → cost-optimal rebalancing → buy list with need-by quarters and the EUV pull-in.

Interview line: *"AI turns a written request into checked inputs and explains the result with citations. The capacity model and the optimizer do every calculation, so the decision can be reproduced."*
