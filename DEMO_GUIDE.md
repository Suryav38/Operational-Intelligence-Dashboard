# Five-minute stakeholder walkthrough

Present this as a self-directed prototype using synthetic data.

## 1. Compare the network — about one minute

Start with Combined, 2027Q2 and the 85% policy. Show the four KPIs, the quarterly heatmap and the equipment comparison. Describe the constraint in each fab. Use wafer-start mix and workload contribution to explain the program drivers. The Combined view retains site-specific capacity constraints; it does not pool all equipment into one utilization figure.

## 2. Test one request — about two minutes

Open Scenario review. Read or enter a request for DRAM Node B, +10%, at Fab A, during 2027Q2 only. If live AI is enabled, use the visible AI request reader and review the extracted fields. Otherwise use the manual controls and state that live AI is disabled.

Select Compare scenario. Show baseline and scenario equipment loading, then the horizon-wide risk chart. Explain which constraints are new and which already existed. Optionally test recovered availability, treating it as an assumption requiring Operations confirmation.

Select Explain results with AI if configured. Its interpretation supplements the computed evidence; it is not an approval.

## 3. Connect an operating observation — about one minute

Open Fab performance. Keep Combined selected. Compare the two fabs' WIP and cycle time. Explain that overall cycle time is weighted by completed full-flow lots and line yield uses matching completed cohorts. Avoid treating this historical simulation as a forecast of the proposed plan.

## 4. Show one cited AI insight — about one minute

Open AI copilot and select Use latest reviewed scenario. Ask:

> Why is the most constrained equipment at risk, and what do the planning notes add?

Show the cited answer, open one source, and show what the AI read or calculated. Distinguish current model values from historical equipment notes. A document provides engineering context; it does not establish a causal explanation by itself.

If no API key is configured, show Preview evidence and explain the retrieval workflow. Do not present a mock response as a live AI result.

## Questions to be ready for

- Where did the data come from? Synthetic configuration, request history and a SimPy simulation.
- What does the AI read? The same selected model results as the dashboard, structured tables through read-only queries, and retrieved passages from the planning documents.
- Why 85%? A demo planning buffer; the correct policy depends on actual equipment behavior and service requirements.
- Why can one fab's spare capacity not cancel another's constraint? Work needs compatible, qualified equipment at a particular site.
- What can this model claim? Aggregate capacity risk and comparisons. Exact lot lateness, ROI and implemented savings require more data and validation.
- What does the AI add? Natural-language access, evidence retrieval, comparison and interpretation, with sources that a planner can inspect.
