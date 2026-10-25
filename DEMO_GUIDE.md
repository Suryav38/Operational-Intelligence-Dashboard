# A focused stakeholder demonstration — v7

Use a six-to-eight-minute main story. All inputs and outputs are synthetic; this is a prototype of a planning workflow, not Micron internal data.

## 1. Establish the operating picture

Open **Fab Performance**. Show the 52-week history and narrow the week slider to a period of interest. Explain starts, completions, average WIP and completed-lot cycle time. The weekly program chart shows the DRAM-heavy mix. HBM belongs to the DRAM technology family; 75% starts is a demo assumption, not a translation of company revenue.

In **Equipment Health**, show quantities by fab, shared-tool time allocation and weekly queue changes. Inspect the dated scanner recovery note through the AI evidence preview. Engineering time is required development work, not automatically avoidable waste. Machine OEE has been removed.

If asked about routes, open **Program Routes**. Show a program's qualified fab, equipment visit counts and ordered repeated visits. The pilot has 44 mask layers and 402 illustrative operations, including eight EUV visits. These synthetic layer modules are not a validated device traveler.

## 2. Show demand and resource exposure

Open **Demand Outlook**, baseline. The requested network rate grows 2,800 → 3,600 WSPW over 2027. Fab A DRAM B rises 750 → 1,125 WSPW. Requested demand, fitted realization and a new program's unestimated ramp are labelled separately.

Select Q1. Explain the two comparisons: 2,950 → 2,800 is a **5.1% decrease from the prior-quarter planning reference**, while 2,727 → 2,800 is a **2.7% increase over the matching prior-year quarter**. The Q4 reference total was user-supplied; its program allocation is assumed and it has no actual starts. Do not describe the whole transition as uninterrupted growth.

Use a graph's dropdown to inspect DRAM including HBM, NAND, HBM alone or a named program. Use its legend to hide/show series. These controls affect that plot; sidebar filters change all demand plots and summary cards. The historical actual line ends at Q3 2026; the future expected line is a model estimate.

In **Capacity & Scenarios → Capacity Outlook**, Q2 Fab A EUV is at 99.2%. Explain required hours divided by available hours. Follow the horizon to the resource exposure in Q3 and the planned second EUV scanner in Q4. Resource gaps at the planning limit already present in the baseline must not all be blamed on a new program.

## 3. Test unexpected growth, unavailability or consolidation

In **Scenario Comparison**:

- **Demand growth:** increase Fab A DRAM B by 10% for Q2. Its requested starts rise by 83/week; EUV loading rises from 99.2% to 107.4%.
- **Tool unavailability:** remove one Fab A EUV tool for selected weeks. With only one released scanner, the model reports no available EUV capacity.
- **Qualified load transfer:** transfer part of DRAM A from Fab A to Fab B. Network starts remain unchanged; evaluate the receiving fab's furnace, deposition and other constraints. DRAM A has no EUV visits, so moving it does not solve an EUV scanner shortage. EUV-required routes cannot be transferred to unqualified Fab B.

Show either affected-fab-only or both-fab results as appropriate. Named at-risk equipment and request priority are explicit. This is a live capacity screen; it does not fabricate new WIP/delivery forecasts.

## 4. Introduce one program in the middle

Select **Pilot case = Pilot introduction**, then **Program Introduction**. Release is 17 May 2027, week 7 of a 16-week study. The program ramps 100 → 200 WSPW; a slower response ramps 50 → 75.

Show the three-case WIP graph and paired existing-lot completion shifts. Across five seeds, the introduction adds about 1,460 input-wafer equivalents of post-release average WIP and 0.59 days of mean completion time. Slower ramp adds about 554 wafers and 0.24 days. These are simulated consequences, not guarantees.

The same 1,613 lot instances across runs are paired from the initial WIP; all finish in the supplied horizon. Show the starting state and source comparison if asked. Processing/move/hold inputs remain matched; the completion shift is explained by changed queues.

In **Demand & Capacity**, show both quarter-average loading and the weekly forward-loading chart. A mid-quarter launch can have a stronger weekly peak than the quarter average. The slower response reduces impact but may leave constraints. Equipment counts, qualification, ramp timing and operating recovery remain planning decisions.

## Optional: AI

Ask a question in the relevant view, such as which equipment constrains a DRAM increase or which operating record supports investigating a queue spike. Explain that the assistant receives displayed values, queries the deployed tables, retrieves dated notes and calls numerical models. Open sources and model operations. No API key is needed for evidence preview; live answers need the configured key.

## Be ready to explain limits

History and planning are separate. Capacity calculates average hours; the DES tracks lots and queues. Pilot results are precomputed reproducible cases. Actual fab calibration, chamber/operator detail, rework, financial ROI and customer due-date promises are outside scope. See MODEL_GUIDE.md for the concise explanation and assumptions.
