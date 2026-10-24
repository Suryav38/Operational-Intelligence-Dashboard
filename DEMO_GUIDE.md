# Five-minute interview walkthrough

Describe this as a **self-directed prototype using synthetic data**. Its value is the decision workflow and explainable calculations.

## 0:00–1:30 — Capacity plan

Start with Fab A, 2027Q2 and the 85% planning limit. Explain the selected quarter and requested-start basis. Point to the number of equipment exceptions, then the three focus groups. The headline count checks all equipment, including groups hidden from the focus chart.

Select EUV in the workload explanation. Trace starts/week into visits and processing hours, then compare required hours with available hours and the planning budget. Explain that programs revisit equipment throughout their routes.

## 1:30–3:30 — Scenario review

Use the guided controls: DRAM Node B, +10%, 2027Q2 through 2027Q2. Select Compare scenario. Inspect 2027Q2, show the equipment loading change, and read the decision brief. Inspect another quarter to show that a one-quarter request leaves other periods unchanged.

If your own AI connection is enabled, open the planning-note panel and use: “Increase DRAM Node B wafer starts by 10% at Fab A during 2027Q2 only.” Show the extracted fields and source quote. Copy the reviewed request, then explicitly compare it. Be clear that AI extracts text; Python calculates capacity.

Optionally test five percentage points of EUV availability recovery. This means 70% becomes 75%, not a 5% relative increase. It is an operational assumption requiring confirmation, not a promised improvement.

## 3:30–4:30 — Fab performance

Explain that this is a separate historical simulation window in Q3 2026. Show wafer starts and completions, then cycle-time composition. Explain why engineering holds or dispatch delays can dominate elapsed time even when equipment loading appears manageable. A future scenario does not modify this historical view.

## 4:30–5:00 — Close with the decision

State the constrained equipment, affected quarter, assumptions requiring review, and the next validation needed before committing a plan. Offer the export as review evidence.

## Be ready for these questions

- **Where did the data come from?** Synthetic requests and configuration, plus a SimPy simulation. It is not Micron data.
- **Why 85%?** A demo planning buffer for variability. The correct operating limit depends on equipment, variability and service requirements.
- **What is loading?** Required equipment hours divided by production-available equipment hours.
- **Is the view omitting workload?** No. Only the chart is focused; the model includes all programs and equipment.
- **What does the AI do?** Extracts one request into typed fields and exposes missing information for review.
- **Can it predict lateness or savings?** The current demo screens aggregate capacity; those claims require scheduling, milestones and cost data.
- **Is line yield die yield?** No. It is wafer survival for completed lots.
- **What is WIP here?** Average weekly snapshots expressed using each active lot's original input quantity.
- **Is it ready for a real fab?** It demonstrates a workflow. Production use requires real event data, tool/recipe qualifications, calibration, refresh controls and operational validation.

## What to study

Master the starts → visits → processing hours → availability → loading chain. Learn one-sentence definitions for lithography, etch and deposition. The complete simulation, ten tool groups and historical regression can remain supporting material.
