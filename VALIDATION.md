# Validation — v5

## Completed

- **72 automated tests passed** under Python 3.12. Includes deterministic capacity/forecast/retrieval tests retained from v3, new execution-cohort/OEE tests and the updated dashboard interaction tests. AI backend tests use mocks; the current UI makes no AI calls.
- Offline capacity/scenario checks passed; source files compiled successfully.
- All-program reconstructed weekly starts, output and WIP match the original published weekly tables exactly across the 13-week window.
- HOT + NORMAL partitions and the five program partitions reconcile to total weekly flow and WIP.
- Filtered completed cohorts use the selected date interval; combined weekly cycle time is weighted by actual completed lots. Empty cohorts and short-loop exclusions are explicit.
- Earlier-cutoff WIP ages are reconstructed from timestamps. Future completions/output are hidden; route progress is unavailable for historical cutoffs because no intermediate progress events exist.
- Default completed full-flow queue + hold share verified at approximately 40.04%, rather than the proposed >50% claim.
- Exact target WSPW applies independently to each affected quarter while preserving baseline inputs and unaffected quarters.
- Machine OEE matches the known worked example: 75% availability × 83.333% performance × 98% quality = 61.25% OEE. Tests reject missing, duplicate, non-finite, negative, inconsistent and >100% performance inputs. Empty-output behavior is explicit.
- Browser checks covered all production tabs, a real two-record OEE CSV upload, capacity/scenario views and lot detail/export. No JavaScript page errors were observed.
- Desktop layout inspected at 1512×1100 and mobile at 390×844. No document-level horizontal overflow. Tabs scroll on mobile; KPI cards stack.
- Actual capacity download contained 76 rows (19 fab/equipment combinations × 4 quarters). Actual WIP lot export contained 359 lots with no future completion/output fields.
- SHA-256 comparison confirmed all **27 original data files unchanged**: 11 CSVs, 15 text documents and one SQLite database.

## Additional v5 checks

- 2027 demand totals reconcile to the input plan at quarter, fab and program levels. Requested rates: 2,790 / 2,900 / 3,010 / 3,040 WSPW. Expected rates: 2,678 / 2,763 / 2,850 / 2,867 WSPW.
- Program/family/route charts preserve totals; Combined equals the sum of fabs and the sum of programs. Filtering a forecast does not alter the capacity baseline.
- Held-out predictions do not change when held-out actuals are modified, confirming training separation. MAPE is evaluated on the same individual held-out records.
- Weekly calendar covers exactly 365 days of 2027 in 53 clipped buckets. Partial weeks use partial hours. Summed weekly EUV hours reconcile to the corresponding quarterly rates integrated over calendar days.
- Weeks crossing quarterly demand or equipment releases prorate both demand and capacity by exact overlap. Exact-target and availability-recovery changes are confined to the selected weeks.
- Outside-window rows, the other fab and source plan remain unchanged.
- Scenario UI names fab and equipment, defaults to the affected fab, optionally shows both, and labels HOT priority. HOT/NORMAL changes metadata without changing capacity arithmetic.
- Browser inspection covered both demand tabs, named-program expected-start charts, weekly HOT scenarios and their scope. Actual demand export contained 32 rows with reconciled quarter totals. Actual weekly export contained 530 affected-fab rows (10 groups × 53 weeks), HOT metadata, no Fab B rows, and zero changes outside the request window. No JavaScript errors or desktop/mobile horizontal overflow were observed.

## Practical limits

- All fab data is synthetic and independent of Micron. Detailed operation covers 29 June–27 September 2026. Start-request history and future planning retain their separately labelled periods.
- No measured individual-machine OEE exists in the fab simulation. The OEE calculator uses a clearly separate example or user-supplied machine-period records; it is not an inferred fab KPI.
- The data supports five program routes and lot/priority detail, not product SKU, per-step event history or machine-level attribution.
- Equipment-state health is quarter-level and shared by programs, so only the site/equipment filters apply. Historical attainment follows site/program filters, not execution weeks or priority.
- The capacity model uses configured routes, equipment, availability and arrivals; requested starts come from the plan CSV. CSV equipment snapshots alone do not override configuration.
- Quarter screens count releases from their configured quarter. Weekly screens prorate at that quarter boundary. Aggregate equipment gaps do not include costs, ROI or qualification implementation plans.
- Future quarter/week capacity scenarios do not rerun the historical simulation or predict new queues, cycle time or lot due dates. HOT priority is not a capacity multiplier or a simulated dispatch change.
- Source lot timestamps have minute resolution. WIP uses original input-wafer equivalents. Finite-window WIP and completed-lot cycle time are not presented as an exact Little's Law balance.
- Uploaded OEE records must represent non-overlapping single-recipe periods under a consistent planned-time policy. The app validates arithmetic and record uniqueness, not the truth of source measurements or overlapping time labels.

Internal browser screenshots and test uploads are excluded from the deliverable.
