# Validation — v4

## Completed

- **51 automated tests passed** under Python 3.12. Includes deterministic capacity/forecast/retrieval tests retained from v3, new execution-cohort/OEE tests and the updated dashboard interaction tests. AI backend tests use mocks; the current UI makes no AI calls.
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

## Practical limits

- All fab data is synthetic and independent of Micron. Detailed operation covers 29 June–27 September 2026. Start-request history and future planning retain their separately labelled periods.
- No measured individual-machine OEE exists in the fab simulation. The OEE calculator uses a clearly separate example or user-supplied machine-period records; it is not an inferred fab KPI.
- The data supports five program routes and lot/priority detail, not product SKU, per-step event history or machine-level attribution.
- Equipment-state health is quarter-level and shared by programs, so only the site/equipment filters apply. Historical attainment follows site/program filters, not execution weeks or priority.
- The capacity model uses configured routes, equipment, availability and arrivals; requested starts come from the plan CSV. CSV equipment snapshots alone do not override configuration.
- Equipment releases contribute a full quarter of capacity. Aggregate equipment gaps do not include costs, ROI or qualification implementation plans.
- Future capacity scenarios do not rerun the historical simulation or predict new queues, cycle time or lot due dates.
- Source lot timestamps have minute resolution. WIP uses original input-wafer equivalents. Finite-window WIP and completed-lot cycle time are not presented as an exact Little's Law balance.
- Uploaded OEE records must represent non-overlapping single-recipe periods under a consistent planned-time policy. The app validates arithmetic and record uniqueness, not the truth of source measurements or overlapping time labels.

Internal browser screenshots and test uploads are excluded from the deliverable.
