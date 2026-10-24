# Data and deferred AI integration — v4

AI features are paused in the stakeholder dashboard. `code/apprag.py` imports neither `assistant.py` nor `planning_copilot.py`, reads no API key and makes no external AI calls. There is no AI page, request parser, explanation button, connection status or document-preview panel in the UI.

The legacy modules and their automated tests remain in the repository for future integration. A later "Ask about this view" interface can reuse structured table queries, document retrieval, typed request extraction and the deterministic capacity/scenario engine. It should declare the selected scope and period, expose proposed changes, calculate through the model and cite supporting notes. A document's historical route change must be reconciled with the baseline before applying it again.

The existing 11 structured tables and 15 text documents remain unchanged. The current UI uses deterministic functions and the original data. Family labels are a display mapping over the five existing programs, not additional product data.

Machine OEE is a separate calculator: its example or uploaded records do not join the fab simulation and do not alter historical metrics or future capacity. Its required schema and definitions are documented in README.md and in `templates/machine_oee_template.csv`.
