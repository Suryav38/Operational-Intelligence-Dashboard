# Validation record

Validated on Python 3.12 with the pinned application dependencies.

- **18 automated tests passed.** Coverage includes known EUV arithmetic, complete workload preservation behind display filters, quarter-scoped scenarios, baseline immutability, qualification rules, percentage-point availability recovery, completed-cohort production metrics, AI extraction with a mock client, and Streamlit interactions.
- **Offline self-check passed.** Capacity, scenario and production calculations run without an API key.
- **Desktop browser review completed at 1512 × 1100.** All three pages were exercised, including a scenario calculation. Chart labels, legends, tables and lower-page content were visually inspected.
- **Mobile browser review completed at 390 × 844.** No horizontal document overflow was detected. No JavaScript page errors were observed in the reviewed flows.
- **Capacity download exercised in the browser.** The exported Fab A review contains all 10 equipment groups across all four quarters (40 rows), with requested-demand basis and the selected 85% planning limit.
- **Original data preserved.** All 27 source data files (11 CSV tables, 15 text documents and the SQLite database) match the uploaded ZIP byte-for-byte.
- **Python source compilation passed.**

The live OpenAI service was not called: no user API credentials were supplied. Its typed extraction contract, source-quote validation and review/copy workflow were tested offline. A user's own valid API key and compatible model are required to verify live extraction.

This validation establishes the behavior of the synthetic prototype. It does not establish suitability or accuracy for a real semiconductor fab.
