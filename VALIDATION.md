# v3 validation record

Validated on Python 3.12 with the pinned dependencies.

- **35 automated tests passed.** Checks cover capacity arithmetic, unchanged baselines, quarter boundaries, qualifications, availability recovery, Combined views, completed-cohort metrics, typed AI request reading, grounded scenario explanations, document retrieval, embeddings with a mock client, read-only SQL, evidence validation, hypothetical calculations and Streamlit interactions.
- **Combined capacity is site-specific.** The default quarter includes 19 installed fab/equipment combinations. The Combined view sums starts and counts each site's exceptions; it never offsets a constrained group with unrelated spare capacity.
- **Combined production is correctly weighted.** Counts and aligned WIP snapshots are summed; full-flow cycle time is weighted by completed lots; wafer-survival yield uses the matching completed cohorts.
- **Offline model self-check passed.**
- **Desktop review completed at 1512 × 1100.** All four views, a calculated scenario, source-document preview, and the lower chart rows were exercised. No JavaScript page errors or horizontal document overflow were observed.
- **Mobile review completed at 390 × 844.** No horizontal document overflow was detected.
- **Browser export verified.** The Combined capacity download contains 76 rows: 19 fab/equipment combinations across four quarters, with the selected planning limit.
- **Read-only SQL was exercised against the actual supplied database.** Rejected write/extension attempts did not modify its bytes.
- **Original data preserved.** The 11 CSV tables, 15 TXT documents and SQLite database match the uploaded archive byte-for-byte.

Live OpenAI calls were not made because no user API credentials were supplied. Tests use mock responses for text generation, tool selection and embeddings, together with real local calculations, SQL and keyword document retrieval. A user's own API key is required to verify live extraction, semantic retrieval and generated answers.

Citation and numeric checks catch some unsupported output but do not establish that every interpretation or source association is correct. Human review remains necessary. This validates the synthetic prototype's behavior, not accuracy or deployment readiness for a real fab.
