# AI and data methodology

The copilot uses the original structured-data-plus-document approach, now integrated with the stakeholder dashboard.

| Evidence | How it is read | Example insight |
|---|---|---|
| Current dashboard | Selected site, quarter, limit, baseline/scenario and computed equipment rows | Identify the current constrained fab/equipment combination |
| Structured tables | Read-only SQLite queries and deterministic capacity/forecast functions | Compare requested vs actual starts or examine lot cycle-time components |
| Unstructured planning files | Chunking, embeddings, keyword matching and retrieved passages with filenames | Explain documented engineering use, equipment release plans or process-change context |

The language model synthesizes the evidence. Sources marked S are structured calculations or query results; sources marked D are document passages. Citation and numeric checks reject some unsupported output, while the user reviews the meaning and relevance of the evidence.

## What to demonstrate

- “Compare the two fabs: which constraints and documented actions matter most?”
- “Which programs drive etch workload, and what mitigations are documented?”
- “What do operating metrics and engineering notes suggest we investigate first?”
- “Can NAND workload move to another qualified fab, and what happens to capacity?”
- “How do historical requested starts compare with actual starts?”
- “Which process-change notices affect etch or deposition, and what needs reconciliation?”

## What is available now

Eleven structured CSV/SQLite tables and fifteen TXT planning documents are already supplied. No additional data is required for this demo. The document index reads TXT and Markdown files in `data/fab/documents`; the current supplied files are TXT. Regenerate the index automatically by changing the document content. New table content must remain consistent between CSVs, model configuration and SQLite.

With a valid configured API key, document embeddings use `text-embedding-3-small` and the chat model uses `OPENAI_MODEL` (default `gpt-4o-mini`). This corpus is small enough for an in-memory vector index. Keyword retrieval remains an identified fallback. The dashboard and evidence preview work offline.

SQL is read-only and bounded. Hypothetical scenario tools operate on a copy of the plan. Neither tool commits a change or modifies source files.

## What remains outside scope

This prototype does not claim real-time ingestion, PDF/Word upload processing, real-fab accuracy, autonomous approval, exact scheduling, cost savings or ROI. AI interpretation can still be wrong even when a number occurs in cited evidence; inspect the source association and the operating assumptions.
