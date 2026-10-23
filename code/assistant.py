"""
AI Factory Assistant.

Design rule: the language model never does arithmetic. It reads the question,
picks a tool, fills in the inputs, and explains the result. Every number comes
from a tool (the same pandas code the dashboard uses, a read-only SQL query, or
the capacity model). Procedures and context come from document search over
SOPs, maintenance tickets, shift reports, weekly summaries and machine notes.
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

import capacity_model as cm
import plant_metrics as pm

MAX_TOOL_CHARS = 6000
MAX_SQL_ROWS = 50

SYSTEM_PROMPT = """You are the AI Factory Assistant for a simulated three-line assembly plant.
All data is synthetic and covers calendar year 2024, plus a Q1 2025 demand forecast.

Plant facts:
- Lines L1, L2, L3. Machines M01-M15 are on L1, M16-M30 on L2, M31-M45 on L3.
- Schedule: Monday-Friday, three 8-hour shifts (MORNING, EVENING, NIGHT) = 120 scheduled hours per line per week.
- Each shift runs 4 SKUs in 2-hour blocks. Materials come from suppliers SUP_A to SUP_D.
- Failures (for MTBF/MTTR) are MECH_FAILURE and SENSOR_FAULT. Jams and changeovers are stoppages, not failures.
- Capacity: loading above 85% is at risk; above 100% the line cannot meet demand.

Current dashboard selection: line={line}, machine={machine}, dates {start} to {end}.
Use this selection when the user does not say otherwise.

Rules:
1. Never calculate, estimate or invent numbers. Every number in your answer must appear in a tool result
   from this conversation. If no specific tool gives it, use run_sql.
2. Use search_documents for procedures, root causes, operator notes and maintenance history.
   For "why" or "what should we do" questions, combine data tools with documents.
3. If the tools and documents don't contain the answer, say so plainly.
4. Lead with the direct answer, then 2-5 short bullets. For recommendations, name the lever and the
   number behind it.
5. End with a line starting "Sources:" listing the tools and document file names you used.
"""


def _json(obj) -> str:
    text = json.dumps(obj, default=str)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + "...(truncated)"


def db_schema(db_path: Path) -> str:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        parts = []
        for t in tables:
            cols = ", ".join(r[1] for r in conn.execute(f"PRAGMA table_info('{t}')"))
            parts.append(f"{t}({cols})")
    return "\n".join(parts)


def run_readonly_sql(db_path: Path, query: str) -> dict:
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise ValueError("Only one statement is allowed.")
    if not re.match(r"(?is)^\s*(select|with)\b", q):
        raise ValueError("Only SELECT queries are allowed.")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    steps = {"n": 0}

    def guard():
        steps["n"] += 1
        return 1 if steps["n"] > 20000 else 0   # abort runaway queries

    conn.set_progress_handler(guard, 1000)
    try:
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    return {"sql": q, "row_count": len(df), "rows": df.head(MAX_SQL_ROWS).to_dict(orient="records"),
            "truncated": len(df) > MAX_SQL_ROWS}


def make_tools(data: dict, scope: dict, db_path: Path, search_fn=None) -> list:
    """Build the tools the model can call. `scope` holds the dashboard's current filters."""
    unit_cost = data["skus"].set_index("sku_id")["unit_cost"]

    def _scope(line=None, machine=None, start_date=None, end_date=None):
        return (line or scope["line"], machine or scope.get("machine", "All"),
                start_date or scope["start"], end_date or scope["end"])

    def get_kpis(line: Optional[str] = None, machine: Optional[str] = None,
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """OEE (availability x performance x quality), scrap rate, plan attainment, line downtime hours,
        MTBF (hours between failures) and MTTR (minutes) for a line and/or machine over a date range.
        line: L1, L2, L3 or All. machine: M01-M45 or All. Dates: YYYY-MM-DD. Omit any argument to use the
        dashboard selection. OEE parts are line-level; with a machine, MTBF/MTTR are for that machine."""
        ln, m, s, e = _scope(line, machine, start_date, end_date)
        return _json(pm.compute_kpis(data, ln, m, s, e))

    def downtime_breakdown(group_by: Literal["machine", "cause", "shift", "line"] = "machine",
                           line: Optional[str] = None, start_date: Optional[str] = None,
                           end_date: Optional[str] = None, failures_only: bool = False,
                           top_n: int = 10) -> str:
        """Downtime minutes, stop counts and failure counts grouped by machine, cause, shift or line,
        sorted worst first, with each group's share and cumulative share (a Pareto)."""
        ln, _, s, e = _scope(line, None, start_date, end_date)
        _, ev, _ = pm.filter_data(data, ln, "All", s, e)
        t = pm.downtime_breakdown(ev, group_by, failures_only).head(top_n)
        return _json({"scope": {"line": ln, "start": s, "end": e}, "rows": t.to_dict(orient="records")})

    def scrap_breakdown(group_by: Literal["line", "supplier", "shift", "sku", "line_supplier", "line_shift"] = "line",
                        line: Optional[str] = None, start_date: Optional[str] = None,
                        end_date: Optional[str] = None, top_n: int = 10) -> str:
        """Produced units, scrap units, scrap rate % and scrap cost grouped by line, supplier, shift, SKU,
        line x supplier or line x shift, sorted by scrap rate (worst first)."""
        ln, _, s, e = _scope(line, None, start_date, end_date)
        p, _, _ = pm.filter_data(data, ln, "All", s, e)
        t = pm.scrap_breakdown(p, group_by, unit_cost).head(top_n)
        overall = round(p["scrap_units"].sum() / p["produced_units"].sum() * 100, 2) if len(p) else None
        return _json({"scope": {"line": ln, "start": s, "end": e}, "overall_scrap_rate_pct": overall,
                      "rows": t.to_dict(orient="records")})

    def machine_reliability(line: Optional[str] = None, start_date: Optional[str] = None,
                            end_date: Optional[str] = None, top_n: int = 10) -> str:
        """Per-machine failures, MTBF (hours) and MTTR (minutes), worst MTBF first."""
        ln, _, s, e = _scope(line, None, start_date, end_date)
        p, ev, _ = pm.filter_data(data, ln, "All", s, e)
        t = pm.machine_reliability_table(p, ev, data["machines"]).head(top_n)
        return _json({"scope": {"line": ln, "start": s, "end": e}, "rows": t.to_dict(orient="records")})

    def run_capacity_scenario(line: str = "All", demand_growth_pct: float = 0.0,
                              availability_gain_pts: float = 0.0, scrap_reduction_pts: float = 0.0,
                              extra_shifts_per_week: int = 0, fix_machine: Optional[str] = None) -> str:
        """Capacity model for the Q1 2025 demand forecast. Returns per-line scheduled, available and required
        hours per week, loading %, gap hours, extra weekend shifts needed, status, and ranked levers.
        Scenario inputs: demand growth %, availability gain (points), scrap reduction (points), extra
        weekend shifts per week (0-6), and fix_machine (e.g. 'M11' = bring that machine's downtime down to
        its line's median). Uses the last 13 weeks of 2024 for availability, performance and quality."""
        return _json(cm.run_scenario(data, line, demand_growth_pct, availability_gain_pts,
                                     scrap_reduction_pts, extra_shifts_per_week, fix_machine))

    def run_sql(query: str) -> str:
        """Run ONE read-only SQLite SELECT against the plant database when no other tool fits.
        Dates are ISO text (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). Returns up to 50 rows. Tables:\n"""
        return _json(run_readonly_sql(db_path, query))

    run_sql.__doc__ += db_schema(db_path)

    tools = [
        StructuredTool.from_function(get_kpis),
        StructuredTool.from_function(downtime_breakdown),
        StructuredTool.from_function(scrap_breakdown),
        StructuredTool.from_function(machine_reliability),
        StructuredTool.from_function(run_capacity_scenario),
        StructuredTool.from_function(run_sql),
    ]

    if search_fn is not None:
        def search_documents(query: str, k: int = 4) -> str:
            """Search SOPs, maintenance tickets, shift reports, weekly huddle summaries and machine operator
            notes. Returns the most relevant passages with their file names."""
            return _json(search_fn(query, min(max(k, 1), 8)))

        tools.append(StructuredTool.from_function(search_documents))
    return tools


# ------------------------------------------------------------------ documents
def load_documents(docs_dir: Path):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = [Document(page_content=f.read_text(), metadata={"source": f.name, "folder": f.parent.name})
            for f in sorted(docs_dir.rglob("*.txt"))]
    return RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(docs)


def build_doc_search(docs_dir: Path, embeddings):
    """Embed the documents into an in-memory Chroma index and return a search function.

    Built at runtime (once per app process) instead of committing a saved index, so
    the index can never go out of sync with the documents or the Chroma version.
    """
    from langchain_chroma import Chroma

    chunks = load_documents(docs_dir)
    store = Chroma.from_documents(chunks, embeddings, collection_name="plant_documents")

    def search(query: str, k: int = 4):
        hits = store.similarity_search(query, k=k)
        return [{"source": h.metadata.get("source"), "folder": h.metadata.get("folder"),
                 "text": h.page_content} for h in hits]

    return search, len(chunks)


# ------------------------------------------------------------------ agent loop
def answer_question(question: str, history: list, llm, tools: list, scope: dict, max_steps: int = 6) -> dict:
    """Tool-calling loop: model -> tool calls -> results -> model ... -> final answer."""
    tool_map = {t.name: t for t in tools}
    msgs = [SystemMessage(SYSTEM_PROMPT.format(**{k: scope.get(k, "All") for k in ("line", "machine", "start", "end")}))]
    for m in history[-6:]:
        msgs.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    msgs.append(HumanMessage(question))

    model = llm.bind_tools(tools)
    trace = []
    for _ in range(max_steps):
        reply = model.invoke(msgs)
        msgs.append(reply)
        if not getattr(reply, "tool_calls", None):
            return {"answer": reply.content, "trace": trace}
        for call in reply.tool_calls:
            tool = tool_map.get(call["name"])
            try:
                result = tool.invoke(call["args"]) if tool else f"ERROR: unknown tool {call['name']}"
            except Exception as exc:  # report the error to the model so it can recover
                result = f"ERROR: {exc}"
            trace.append({"tool": call["name"], "args": call["args"], "result": result})
            msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    msgs.append(HumanMessage("Answer now using only the tool results above."))
    final = llm.bind_tools(tools, tool_choice="none").invoke(msgs)
    return {"answer": final.content, "trace": trace}
