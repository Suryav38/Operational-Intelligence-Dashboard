"""
AI planning copilot for the TD Fab Capacity & Wafer Loading Planner.

Three AI jobs, one rule. The rule: the language model never calculates. Numbers
come from the capacity model, the forecast model, the simulation tables or a
read-only SQL query; the model chooses what to run and explains the result.

  1. extract_changes   - turns a free-text route-change / plan note into
                         structured PlanChange records for a person to confirm.
  2. answer_question   - tool-calling chat: picks tools, runs them, explains.
  3. draft_readout     - writes a short leadership readout from a scenario
                         result, using only the numbers in that result.
Document search (retrieval over planning-review minutes, route-change notices,
tool notes and guidelines) supplies context and past decisions.
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import fab_capacity as fc
import fab_config as C
import fab_data as fd
import forecast_model as fm

MAX_TOOL_CHARS = 7000
MAX_SQL_ROWS = 50

CODES = ("Programs: " + ", ".join(f"{k} = {v['name']}" for k, v in C.PROGRAMS.items())
         + ". Sites: FAB_A (has EUV), FAB_B (DUV only). Tool groups: "
         + ", ".join(f"{k} = {v['name']}" for k, v in C.TOOL_GROUPS.items())
         + f". Planning quarters: {', '.join(C.PLAN_QUARTERS)}.")

SYSTEM_PROMPT = f"""You are the planning copilot for a simulated network of two Technology Development (TD)
semiconductor fabs. All data is synthetic and illustrative.
{CODES}
Qualification: FAB_A runs all programs; FAB_B runs DRAM_A, NAND_C and SHORT only.
Loading = required tool hours / available production hours. Above {int(C.LOADING_CEILING * 100)}% is at risk;
above 100% the plan cannot run. "requested" = program teams' wafer-start requests; "corrected" = requests
adjusted by each program's historical realization rate (forecast model).
Fab performance data is a 13-week SimPy simulation of Q3 2026.

Current dashboard selection: site={{site}}, forecast basis={{basis}}.

Rules:
1. Never calculate, estimate or invent numbers. Every number you state must appear in a tool result in this
   conversation. Use run_sql if no other tool gives it.
2. For "what if" questions, call run_scenario with structured changes. State which changes you applied.
3. Use search_documents for past decisions, route-change notices, tool notes and planning guidelines.
4. If the data does not answer the question, say so.
5. Lead with the answer, then at most 5 short bullets. Name the constraint, the quarter, the loading and the
   option that addresses it.
6. End with "Sources:" listing the tools and documents used.
"""


class ChangeSet(BaseModel):
    """Structured reading of a planning note."""
    changes: List[fc.PlanChange] = Field(description="One record per distinct change in the note")
    summary: str = Field(description="One sentence describing the note")
    open_questions: List[str] = Field(default_factory=list,
                                      description="Anything ambiguous a planner should confirm before running")


EXTRACT_PROMPT = f"""You convert planning notes from Process Integration, program teams and equipment groups into
structured changes for a fab capacity model. {CODES}
Rules:
- Use only codes from the lists above. Map plain names to codes (e.g. "Node B" -> DRAM_B, "HBM" -> HBM_TV).
- A route change that swaps layers is two records (e.g. EUV +2 and DUV -2). Extra etch/clean/deposition steps
  per added layer are separate route_change records.
- "Pull in by one quarter" -> pull_in with quarters=1. A slipped tool delivery -> tool_delay.
- Extra wafers per week -> starts_change with wspw_delta; a percentage -> pct.
- Put a short quote from the note in source_text.
- Do not guess missing values. If a needed value (quarter, amount, site) is unclear, leave it out and add an
  open question instead.
"""

BRIEF_RULES = """Use only numbers that appear in the JSON. Qualitative facts (lead times, qualification, past
decisions) may come only from "context_documents". Write for a director: short, specific, no preamble.
Use exactly these headings, each followed by one or two short sentences or bullets:
**Recommendation:**
**Why:**
**Trade-off:**
**Uncertainty:**
**Needs sign-off:** (which of Engineering, Process Integration, Operations, Finance must confirm what)
"""

READOUT_PROMPT = ("Write a decision brief for a TD quarterly planning review about ONE proposed plan change, "
                  "using the scenario result JSON. The recommendation is one of: proceed, proceed with a named "
                  "mitigation, or do not proceed.\n" + BRIEF_RULES)

COMPARE_PROMPT = ("Write a decision brief for a TD quarterly planning review comparing the options in the JSON "
                  "against the baseline. Recommend one option (or a combination, if the numbers support it) and "
                  "say what it leaves unresolved.\n" + BRIEF_RULES)

FALLBACK_MODELS = ["gpt-4.1-mini", "gpt-4o-mini"]


def _json(obj) -> str:
    text = json.dumps(obj, default=str)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + "...(truncated)"


# ------------------------------------------------------------------ SQL
def db_schema(db_path: Path) -> str:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return "\n".join(f"{t}({', '.join(r[1] for r in conn.execute(f'PRAGMA table_info({chr(39)}{t}{chr(39)})'))})"
                         for t in tables)


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
        return 1 if steps["n"] > 20000 else 0

    conn.set_progress_handler(guard, 1000)
    try:
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()
    return {"sql": q, "row_count": len(df), "rows": df.head(MAX_SQL_ROWS).to_dict(orient="records"),
            "truncated": len(df) > MAX_SQL_ROWS}


# ------------------------------------------------------------------ tools
def make_tools(data: dict, ctx: dict, db_path: Path = fd.DB_PATH, search_fn=None) -> list:
    """ctx holds the dashboard selection: site and forecast basis."""

    def fab_performance(site: Literal["Both", "FAB_A", "FAB_B"] = "Both") -> str:
        """Q3 2026 fab performance from the simulation: wafer starts vs plan, outs, moves per day, WIP,
        cycle time (hot vs normal), days per mask layer, X-factor, line yield, cycle-time breakdown
        (process / queue / transport / engineering hold), and per-tool-group loading and queue time."""
        tg = fd.tool_group_states(data, site)[["site", "tool_group", "tools", "loading_pct", "engineering_pct",
                                               "unsched_down_pct", "avg_queue_h"]]
        return _json({"summary": fd.performance_summary(data, site),
                      "cycle_time_by_program": fd.cycle_time_by_program(data, site).to_dict(orient="records"),
                      "tool_groups": tg.sort_values("loading_pct", ascending=False).head(8).to_dict(orient="records")})

    def capacity_plan(basis: Optional[Literal["requested", "corrected"]] = None,
                      site: Optional[Literal["FAB_A", "FAB_B"]] = None,
                      tool_group: Optional[str] = None) -> str:
        """2027 capacity plan: constrained cells (site x tool group x quarter at or above the loading ceiling)
        with loading %, installed tools, tools needed and the programs driving each. basis: requested or
        corrected (defaults to the dashboard selection). Optional site / tool_group filters."""
        b = basis or ctx["basis"]
        state = fc.base_state(data, b)
        t = fc.loading_table(state)
        if site:
            t = t[t["site"] == site]
        if tool_group:
            t = t[t["tool_group"] == tool_group.upper()]
        cons = fc.constraints(t)
        rows = [{**r, "drivers": fc.drivers(state, r["site"], r["tool_group"], r["quarter"])[:3]}
                for r in cons.drop(columns=["required_h_wk", "available_h_wk"]).to_dict(orient="records")]
        peak = t.sort_values("loading_pct", ascending=False).head(5)[["site", "tool_group", "quarter", "loading_pct"]]
        return _json({"basis": b, "ceiling_pct": C.LOADING_CEILING * 100, "constrained_cells": len(rows),
                      "constraints": rows, "highest_loading_cells": peak.to_dict(orient="records")})

    def forecast_accuracy() -> str:
        """How well wafer-start requests predicted actual starts, by program: realization rate (actual /
        requested from a regression through the origin), average over-request %, and backtest error
        (MAPE) of raw requests vs corrected forecasts, plus 2027 requested vs corrected totals."""
        return _json(fm.summary(data["wafer_start_history"], data["plan_2027"]))

    def run_scenario(changes: List[fc.PlanChange],
                     basis: Optional[Literal["requested", "corrected"]] = None) -> str:
        """Run a what-if on the 2027 plan. changes: list of PlanChange records (route_change, pull_in,
        starts_change, tool_delay, tool_add, availability_change, move_load). Returns applied changes, warnings,
        constraints before/after, new and cleared constraints, the largest loading changes, and the worst cells
        with their program drivers."""
        res = fc.run_scenario(data, changes, basis or ctx["basis"])
        return _json({k: v for k, v in res.items() if not k.startswith("_")})

    def run_sql(query: str) -> str:
        """Run ONE read-only SQLite SELECT against the planning and simulation database when no other tool
        fits. Returns up to 50 rows. Tables:\n"""
        return _json(run_readonly_sql(db_path, query))

    run_sql.__doc__ += db_schema(db_path)

    tools = [StructuredTool.from_function(f) for f in
             (fab_performance, capacity_plan, forecast_accuracy, run_scenario, run_sql)]
    if search_fn is not None:
        def search_documents(query: str, k: int = 4) -> str:
            """Search planning-review minutes, process change notices, tool-group notes and TD capacity
            planning guidelines. Returns the most relevant passages with file names."""
            return _json(search_fn(query, min(max(k, 1), 8)))

        tools.append(StructuredTool.from_function(search_documents))
    return tools


# ------------------------------------------------------------------ documents
def build_doc_search(docs_dir: Path, embeddings):
    """Embed the documents into an in-memory Chroma index (built once per app process)."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = [Document(page_content=f.read_text(), metadata={"source": f.name, "folder": f.parent.name})
            for f in sorted(docs_dir.rglob("*.txt"))]
    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120).split_documents(docs)
    store = Chroma.from_documents(chunks, embeddings, collection_name="td_planning_documents")

    def search(query: str, k: int = 4):
        return [{"source": h.metadata.get("source"), "folder": h.metadata.get("folder"), "text": h.page_content}
                for h in store.similarity_search(query, k=k)]

    return search, len(chunks)


# ------------------------------------------------------------------ AI jobs
def extract_changes(llm, note: str) -> ChangeSet:
    model = llm.with_structured_output(ChangeSet, method="function_calling")
    return model.invoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(note)])


def context_documents(changes, cells) -> list:
    """Deterministic context for briefs: the planning guideline plus tool notes for the tool groups involved."""
    keys = set()
    for ch in changes:
        c = ch if isinstance(ch, dict) else ch.model_dump()
        if c.get("site") and c.get("tool_group"):
            keys.add((c["site"], c["tool_group"]))
        if c.get("type") == "move_load" and c.get("to_site"):
            keys.update((c["to_site"], tg) for tg in ("ETCH", "DEP", "EUV"))
    for cell in cells:
        parts = cell.split()
        if len(parts) >= 2:
            keys.add((parts[0], parts[1]))
    docs = []
    for site, tg in sorted(keys):
        f = fd.DOCS / "tool_notes" / f"tool_note_{site}_{tg}.txt"
        if f.exists():
            docs.append({"source": f.name, "text": f.read_text()})
    g = fd.DOCS / "guidelines" / "td_capacity_planning_guidelines.txt"
    docs.append({"source": g.name, "text": g.read_text()})
    return docs


def draft_readout(llm, result: dict, changes=()) -> str:
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    payload["context_documents"] = context_documents(changes, result.get("new_constraints", []))
    return llm.invoke([SystemMessage(READOUT_PROMPT), HumanMessage(_json(payload))]).content


def draft_comparison(llm, comparison: dict, option_changes: dict) -> str:
    payload = dict(comparison)
    cells = [c for o in comparison["options"] for c in o["new_constraints"]]
    payload["context_documents"] = context_documents([c for v in option_changes.values() for c in v], cells)
    return llm.invoke([SystemMessage(COMPARE_PROMPT), HumanMessage(_json(payload))]).content


# ------------------------------------------------------------------ OpenAI connection
def explain_error(exc, model: str = "") -> str:
    """Turn an OpenAI error into a plain-English fix."""
    import openai
    msg = str(exc)
    if isinstance(exc, openai.AuthenticationError):
        return ("OpenAI rejected the API key. Create a new key at platform.openai.com/api-keys and paste it "
                "into the app's Secrets as OPENAI_API_KEY.")
    if isinstance(exc, openai.RateLimitError) and ("insufficient_quota" in msg or "credit" in msg):
        return ("The OpenAI account behind this key has no credit left. Add credit at "
                "platform.openai.com/settings/organization/billing, wait a few minutes, then try again.")
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI's rate limit was hit. Wait a minute and try again."
    if isinstance(exc, openai.NotFoundError) or "model_not_found" in msg:
        return (f"The model '{model}' isn't available to this key. Add OPENAI_MODEL to the app's Secrets "
                "with a model your account can use, for example gpt-4.1-mini.")
    if isinstance(exc, openai.PermissionDeniedError):
        return "This API key doesn't have permission for that model or project. Check the key's project settings."
    if isinstance(exc, openai.APIConnectionError):
        return "The app couldn't reach OpenAI. Reboot the app in Streamlit Cloud and try again."
    return f"{type(exc).__name__}: {msg[:300]}"


def make_llm(key: str, model: str, **kw):
    from langchain_openai import ChatOpenAI
    reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
    return ChatOpenAI(model=model, api_key=key, **({} if reasoning else {"temperature": 0}), **kw)


def pick_model(key: str, preferred: str) -> str:
    """Return the first model this key can use: the preferred one, then common fallbacks."""
    import openai
    tried = []
    for m in [preferred] + [x for x in FALLBACK_MODELS if x != preferred]:
        try:
            make_llm(key, m, max_tokens=5, max_retries=0).invoke("Reply OK")
            return m
        except openai.NotFoundError:
            tried.append(m)
        except openai.BadRequestError as exc:
            if "model" not in str(exc).lower():
                raise
            tried.append(m)
    raise RuntimeError(f"None of these models are available to this key: {', '.join(tried)}. "
                       "Add OPENAI_MODEL to the app's Secrets with a model your account can use.")


def check_connection(key: str, preferred: str):
    """Returns (ok, message) after testing the chat model and the embeddings model."""
    from langchain_openai import OpenAIEmbeddings
    try:
        model = pick_model(key, preferred)
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=key, max_retries=0).embed_query("test")
        note = "" if model == preferred else f" ('{preferred}' isn't available, so it uses {model}.)"
        return True, f"Connected. Chat model {model} and document search both work.{note}"
    except Exception as exc:
        return False, explain_error(exc, preferred)


def answer_question(question: str, history: list, llm, tools: list, ctx: dict, max_steps: int = 6) -> dict:
    tool_map = {t.name: t for t in tools}
    msgs = [SystemMessage(SYSTEM_PROMPT.format(site=ctx.get("site", "Both"), basis=ctx.get("basis", "requested")))]
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
            except Exception as exc:
                result = f"ERROR: {exc}"
            trace.append({"tool": call["name"], "args": call["args"], "result": result})
            msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    msgs.append(HumanMessage("Answer now using only the tool results above."))
    final = llm.bind_tools(tools, tool_choice="none").invoke(msgs)
    return {"answer": final.content, "trace": trace}
