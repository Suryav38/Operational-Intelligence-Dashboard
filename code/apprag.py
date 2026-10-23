import json
import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import assistant as ai
import capacity_model as cm
import plant_metrics as pm

# ======================================================================
# 1. Page setup
# ======================================================================
st.set_page_config(page_title="Operational Intelligence Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .kpi {background: var(--secondary-background-color); padding: 14px 10px; border-radius: 10px;
        border-top: 4px solid #3498db; text-align: center;}
  .kpi-label {font-size: 0.75rem; text-transform: uppercase; letter-spacing: .5px; font-weight: 600; opacity: .75;}
  .kpi-value {font-size: 1.7rem; font-weight: 700; margin: 4px 0;}
  .kpi-sub {font-size: 0.75rem; opacity: .7;}
  .flag {padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: .9rem; border-left: 4px solid;}
  .flag-Quality {background: rgba(241,196,15,.12); border-color: #f1c40f;}
  .flag-Reliability {background: rgba(231,76,60,.10); border-color: #e74c3c;}
</style>
""", unsafe_allow_html=True)

GOOD, BAD = "#27ae60", "#e74c3c"
load_dotenv(pm.ROOT / ".env")


@st.cache_data(show_spinner=False)
def get_data():
    return pm.load_data()


def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file locally
        value = None
    return value or os.getenv(name, default)


@st.cache_resource(show_spinner=False)
def get_llm(model, key):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0, api_key=key)


@st.cache_resource(show_spinner="Indexing plant documents (first question only)...")
def get_doc_search(key):
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)
    search, _ = ai.build_doc_search(pm.DOCS_DIR, embeddings)
    return search


try:
    data = get_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ======================================================================
# 2. Sidebar filters
# ======================================================================
st.sidebar.title("Filters")
lines = ["All"] + sorted(data["prod"]["line_id"].unique())
sel_line = st.sidebar.selectbox("Line", lines)
mach = data["machines"] if sel_line == "All" else data["machines"][data["machines"]["line_id"] == sel_line]
sel_machine = st.sidebar.selectbox("Machine", ["All"] + sorted(mach["machine_id"]),
                                   help="Production is recorded per line, so OEE and scrap stay line-level. "
                                        "Selecting a machine changes the reliability metrics.")
min_d, max_d = pm.date_bounds(data)
c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start", value=max_d - timedelta(days=90), min_value=min_d, max_value=max_d)
end_date = c2.date_input("End", value=max_d, min_value=min_d, max_value=max_d)
if start_date > end_date:
    st.sidebar.error("Start date is after end date.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Assumptions")
dt_cost = st.sidebar.number_input("Downtime cost ($/line-hour)", min_value=0, value=800, step=50)
st.sidebar.caption("Scrap cost uses each SKU's unit cost. All data is synthetic.")

p_f, e_line, e_mach = pm.filter_data(data, sel_line, sel_machine, start_date, end_date)
kpis = pm.compute_kpis(data, sel_line, sel_machine, start_date, end_date, dt_cost)

# ======================================================================
# 3. Header, KPI cards, watch list
# ======================================================================
st.title("Operational Intelligence Dashboard")
st.caption("Simulated three-line assembly plant · synthetic 2024 data · Q1 2025 demand forecast")
st.markdown(f"**Scope:** Line {sel_line} · Machine {sel_machine} · {start_date} to {end_date}")


def card(col, label, value, sub, ok):
    color = "#3498db" if ok is None else (GOOD if ok else BAD)
    col.markdown(f"""<div class="kpi" style="border-top-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


T = pm.TARGETS
mtbf_target = T["mtbf_machine_h"] if sel_machine != "All" else T["mtbf_line_h"]
mtbf_label = f"MTBF ({sel_machine})" if sel_machine != "All" else "MTBF (line)"
cols = st.columns(6)
card(cols[0], "OEE", f"{kpis['oee_pct']}%",
     f"A {kpis['availability_pct']} × P {kpis['performance_pct']} × Q {kpis['quality_pct']:.1f}",
     kpis["oee_pct"] >= T["oee"])
card(cols[1], "Availability", f"{kpis['availability_pct']}%",
     f"Target ≥{T['availability']:.0f}% · {kpis['line_downtime_h']} h down", kpis["availability_pct"] >= T["availability"])
card(cols[2], "Scrap rate", f"{kpis['scrap_rate_pct']}%",
     f"Target ≤{T['scrap_rate']:.0f}% · ${kpis['scrap_cost_usd']:,}", kpis["scrap_rate_pct"] <= T["scrap_rate"])
card(cols[3], mtbf_label, f"{kpis['mtbf_h']} h" if kpis["mtbf_h"] else "n/a",
     f"Target ≥{mtbf_target:.0f} h · {kpis['failures']} failures",
     None if kpis["mtbf_h"] is None else kpis["mtbf_h"] >= mtbf_target)
card(cols[4], "MTTR", f"{kpis['mttr_min']} min" if kpis["mttr_min"] else "n/a",
     f"Target ≤{T['mttr_min']:.0f} min", None if kpis["mttr_min"] is None else kpis["mttr_min"] <= T["mttr_min"])
card(cols[5], "Plan attainment", f"{kpis['plan_attainment_pct']}%",
     f"{kpis['good_units']:,} good units", kpis["plan_attainment_pct"] >= 95)

if sel_machine != "All":
    st.caption(f"OEE, availability and scrap are for line {sel_line} (production is recorded per line). "
               f"MTBF and MTTR are for {sel_machine}.")

flags = pm.exceptions(p_f, e_line)
if flags:
    st.markdown("##### Watch list")
    for f in flags:
        st.markdown(f"<div class='flag flag-{f['type']}'><b>{f['type']}:</b> {f['text']}</div>",
                    unsafe_allow_html=True)

# ======================================================================
# 4. Analytics tabs
# ======================================================================
tab_prod, tab_qual, tab_rel, tab_cap, tab_how = st.tabs(
    ["Production", "Quality", "Reliability", "Capacity planning", "How the AI works"])

with tab_prod:
    c1, c2 = st.columns([3, 2])
    daily = pm.daily_series(p_f)
    with c1:
        st.markdown("##### Good units vs plan (daily)")
        fig = go.Figure()
        fig.add_bar(x=daily["date"], y=daily["good"], name="Good units", marker_color="#3498db")
        fig.add_scatter(x=daily["date"], y=daily["planned_units"], name="Plan", mode="lines",
                        line=dict(color=BAD, width=2))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Weekly OEE and its parts")
        w = pm.weekly_oee(p_f).melt(id_vars="week", var_name="Metric", value_name="Percent")
        fig = px.line(w, x="week", y="Percent", color="Metric")
        fig.add_hline(y=T["oee"], line_dash="dot", annotation_text=f"OEE target {T['oee']:.0f}%")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

with tab_qual:
    unit_cost = data["skus"].set_index("sku_id")["unit_cost"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Scrap rate % by line and supplier")
        ls = pm.scrap_breakdown(p_f, "line_supplier").pivot(index="line_id", columns="supplier_id",
                                                             values="scrap_rate_pct")
        fig = px.imshow(ls, text_auto=".1f", color_continuous_scale="Reds", aspect="auto",
                        labels=dict(color="Scrap %"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Scrap rate % by shift")
        sh = pm.scrap_breakdown(p_f, "shift")
        fig = px.bar(sh, x="shift_id", y="scrap_rate_pct", text="scrap_rate_pct",
                     labels={"shift_id": "Shift", "scrap_rate_pct": "Scrap %"})
        fig.add_hline(y=kpis["scrap_rate_pct"], line_dash="dot", annotation_text="Selection average")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    st.markdown("##### SKUs with the highest scrap cost")
    sku = pm.scrap_breakdown(p_f, "sku", unit_cost).sort_values("scrap_cost_usd", ascending=False).head(8)
    st.dataframe(sku, hide_index=True, width="stretch")

with tab_rel:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### Pareto: downtime by machine")
        par = pm.downtime_breakdown(e_line, "machine").head(15)
        fig = go.Figure()
        fig.add_bar(x=par["machine_id"], y=par["downtime_min"], name="Downtime (min)", marker_color="#e67e22")
        fig.add_scatter(x=par["machine_id"], y=par["cumulative_pct"], name="Cumulative % of all downtime",
                        yaxis="y2", mode="lines+markers", line=dict(color="#2c3e50"))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12),
                          yaxis=dict(title="Minutes"),
                          yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Downtime by cause")
        cause = pm.downtime_breakdown(e_line, "cause")
        cause["type"] = cause["cause_code"].map(lambda c: "Failure" if c in pm.FAILURE_CAUSES else "Stoppage")
        fig = px.bar(cause, x="cause_code", y="downtime_min", color="type",
                     color_discrete_map={"Failure": BAD, "Stoppage": "#95a5a6"},
                     labels={"cause_code": "Cause", "downtime_min": "Minutes"})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
    st.markdown("##### Machines with the lowest MTBF (failures only)")
    st.dataframe(pm.machine_reliability_table(p_f, e_line, data["machines"]).head(10),
                 hide_index=True, width="stretch")

with tab_cap:
    st.markdown("##### Can each line meet the Q1 2025 demand forecast?")
    st.caption("Rates come from the last 13 weeks of 2024; demand from the Q1 2025 forecast. "
               "Move the levers to test a scenario.")
    hist_p, hist_e, _ = cm._history(data)
    worst = hist_e.groupby("machine_id")["duration_min"].sum().sort_values(ascending=False).head(5).index.tolist()
    c = st.columns(6)
    growth = c[0].slider("Demand growth %", -20, 40, 0, 5)
    avail = c[1].slider("Availability gain (pts)", 0.0, 5.0, 0.0, 0.5)
    scrap_cut = c[2].slider("Scrap reduction (pts)", 0.0, 2.0, 0.0, 0.1)
    shifts = c[3].slider("Weekend shifts/week", 0, cm.MAX_EXTRA_SHIFTS, 0)
    fix = c[4].selectbox("Fix machine", ["None"] + worst,
                         help="Bring this machine's downtime down to its line's median machine.")
    ceiling = c[5].slider("Loading ceiling %", 70, 95, 85, 5)
    res = cm.run_scenario(data, "All", growth, avail, scrap_cut, shifts, None if fix == "None" else fix,
                          ceiling / 100)
    df = pd.DataFrame(res["lines"])
    status_color = {"OK": GOOD, "At risk (above ceiling)": "#f39c12", "Cannot meet demand": BAD}

    c1, c2 = st.columns([2, 3])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=df["line"], y=df["loading_pct"], text=df["loading_pct"].map(lambda v: f"{v:.0f}%"),
                    marker_color=df["status"].map(status_color), name="Average week")
        fig.add_scatter(x=df["line"], y=df["peak_week_loading_pct"], mode="markers", name="Peak week",
                        marker=dict(symbol="line-ew-open", size=28, color="#2c3e50"))
        fig.add_hline(y=ceiling, line_dash="dot", annotation_text=f"Ceiling {ceiling}%")
        fig.add_hline(y=100, line_color=BAD, annotation_text="100%")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Loading %",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
    with c2:
        show = df[["line", "status", "loading_pct", "available_h_wk", "required_h_wk", "gap_h_wk",
                   "extra_shifts_to_reach_ceiling", "availability_pct", "performance_pct", "quality_pct"]]
        st.dataframe(show.rename(columns={
            "line": "Line", "status": "Status", "loading_pct": "Loading %", "available_h_wk": "Available h/wk",
            "required_h_wk": "Required h/wk", "gap_h_wk": "Gap h/wk",
            "extra_shifts_to_reach_ceiling": "Weekend shifts needed", "availability_pct": "A %",
            "performance_pct": "P %", "quality_pct": "Q %"}), hide_index=True, width="stretch")
        st.markdown("**Best single levers per line** (on top of this scenario)")
        lever_rows = [{"Line": ln, **lv} for ln, lvs in res["levers"].items() for lv in lvs[:3]]
        st.dataframe(pd.DataFrame(lever_rows).rename(columns={
            "lever": "Lever", "loading_pct": "Loading %", "loading_change_pts": "Change (pts)", "status": "Status"}),
            hide_index=True, width="stretch")
    with st.expander("How the capacity model works"):
        st.markdown("""
- **Available hours/week** = scheduled hours × availability × performance (120 scheduled hours = 3 shifts × 5 days × 8 h)
- **Required hours/week** = Σ over SKUs of (weekly demand ÷ quality yield) × ideal cycle time
- **Loading** = required ÷ available. Above the ceiling the line is at risk: queues and lead times grow
  sharply as loading approaches 100%.
- **Gap** = required − available × ceiling. A positive gap is the shortfall to close.
- **Fix machine** removes that machine's downtime above its line's median machine (a stop anywhere on a
  serial line stops the line).
- Limits: lines are treated as single resources, the SKU mix is fixed per line, and changeover time is
  inside the historical availability.
""")

with tab_how:
    st.markdown("##### How the AI assistant answers a question")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4fb", fontname="Helvetica", fontsize=11];
  q [label="Question in\\nplain English", fillcolor="#fdf2e9"];
  llm [label="GPT (tool calling)\\nchooses tools and inputs", fillcolor="#e8f8f5"];
  k [label="KPI + breakdown functions\\n(same pandas code as the charts)"];
  c [label="Capacity model\\n(scenario levers)"];
  s [label="Read-only SQL\\n(SQLite plant database)"];
  d [label="Document search\\n(SOPs, tickets, shift reports)"];
  a [label="Answer + sources\\n(every step shown)", fillcolor="#fdf2e9"];
  q -> llm; llm -> k; llm -> c; llm -> s; llm -> d; k -> a; c -> a; s -> a; d -> a; llm -> a [style=dashed, label="explains"];
}
""")
    st.markdown("""
**Design choices**
- **The model never does the math.** Every number comes from a tool: the same functions that draw the
  charts, the capacity model, or a read-only SQL query. The model picks the tool and explains the result.
- **Documents add the "why" and "what to do".** Retrieval over SOPs, maintenance tickets, shift reports
  and weekly summaries supplies root causes and procedures.
- **Every answer shows its work.** Expand "How this answer was built" under a reply to see each tool call,
  its inputs, the SQL it ran and the documents it used.

**Guardrails**
- SQL runs on a read-only connection, accepts one SELECT statement, and returns at most 50 rows.
- The assistant is told to say when the data doesn't contain an answer instead of guessing.
- It defaults to the dashboard's current filters, so chart and answer use the same scope.

**Limits**
- The plant and its data are simulated. Connecting a real plant means pointing these tools at the
  MES/ERP database instead of the SQLite file.
""")

# ======================================================================
# 5. AI assistant
# ======================================================================
st.markdown("---")
h1, h2 = st.columns([6, 1])
h1.subheader("AI Factory Assistant")
h1.caption("Ask in plain English. Numbers come from the plant data, procedures from SOPs and logs, "
           "and every answer shows how it was built.")
if h2.button("Clear chat", width="stretch"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

SUGGESTIONS = [
    "Which supplier is driving scrap on Line 2, and what does our procedure say to do?",
    "Why is M11 losing so much time, and how much capacity would fixing it recover on Line 1?",
    "Can Line 2 meet Q1 demand? If not, what's the cheapest way to close the gap?",
]


def render_trace(trace):
    if not trace:
        return
    with st.expander(f"How this answer was built ({len(trace)} step{'s' if len(trace) != 1 else ''})"):
        for i, step in enumerate(trace, 1):
            st.markdown(f"**{i}. `{step['tool']}`**")
            args = step.get("args", {})
            try:
                result = json.loads(step["result"])
            except (TypeError, ValueError):
                result = step["result"]
            if step["tool"] == "run_sql":
                st.code(args.get("query", ""), language="sql")
            elif args:
                st.caption("Inputs: " + ", ".join(f"{k}={v}" for k, v in args.items()))
            if step["tool"] == "search_documents" and isinstance(result, list):
                for hit in result:
                    st.markdown(f"📄 **{hit['source']}** ({hit['folder']})")
                    st.caption(hit["text"][:220].replace("\n", " ") + "...")
            elif isinstance(result, (dict, list)):
                st.json(result, expanded=False)
            else:
                st.caption(str(result)[:500])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_trace(msg.get("trace"))

api_key = get_secret("OPENAI_API_KEY")
if not api_key:
    st.info("The assistant needs an OpenAI API key. Add OPENAI_API_KEY to the app's Streamlit secrets "
            "(or a local .env file). The dashboard works without it.")
else:
    s_cols = st.columns(len(SUGGESTIONS))
    clicked = None
    for col, q in zip(s_cols, SUGGESTIONS):
        if col.button(q, width="stretch"):
            clicked = q
    typed = st.chat_input("Ask about output, scrap, downtime, capacity or procedures...")
    question = typed or clicked
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            scope = {"line": sel_line, "machine": sel_machine, "start": str(start_date), "end": str(end_date)}
            try:
                search = get_doc_search(api_key)
                tools = ai.make_tools(data, scope, pm.DB_PATH, search)
                llm = get_llm(get_secret("OPENAI_MODEL", "gpt-4o-mini"), api_key)
                with st.spinner("Checking the plant data..."):
                    out = ai.answer_question(question, st.session_state.messages[:-1], llm, tools, scope)
                answer, trace = out["answer"], out["trace"]
            except Exception as exc:
                answer, trace = f"Sorry, the assistant hit an error: `{exc}`", []
            st.markdown(answer)
            render_trace(trace)
        st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})

st.markdown("---")
st.caption("Portfolio project by Surya Vasanth · synthetic data for demonstration")
