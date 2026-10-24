import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import assistant as ai
import fab_capacity as fc
import fab_config as C
import fab_data as fd
import forecast_model as fm

# ======================================================================
# Setup
# ======================================================================
st.set_page_config(page_title="TD Fab Capacity Planner", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .kpi {background: var(--secondary-background-color); padding: 12px 10px; border-radius: 10px; text-align: center;
        border-top: 4px solid #4a6fa5;}
  .kpi-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .5px; font-weight: 600; opacity: .7;}
  .kpi-value {font-size: 1.55rem; font-weight: 700; margin: 3px 0;}
  .kpi-sub {font-size: .74rem; opacity: .7;}
</style>
""", unsafe_allow_html=True)
load_dotenv(fd.ROOT / ".env")

GREEN, AMBER, ORANGE, RED, BLUE = "#2e8b57", "#d4a017", "#e67e22", "#c0392b", "#4a6fa5"
SITE_LABEL = {"FAB_A": "Fab A", "FAB_B": "Fab B", "Both": "Both fabs"}
PROG = {k: v["name"] for k, v in C.PROGRAMS.items()}
BASIS_LABEL = {"requested": "Requested starts", "corrected": "Bias-corrected forecast"}


@st.cache_data(show_spinner=False)
def get_data():
    return fd.load()


def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


@st.cache_resource(show_spinner="Connecting to OpenAI...")
def get_llm(key, preferred):
    return ai.make_llm(key, ai.pick_model(key, preferred))


@st.cache_resource(show_spinner="Indexing planning documents (first question only)...")
def get_doc_search(key):
    from langchain_openai import OpenAIEmbeddings
    search, _ = ai.build_doc_search(fd.DOCS, OpenAIEmbeddings(model="text-embedding-3-small", api_key=key))
    return search


try:
    data = get_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

API_KEY = (get_secret("OPENAI_API_KEY") or "").strip().strip('"') or None
MODEL = (get_secret("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()

# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.title("Planning controls")
site = st.sidebar.radio("Site", ["Both", "FAB_A", "FAB_B"], format_func=lambda s: SITE_LABEL[s], horizontal=True)
basis = st.sidebar.radio("2027 demand", ["requested", "corrected"], format_func=BASIS_LABEL.get,
                         help="Bias-corrected = each program's requests scaled by how much of its past "
                              "requests actually ran (see the Forecast tab).")
ceiling = st.sidebar.slider("Loading ceiling %", 75, 95, int(C.LOADING_CEILING * 100), 5) / 100
st.sidebar.markdown("---")
with st.sidebar.expander("AI status"):
    if not API_KEY:
        st.error("No OPENAI_API_KEY found. In Streamlit Cloud: app menu → Settings → Secrets, add "
                 'OPENAI_API_KEY = "sk-..." and save.')
    else:
        st.caption(f"Key ending …{API_KEY[-4:]} · model {MODEL}")
        if st.button("Test AI connection", width="stretch"):
            with st.spinner("Testing..."):
                ok, msg = ai.check_connection(API_KEY, MODEL)
            (st.success if ok else st.error)(msg)
st.sidebar.caption("Simulated two-fab development network. All numbers are illustrative; "
                   "assumptions are under *How it works*.")


def sites():
    return list(C.SITES) if site == "Both" else [site]


def card(col, label, value, sub="", color=BLUE):
    col.markdown(f"""<div class="kpi" style="border-top-color:{color}"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def row_label(df):
    return df["site"].map(SITE_LABEL) + " · " + df["tool_group"]


def heatmap(table, rows=None, delta=None):
    """Tool group x quarter loading. rows: optional set of (site, tool_group) to show."""
    t = table[table["site"].isin(sites())].copy()
    if rows is not None:
        t = t[[k in rows for k in zip(t["site"], t["tool_group"])]]
    t["row"] = row_label(t)
    order = [f"{SITE_LABEL[s]} · {tg}" for s in sites() for tg in C.TOOL_GROUP_ORDER]
    piv = t.pivot(index="row", columns="quarter", values="loading_pct").reindex([o for o in order if o in set(t["row"])])
    z = piv.fillna(130).to_numpy()
    text = piv.map(lambda v: "no tool" if pd.isna(v) else f"{v:.0f}%")
    if delta is not None:
        d = delta[delta["site"].isin(sites())].copy()
        d["row"] = row_label(d)
        dp = d.pivot(index="row", columns="quarter", values="loading_change_pts").reindex(piv.index)
        text = text.where(dp.abs() < 0.5, text + dp.map(lambda v: f" ({v:+.0f})" if pd.notna(v) else ""))
    lo, hi = 40, 120
    s = [(75 - lo) / (hi - lo), (ceiling * 100 - lo) / (hi - lo), (100 - lo) / (hi - lo)]
    scale = [[0, "#cfe8d5"], [s[0], "#cfe8d5"], [s[0], "#f6e7b0"], [s[1], "#f6e7b0"],
             [s[1], "#f5b98a"], [s[2], "#f5b98a"], [s[2], "#ec7f74"], [1, "#ec7f74"]]
    fig = go.Figure(go.Heatmap(z=z, x=piv.columns, y=piv.index, text=text.to_numpy(), texttemplate="%{text}",
                               textfont={"color": "#1d1d1d", "size": 13}, colorscale=scale, zmin=lo, zmax=hi,
                               showscale=False, xgap=2, ygap=2, hovertemplate="%{y} %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=34 * len(piv) + 50, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    return fig


# ======================================================================
# Models (shared by all tabs)
# ======================================================================
state_req, state_cor = fc.base_state(data, "requested"), fc.base_state(data, "corrected")
tab_req, tab_cor = fc.loading_table(state_req, ceiling), fc.loading_table(state_cor, ceiling)
table, state = (tab_cor, state_cor) if basis == "corrected" else (tab_req, state_req)
in_scope = lambda t: t[t["site"].isin(sites())]
cons = fc.constraints(in_scope(table), ceiling)
n_req = len(fc.constraints(in_scope(tab_req), ceiling))
n_cor = len(fc.constraints(in_scope(tab_cor), ceiling))
worst = in_scope(table).sort_values("loading_pct", ascending=False).iloc[0]
worst_driver = fc.drivers(state, worst["site"], worst["tool_group"], worst["quarter"])[0]
fsum = fm.summary(data["wafer_start_history"], data["plan_2027"])

# Tool groups worth showing: within 5 points of the ceiling in either demand view.
both = pd.concat([tab_req, tab_cor])
peak = both.groupby(["site", "tool_group"])["loading_pct"].max()
HOT_ROWS = set(peak[peak >= ceiling * 100 - 5].index)

# ======================================================================
# Header: three numbers
# ======================================================================
st.title("TD Fab Capacity & Wafer Loading Planner")
st.caption("Can two simulated development fabs run the 2027 wafer-start plan, and what changes if the plan changes?")
k = st.columns(3)
card(k[0], f"At-risk cells · {BASIS_LABEL[basis].lower()}", f"{len(cons)}",
     f"tool group × quarter ≥ {ceiling * 100:.0f}% · requested {n_req}, corrected {n_cor}",
     RED if len(cons) else GREEN)
card(k[1], "Tightest constraint", f"{worst['loading_pct']:.0f}%",
     f"{SITE_LABEL[worst['site']]} {worst['tool_group']} {worst['quarter']} · {worst_driver['program']} "
     f"{worst_driver['share_pct']:.0f}% of load", RED if worst["loading_pct"] > 100 else ORANGE)
card(k[2], "Demand forecast error", f"{fsum['overall_backtest_mape_raw_pct']}% → {fsum['overall_backtest_mape_corrected_pct']}%",
     "raw requests → bias-corrected (backtest)", GREEN)
st.write("")

t_cap, t_scen, t_fc, t_perf, t_how = st.tabs(
    ["2027 capacity", "Scenario lab (AI)", "Forecast", "Fab today", "How it works"])

# ---------------------------------------------------------------- 2027 capacity
with t_cap:
    c1, c2 = st.columns([3, 2])
    with c1:
        show_all = st.toggle("Show all tool groups", value=False)
        st.plotly_chart(heatmap(table, None if show_all else HOT_ROWS), width="stretch", key="heat_main")
        st.caption(f"Loading by quarter, including planned tool arrivals. Green < 75% · yellow < {ceiling * 100:.0f}% "
                   f"ceiling · orange ≤ 100% · red > 100%."
                   + ("" if show_all else f" Showing the {len(HOT_ROWS)} tool groups within 5 points of the ceiling."))
    with c2:
        st.markdown("**Constraints**")
        if cons.empty:
            st.success("No tool group is above the ceiling in 2027.")
        else:
            show = cons.copy()
            show["Tools"] = show["tools"].astype(str) + " → " + show["tools_needed_at_ceiling"].astype(str)
            show["Main driver"] = show.apply(lambda r: fc.drivers(state, r["site"], r["tool_group"], r["quarter"])[0]["program"], axis=1)
            show["site"] = show["site"].map(SITE_LABEL)
            st.dataframe(show[["site", "tool_group", "quarter", "loading_pct", "Tools", "Main driver"]]
                         .rename(columns={"site": "Site", "tool_group": "Group", "quarter": "Quarter", "loading_pct": "Load %"}),
                         hide_index=True, width="stretch", height=min(420, 35 * len(show) + 40))
            st.caption("Tools = installed → needed to stay under the ceiling.")
        arrivals = " · ".join(f"{SITE_LABEL[s]} {tg} +{n} in {q}" for s, tg, q, n in C.TOOL_ARRIVALS)
        st.caption(f"Planned arrivals: {arrivals}.")

# ---------------------------------------------------------------- scenario lab
EXAMPLES = {
    "Process change (Node B EUV conversion)":
        "PCN-2026-031 from Process Integration: starting 2027Q2, DRAM Node B moves two critical layers "
        "(bit-line contact and storage-node landing pad) from DUV to EUV. Each converted layer also needs one "
        "additional etch step.",
    "Program request (HBM pull-in)":
        "HBM program update: please pull the HBM test vehicle ramp in by one quarter. We also need an extra "
        "40 wafers per week in 2027Q3 for a TSV reliability split.",
    "Equipment update (EUV slip, etch upgrade)":
        "Equipment update: the second EUV scanner for Fab A slips one quarter. Separately, Fab B etch will lose "
        "about 5 points of availability in 2027Q2 while chambers get a source upgrade.",
    "Load balancing proposal (NAND to Fab A)":
        "Proposal from IE Strategy: move 25% of 3D NAND Gen C wafer starts from Fab B to Fab A for all of 2027 "
        "to relieve Fab B etch and deposition.",
}
EXAMPLE_CHANGES = {
    "Process change (Node B EUV conversion)": [
        dict(type="route_change", program="DRAM_B", tool_group="EUV", passes_delta=2, from_quarter="2027Q2"),
        dict(type="route_change", program="DRAM_B", tool_group="DUV", passes_delta=-2, from_quarter="2027Q2"),
        dict(type="route_change", program="DRAM_B", tool_group="ETCH", passes_delta=2, from_quarter="2027Q2")],
    "Program request (HBM pull-in)": [
        dict(type="pull_in", program="HBM_TV", quarters=1),
        dict(type="starts_change", program="HBM_TV", site="FAB_A", wspw_delta=40, from_quarter="2027Q3", to_quarter="2027Q3")],
    "Equipment update (EUV slip, etch upgrade)": [
        dict(type="tool_delay", site="FAB_A", tool_group="EUV", quarters=1),
        dict(type="availability_change", site="FAB_B", tool_group="ETCH", availability_pts=-5,
             from_quarter="2027Q2", to_quarter="2027Q2")],
    "Load balancing proposal (NAND to Fab A)": [
        dict(type="move_load", program="NAND_C", site="FAB_B", to_site="FAB_A", pct=25)],
}
CHANGE_COLS = ["type", "program", "site", "to_site", "tool_group", "passes_delta", "wspw_delta", "pct",
               "quarters", "tools", "availability_pts", "from_quarter", "to_quarter", "source_text"]
NUM_COLS = ["passes_delta", "wspw_delta", "pct", "quarters", "tools", "availability_pts"]


def changes_df(records):
    df = pd.DataFrame([{c: r.get(c) for c in CHANGE_COLS} for r in records], columns=CHANGE_COLS)
    for c in CHANGE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64") if c in NUM_COLS else df[c].astype("object")
    return df


def df_to_changes(df):
    out, errors = [], []
    for i, row in df.iterrows():
        rec = {c: row[c] for c in CHANGE_COLS if c in row and pd.notna(row[c]) and row[c] != ""}
        if not rec.get("type"):
            continue
        for c in ("passes_delta", "quarters", "tools"):
            if c in rec:
                rec[c] = int(rec[c])
        try:
            out.append(fc.PlanChange(**rec))
        except Exception as exc:
            errors.append(f"Row {i + 1}: {str(exc).splitlines()[0]}")
    return out, errors


RESPONSES = {
    "Move 25% of NAND Gen C to Fab A": [
        dict(type="move_load", program="NAND_C", site="FAB_B", to_site="FAB_A", pct=25)],
    "Add 3 etch chambers at Fab B from 2027Q2": [
        dict(type="tool_add", site="FAB_B", tool_group="ETCH", tools=3, from_quarter="2027Q2")],
    "Push the NAND Gen C ramp out one quarter": [
        dict(type="pull_in", program="NAND_C", quarters=-1)],
    "Recover 5 points of Fab B etch availability": [
        dict(type="availability_change", site="FAB_B", tool_group="ETCH", availability_pts=5)],
}
NOTE_OPTION = "Changes from the note (One scenario)"

ss = st.session_state
ss.setdefault("editor_v", 0)
ss.setdefault("changes", changes_df([]))
ss.setdefault("note_changes", [])
ss.setdefault("extraction", None)
ss.setdefault("scenario", None)
ss.setdefault("readout", None)
ss.setdefault("compare_brief", None)


def ai_error(exc):
    return ai.explain_error(exc, MODEL)


with t_scen:
    mode = st.radio("Mode", ["One scenario from a note", "Compare two responses"], horizontal=True,
                    label_visibility="collapsed")

    if mode == "One scenario from a note":
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**1 · Paste a planning note**")
            ex = st.selectbox("Example notes", list(EXAMPLES), label_visibility="collapsed")
            note = st.text_area("Planning note", EXAMPLES[ex], height=100, key=f"note_{ex}",
                                label_visibility="collapsed")
            b1, b2 = st.columns(2)
            if b1.button("Read note with AI", type="primary", width="stretch", disabled=not API_KEY):
                try:
                    with st.spinner("Reading the note..."):
                        cs = ai.extract_changes(get_llm(API_KEY, MODEL), note)
                    ss.changes = changes_df([c.model_dump() for c in cs.changes])
                    ss.extraction = {"summary": cs.summary, "open_questions": cs.open_questions, "error": None}
                except Exception as exc:
                    ss.extraction = {"summary": None, "open_questions": [], "error": ai_error(exc)}
                ss.editor_v += 1
                ss.scenario = ss.readout = None
            if b2.button("Use example changes (no AI)", width="stretch"):
                ss.changes = changes_df(EXAMPLE_CHANGES[ex])
                ss.extraction = None
                ss.editor_v += 1
                ss.scenario = ss.readout = None
        with c2:
            st.markdown("**2 · Check what it found**")
            found = st.container()
            with st.expander("Edit changes"):
                edited = st.data_editor(
                    ss.changes, num_rows="dynamic", width="stretch", key=f"editor_{ss.editor_v}", hide_index=True,
                    column_order=[c for c in CHANGE_COLS if c != "source_text"],
                    column_config={
                        "type": st.column_config.SelectboxColumn("type", options=list(fc.ChangeType.__args__), required=True),
                        "program": st.column_config.SelectboxColumn("program", options=C.PROGRAM_ORDER),
                        "site": st.column_config.SelectboxColumn("site", options=list(C.SITES)),
                        "to_site": st.column_config.SelectboxColumn("to_site", options=list(C.SITES)),
                        "tool_group": st.column_config.SelectboxColumn("tool_group", options=C.TOOL_GROUP_ORDER),
                        "from_quarter": st.column_config.SelectboxColumn("from_quarter", options=C.PLAN_QUARTERS),
                        "to_quarter": st.column_config.SelectboxColumn("to_quarter", options=C.PLAN_QUARTERS),
                    })
            changes, errors = df_to_changes(edited)
            ss.note_changes = [c.model_dump(exclude_none=True) for c in changes]
            with found:
                if ss.extraction and ss.extraction["error"]:
                    st.error(ss.extraction["error"])
                elif ss.extraction:
                    st.caption(f"AI summary: {ss.extraction['summary']}")
                if changes:
                    _, described, skipped = fc.apply_changes(fc.base_state(data, basis), changes)
                    st.markdown("\n".join(f"- {d}" for d in described))
                    for w in skipped + errors:
                        st.warning(w)
                else:
                    st.caption("Nothing yet. Read the note with AI, or load the example changes.")
                if ss.extraction and ss.extraction["open_questions"]:
                    st.warning("Confirm first: " + " ".join(ss.extraction["open_questions"]))
            if st.button("3 · Run scenario", type="primary", disabled=not changes):
                ss.scenario = fc.run_scenario(data, changes, basis, ceiling)
                ss.readout = None

        res = ss.scenario
        if res:
            st.markdown("---")
            k = st.columns(3)
            card(k[0], "At-risk cells", f"{res['constraints_before']} → {res['constraints_after']}", "both fabs, 2027",
                 GREEN if res["constraints_after"] <= res["constraints_before"] else RED)
            card(k[1], "New constraints", f"{len(res['new_constraints'])}",
                 ", ".join(res["new_constraints"][:3]) or "none", RED if res["new_constraints"] else GREEN)
            card(k[2], "Cleared", f"{len(res['cleared_constraints'])}",
                 ", ".join(res["cleared_constraints"][:3]) or "none", GREEN)
            c1, c2 = st.columns([3, 2])
            with c1:
                delta = res["_tables"]["base"].merge(res["_tables"]["scenario"], on=["site", "tool_group", "quarter"],
                                                     suffixes=("_b", ""))
                delta["loading_change_pts"] = delta["loading_pct"] - delta["loading_pct_b"]
                moved = delta[delta["loading_change_pts"].abs() >= 0.5]
                rows = set(zip(moved["site"], moved["tool_group"]))
                if rows:
                    st.markdown("**Tool groups that changed** (change vs baseline in brackets)")
                    st.plotly_chart(heatmap(res["_tables"]["scenario"], rows, delta), width="stretch", key="heat_scen")
                else:
                    st.info("No tool-group loading changed by half a point or more.")
            with c2:
                st.markdown("**4 · Decision brief**")
                if st.button("Draft decision brief with AI", disabled=not API_KEY, width="stretch"):
                    try:
                        with st.spinner("Drafting..."):
                            ss.readout = ai.draft_readout(get_llm(API_KEY, MODEL), res, ss.note_changes)
                    except Exception as exc:
                        ss.readout = "⚠️ " + ai_error(exc)
                if ss.readout:
                    st.markdown(ss.readout)
                    with st.expander("Numbers the brief used"):
                        st.json({k: v for k, v in res.items() if not k.startswith("_")}, expanded=False)

    else:
        options = dict(RESPONSES)
        if ss.note_changes:
            options[NOTE_OPTION] = ss.note_changes
        names = list(options)
        base_cons_all = fc.constraints(table, ceiling)
        st.caption(f"Baseline ({BASIS_LABEL[basis].lower()}, both fabs): {len(base_cons_all)} at-risk cells. "
                   f"Pick two responses to compare against it.")
        c1, c2 = st.columns(2)
        a = c1.selectbox("Option A", names, index=0)
        b = c2.selectbox("Option B", names, index=1)
        cmp_res = fc.compare_options(data, {"A": options[a], "B": options[b]}, basis, ceiling)
        oa, ob = cmp_res["options"]
        base = cmp_res["baseline"]

        def cells(lst):
            return ", ".join(lst) if lst else "none"

        rows = [
            ("At-risk cells", base["at_risk_cells"], oa["at_risk_cells"], ob["at_risk_cells"]),
            ("Over 100%", base["over_100_pct"], oa["over_100_pct"], ob["over_100_pct"]),
            ("Cleared", "", len(oa["cleared_constraints"]), len(ob["cleared_constraints"])),
            ("New constraints", "", cells(oa["new_constraints"]), cells(ob["new_constraints"])),
            ("Tightest cell", base["tightest_cell"], oa["tightest_cell"], ob["tightest_cell"]),
            ("Tools added", 0, oa["tools_added"], ob["tools_added"]),
            ("Starts moved or deferred (avg WSPW)", 0, oa["avg_wspw_moved_or_deferred"], ob["avg_wspw_moved_or_deferred"]),
        ]
        cmp_df = pd.DataFrame(rows, columns=["", "Baseline", f"A · {a}", f"B · {b}"]).astype(str)
        st.dataframe(cmp_df, hide_index=True, width="stretch")
        for o in (oa, ob):
            for w in o["warnings"]:
                st.warning(f"Option {o['option']}: {w}")
        key = (a, b, basis, ceiling)
        if st.button("Draft decision brief with AI", key="cmp_brief", disabled=not API_KEY):
            try:
                with st.spinner("Drafting..."):
                    text = ai.draft_comparison(get_llm(API_KEY, MODEL), cmp_res, {"A": options[a], "B": options[b]})
            except Exception as exc:
                text = "⚠️ " + ai_error(exc)
            ss.compare_brief = (key, text)
        if ss.compare_brief and ss.compare_brief[0] == key:
            st.markdown(ss.compare_brief[1])
            with st.expander("Numbers the brief used"):
                st.json(cmp_res, expanded=False)

    if not API_KEY:
        st.caption("Add OPENAI_API_KEY in the app's Secrets to turn on the AI steps.")

# ---------------------------------------------------------------- forecast
with t_fc:
    hist = data["wafer_start_history"]
    model = fm.fit(hist)
    st.markdown(f"**Requests aren't demand.** Correcting each program's request by how much of it historically ran "
                f"cuts forecast error from {fsum['overall_backtest_mape_raw_pct']}% to "
                f"{fsum['overall_backtest_mape_corrected_pct']}%, and 2027 at-risk cells from {n_req} to {n_cor}.")
    c1, c2 = st.columns([3, 2])
    with c1:
        prog = st.selectbox("Program", C.PROGRAM_ORDER, index=1, format_func=PROG.get)
        h = hist[hist["program"] == prog].groupby("quarter", as_index=False)[["requested_wspw", "actual_wspw"]].sum()
        rate = model.set_index("program").loc[prog, "realization_rate"]
        fig = go.Figure()
        fig.add_scatter(x=h["quarter"], y=h["requested_wspw"], name="Requested", mode="lines+markers",
                        line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=h["quarter"], y=h["actual_wspw"], name="Actual", mode="lines+markers", line=dict(color=BLUE))
        fig.add_scatter(x=h["quarter"], y=(h["requested_wspw"] * rate).round(), name=f"Corrected (× {rate:.2f})",
                        mode="lines", line=dict(color=GREEN, dash="dot"))
        fig.add_vrect(x0=fm.TRAIN_END, x1=h["quarter"].max(), fillcolor="grey", opacity=0.08,
                      annotation_text="backtest", annotation_position="top left")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Wafer starts per week",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch", key="fc_chart")
    with c2:
        st.dataframe(model[["program_name", "realization_rate", "backtest_mape_raw_pct", "backtest_mape_corrected_pct"]]
                     .rename(columns={"program_name": "Program", "realization_rate": "Actual ÷ requested",
                                      "backtest_mape_raw_pct": "Error raw %", "backtest_mape_corrected_pct": "Error corrected %"}),
                     hide_index=True, width="stretch")
        st.caption(f"One regression per program: actual = b × requested. Fitted through {fm.TRAIN_END}, "
                   f"error measured on {', '.join(fsum['backtest_quarters'])}.")

# ---------------------------------------------------------------- fab today
with t_perf:
    perf = fd.performance_summary(data, site)
    st.caption(f"{SITE_LABEL[site]} · Q3 2026 · 13 weeks from the SimPy simulation")
    k = st.columns(5)
    card(k[0], "Wafer starts / wk", f"{perf['wafer_starts_per_week']:,}", f"plan {perf['planned_starts_per_week']:,}")
    card(k[1], "Moves / day", f"{perf['moves_per_day']:,}", f"WIP {perf['wip_wafers']:,} wafers")
    card(k[2], "Cycle time", f"{perf['cycle_time_days_full_flow']} d",
         f"hot {perf['cycle_time_days_hot']} d · normal {perf['cycle_time_days_normal']} d")
    card(k[3], "Days / mask layer", f"{perf['days_per_mask_layer']}", f"X-factor {perf['x_factor']}")
    card(k[4], "Line yield", f"{perf['line_yield_pct']}%", "wafers out ÷ wafers in")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Where cycle time goes**")
        cp = fd.cycle_time_by_program(data, site)
        cp = cp[cp["program"] != "SHORT"]
        cp["lot"] = cp["site"].map(SITE_LABEL) + " · " + cp["program"].map(PROG)
        m = cp.melt(id_vars="lot", value_vars=["process", "queue", "transport", "hold"], var_name="Part", value_name="Days")
        m["Part"] = m["Part"].map({"process": "Processing", "queue": "Queue at tool", "transport": "Transport",
                                   "hold": "Engineering hold"})
        fig = px.bar(m, y="lot", x="Days", color="Part", orientation="h",
                     color_discrete_sequence=[GREEN, ORANGE, "#95a5a6", RED], labels={"lot": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, width="stretch", key="ct_chart")
    with c2:
        st.markdown("**Busiest tool groups: time by SEMI E10 state**")
        tg = fd.tool_group_states(data, site).sort_values("loading_pct", ascending=False).head(6)
        tg["group"] = row_label(tg)
        m = tg.melt(id_vars="group", value_vars=["productive_pct", "standby_pct", "engineering_pct",
                                                 "sched_down_pct", "unsched_down_pct"], var_name="State", value_name="%")
        m["State"] = m["State"].map({"productive_pct": "Productive", "standby_pct": "Standby", "engineering_pct": "Engineering",
                                     "sched_down_pct": "Scheduled down", "unsched_down_pct": "Unscheduled down"})
        fig = px.bar(m, y="group", x="%", color="State", orientation="h",
                     color_discrete_sequence=[GREEN, "#cfd8dc", BLUE, AMBER, RED], labels={"group": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"),
                          legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, width="stretch", key="e10_chart")

# ---------------------------------------------------------------- how it works
with t_how:
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fa", fontname="Helvetica", fontsize=11];
  note [label="Planning note", fillcolor="#fdf2e9"];
  ext [label="AI reads it into\\nstructured changes", fillcolor="#e8f6ef"];
  human [label="Planner checks", fillcolor="#fff7d6"];
  model [label="Capacity model"];
  fc [label="Forecast model"];
  read [label="AI drafts readout", fillcolor="#e8f6ef"];
  note -> ext -> human -> model -> read; fc -> model;
}
""")
    st.markdown("**The AI never does the math.** It reads notes, picks which model or query to run, and explains "
                "the result. A planner confirms changes before the model runs.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Formulas**")
        st.markdown("- Loading = required hours ÷ available hours, per site, tool group and quarter\n"
                    "- Required = Σ wafer starts/wk × passes × minutes per pass ÷ 60\n"
                    "- Available = tools × 168 h × availability (minus PM, breakdowns, engineering time)\n"
                    "- Forecast: actual = b × requested, per program")
        mv = fc.model_vs_simulation(data)
        ll = fd.littles_law_check(data).set_index("site")
        st.markdown("**Validation**")
        st.markdown(f"- Capacity model vs simulation: {mv['difference_pts'].abs().mean():.1f} points apart on average "
                    f"across {len(mv)} tool groups\n"
                    f"- Little's Law (WIP = throughput × cycle time): Fab A {ll.loc['FAB_A', 'predicted_wip_lots']} "
                    f"predicted vs {ll.loc['FAB_A', 'simulated_wip_lots']} simulated lots; Fab B "
                    f"{ll.loc['FAB_B', 'predicted_wip_lots']} vs {ll.loc['FAB_B', 'simulated_wip_lots']}")
        st.markdown("**Limits**")
        st.markdown("- Illustrative data. Tool groups are pooled; routes are summarized as passes per tool group.\n"
                    "- Loading is a weekly average; the simulation shows queues, the model does not.")
    with c2:
        st.markdown("**Why plan below the ceiling: EUV queue vs loading**")
        oc = fd.operating_curve(data, "FAB_A", "EUV")
        Y_MAX = 60
        fig = go.Figure()
        fig.add_vrect(x0=ceiling * 100, x1=100, fillcolor=ORANGE, opacity=0.10, line_width=0)
        fig.add_vline(x=ceiling * 100, line_dash="dot", line_color="grey")
        fig.add_annotation(x=ceiling * 100, y=Y_MAX, text=f"ceiling {ceiling * 100:.0f}%", showarrow=False,
                           xanchor="right", yanchor="top", xshift=-4, font=dict(size=11))
        fig.add_scatter(x=oc["u_pct"], y=oc["queue_h"], mode="lines", name="Expected queue per pass",
                        line=dict(color=BLUE, width=3))
        fig.add_scatter(x=[oc["sim_point"][0]], y=[oc["sim_point"][1]], mode="markers", name="Q3 2026 simulated",
                        marker=dict(size=12, color=GREEN))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(title="Loading %", range=[40, 100]),
                          yaxis=dict(title="Queue hours per pass", range=[0, Y_MAX]), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch", key="oc_chart")
        st.caption("Queue ∝ u ÷ (1 − u), calibrated to the simulated EUV queue. It more than doubles between "
                   "80% and 90% loading and has no limit at 100%.")
    with st.expander("Assumptions and data"):
        st.markdown("**Tool groups**")
        st.dataframe(data["tool_groups"], hide_index=True, width="stretch")
        a1, a2 = st.columns(2)
        a1.markdown("**Passes per wafer by program**")
        a1.dataframe(data["routes"].pivot(index="tool_group", columns="program", values="passes")
                     .reindex(C.TOOL_GROUP_ORDER), width="stretch")
        a2.markdown("**Installed tools**")
        a2.dataframe(data["installed_tools"].pivot(index="tool_group", columns="site", values="tools")
                     .reindex(C.TOOL_GROUP_ORDER), width="stretch")
        st.markdown("**2027 wafer-start plan (requested, wafers per week)**")
        st.dataframe(data["plan_2027"].pivot_table(index=["site", "program"], columns="quarter",
                                                   values="requested_wspw"), width="stretch")
        st.markdown("**Capacity model vs simulation, Q3 2026 loading by tool group**")
        st.dataframe(mv, hide_index=True, width="stretch")

# ======================================================================
# Copilot chat
# ======================================================================
st.markdown("---")
h1, h2 = st.columns([6, 1])
h1.subheader("Planning copilot")
h1.caption("Ask in plain English. It runs the models, SQL and document search, and shows every step.")
if h2.button("Clear chat", width="stretch"):
    ss.messages = []
    st.rerun()
ss.setdefault("messages", [])

SUGGESTIONS = {
    "What's constrained in 2027?":
        "Which tool groups are constrained in 2027, and how does the bias-corrected forecast change that?",
    "Fix Fab B etch?":
        "Fab B etch goes over 100% in 2027. Would moving 25% of NAND Gen C starts to Fab A fix it without new constraints?",
    "What drives cycle time?":
        "What is driving cycle time at Fab B, and what did past planning reviews decide about it?",
}


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
                st.caption("Inputs: " + json.dumps(args)[:400])
            if step["tool"] == "search_documents" and isinstance(result, list):
                for hit in result:
                    st.markdown(f"📄 **{hit['source']}** ({hit['folder']})")
                    st.caption(hit["text"][:220].replace("\n", " ") + "...")
            elif isinstance(result, (dict, list)):
                st.json(result, expanded=False)
            else:
                st.caption(str(result)[:500])


for msg in ss.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_trace(msg.get("trace"))

if not API_KEY:
    st.info("The copilot needs an OpenAI API key: add OPENAI_API_KEY to the app's Streamlit secrets.")
else:
    cols = st.columns(len(SUGGESTIONS) + 2)
    clicked = None
    for col, (label, q) in zip(cols, SUGGESTIONS.items()):
        if col.button(label, width="stretch", help=q):
            clicked = q
    question = st.chat_input("Ask about loading, constraints, forecasts, scenarios or fab performance...") or clicked
    if question:
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            ctx = {"site": site, "basis": basis}
            try:
                search = get_doc_search(API_KEY)
            except Exception:
                search = None   # copilot still works with the models and SQL; only document search is off
            try:
                tools = ai.make_tools(data, ctx, fd.DB_PATH, search)
                with st.spinner("Running the models..."):
                    out = ai.answer_question(question, ss.messages[:-1], get_llm(API_KEY, MODEL), tools, ctx)
                answer, trace = out["answer"], out["trace"]
            except Exception as exc:
                answer, trace = "⚠️ " + ai_error(exc), []
            st.markdown(answer)
            render_trace(trace)
        ss.messages.append({"role": "assistant", "content": answer, "trace": trace})

st.caption("Portfolio project by Surya Vasanth · synthetic, illustrative data · not affiliated with any company")
