import json
import os

import numpy as np
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
  .callout {padding: 10px 14px; border-radius: 8px; border-left: 4px solid #4a6fa5;
            background: var(--secondary-background-color); margin: 6px 0 12px 0; font-size: .93rem;}
</style>
""", unsafe_allow_html=True)
load_dotenv(fd.ROOT / ".env")

GREEN, AMBER, ORANGE, RED, BLUE = "#2e8b57", "#d4a017", "#e67e22", "#c0392b", "#4a6fa5"
SITE_LABEL = {"FAB_A": "Fab A", "FAB_B": "Fab B", "Both": "Both fabs"}
PROG = {k: v["name"] for k, v in C.PROGRAMS.items()}


@st.cache_data(show_spinner=False)
def get_data():
    return fd.load()


def get_secret(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


@st.cache_resource(show_spinner=False)
def get_llm(model, key):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0, api_key=key)


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

API_KEY = get_secret("OPENAI_API_KEY")
MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.title("Planning controls")
site = st.sidebar.radio("Site", ["Both", "FAB_A", "FAB_B"], format_func=lambda s: SITE_LABEL[s], horizontal=True)
basis = st.sidebar.radio("2027 demand basis", ["requested", "corrected"],
                         format_func=lambda b: {"requested": "Requested starts", "corrected": "Bias-corrected forecast"}[b],
                         help="Bias-corrected = each program's requests scaled by how much of past requests "
                              "actually ran (see Forecast accuracy).")
ceiling = st.sidebar.slider("Loading ceiling %", 75, 95, int(C.LOADING_CEILING * 100), 5) / 100
st.sidebar.markdown("---")
st.sidebar.caption("Simulated two-site development-fab network. All numbers are illustrative; "
                   "assumptions are listed under *How it works*.")


def sites():
    return list(C.SITES) if site == "Both" else [site]


def card(col, label, value, sub="", color=BLUE):
    col.markdown(f"""<div class="kpi" style="border-top-color:{color}"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def row_label(df):
    return df["site"].map(SITE_LABEL) + " · " + df["tool_group"]


def heatmap(table, height=None, delta=None):
    t = table[table["site"].isin(sites())].copy()
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
    c = ceiling * 100
    s = [(0.75 * 100 - lo) / (hi - lo), (c - lo) / (hi - lo), (100 - lo) / (hi - lo)]
    scale = [[0, "#cfe8d5"], [s[0], "#cfe8d5"], [s[0], "#f6e7b0"], [s[1], "#f6e7b0"],
             [s[1], "#f5b98a"], [s[2], "#f5b98a"], [s[2], "#ec7f74"], [1, "#ec7f74"]]
    fig = go.Figure(go.Heatmap(z=z, x=piv.columns, y=piv.index, text=text.to_numpy(), texttemplate="%{text}",
                               textfont={"color": "#1d1d1d", "size": 12}, colorscale=scale, zmin=lo, zmax=hi,
                               showscale=False, xgap=2, ygap=2,
                               hovertemplate="%{y} %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=height or 28 * len(piv) + 60, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    return fig


# ======================================================================
# Header and KPI strips
# ======================================================================
st.title("TD Fab Capacity & Wafer Loading Planner")
st.caption("Simulated network of two technology-development fabs · illustrative data · capacity model, "
           "forecast correction and an AI planning copilot")

perf = fd.performance_summary(data, site)
st.markdown(f"##### Fab performance · Q3 2026 · {SITE_LABEL[site]} (13 simulated weeks)")
k = st.columns(7)
card(k[0], "Wafer starts / wk", f"{perf['wafer_starts_per_week']:,}", f"plan {perf['planned_starts_per_week']:,}")
card(k[1], "Wafer outs / wk", f"{perf['wafer_outs_per_week']:,}", "completed lots")
card(k[2], "Moves / day", f"{perf['moves_per_day']:,}", "wafer steps completed")
card(k[3], "WIP", f"{perf['wip_wafers']:,}", "wafers in the line")
card(k[4], "Cycle time", f"{perf['cycle_time_days_full_flow']} d",
     f"hot {perf['cycle_time_days_hot']} d · normal {perf['cycle_time_days_normal']} d")
card(k[5], "Days / mask layer", f"{perf['days_per_mask_layer']}", f"X-factor {perf['x_factor']}")
card(k[6], "Line yield", f"{perf['line_yield_pct']}%", "wafers out / wafers in")

state_req, state_cor = fc.base_state(data, "requested"), fc.base_state(data, "corrected")
tab_req, tab_cor = fc.loading_table(state_req, ceiling), fc.loading_table(state_cor, ceiling)
table = tab_cor if basis == "corrected" else tab_req
state = state_cor if basis == "corrected" else state_req
cons = fc.constraints(table[table["site"].isin(sites())], ceiling)
n_req = len(fc.constraints(tab_req[tab_req["site"].isin(sites())], ceiling))
n_cor = len(fc.constraints(tab_cor[tab_cor["site"].isin(sites())], ceiling))
over = cons[cons["status"].str.startswith("Cannot")]
worst = table[table["site"].isin(sites())].sort_values("loading_pct", ascending=False).iloc[0]

st.markdown(f"##### 2027 plan · {'bias-corrected forecast' if basis == 'corrected' else 'requested starts'}")
k = st.columns(4)
card(k[0], "At-risk cells", f"{len(cons)}", f"tool group × quarter ≥ {ceiling * 100:.0f}%",
     RED if len(cons) else GREEN)
card(k[1], "Over 100%", f"{len(over)}", "plan cannot run as requested", RED if len(over) else GREEN)
card(k[2], "Requested vs corrected", f"{n_req} → {n_cor}", "at-risk cells after forecast correction", BLUE)
card(k[3], "Highest loading", f"{worst['loading_pct']:.0f}%",
     f"{SITE_LABEL[worst['site']]} {worst['tool_group']} {worst['quarter']}", RED if worst["loading_pct"] > 100 else ORANGE)

# ======================================================================
# Tabs
# ======================================================================
t_cap, t_fc, t_scen, t_perf, t_how = st.tabs(
    ["Capacity & loading", "Forecast accuracy", "Scenario lab (AI)", "Fab performance", "How it works"])

# ---------------------------------------------------------------- capacity
with t_cap:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### Tool-group loading by quarter")
        st.caption(f"Green < 75% · yellow < {ceiling * 100:.0f}% ceiling · orange ≤ 100% · red over 100% "
                   "(includes planned tool arrivals)")
        st.plotly_chart(heatmap(table), width="stretch", key="chart_1")
    with c2:
        st.markdown("##### Constraints and what drives them")
        if cons.empty:
            st.success("No tool group is above the ceiling in 2027.")
        else:
            show = cons.copy()
            show["drivers"] = show.apply(lambda r: ", ".join(
                f"{d['program']} {d['share_pct']:.0f}%" for d in fc.drivers(state, r["site"], r["tool_group"], r["quarter"])[:2]), axis=1)
            show["site"] = show["site"].map(SITE_LABEL)
            st.dataframe(show[["site", "tool_group", "quarter", "loading_pct", "tools", "tools_needed_at_ceiling", "drivers"]]
                         .rename(columns={"site": "Site", "tool_group": "Tool group", "quarter": "Quarter",
                                          "loading_pct": "Loading %", "tools": "Tools",
                                          "tools_needed_at_ceiling": "Needed", "drivers": "Top drivers"}),
                         hide_index=True, width="stretch", height=min(420, 36 * len(show) + 40))
        arr = pd.DataFrame(C.TOOL_ARRIVALS, columns=["Site", "Tool group", "Release quarter", "Tools"])
        arr["Site"] = arr["Site"].map(SITE_LABEL)
        st.markdown("**Planned tool arrivals**")
        st.dataframe(arr, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Why plan below the ceiling: EUV operating curve")
        oc = fd.operating_curve(data, "FAB_A", "EUV")
        q3 = table[(table["site"] == "FAB_A") & (table["tool_group"] == "EUV")].set_index("quarter")["loading_pct"]
        fig = go.Figure()
        fig.add_scatter(x=oc["u_pct"], y=oc["queue_h"], mode="lines", name="Expected queue per pass",
                        line=dict(color=BLUE, width=3))
        fig.add_scatter(x=[oc["sim_point"][0]], y=[oc["sim_point"][1]], mode="markers", name="Q3 2026 simulated",
                        marker=dict(size=12, color=GREEN))
        fig.add_vline(x=ceiling * 100, line_dash="dot", annotation_text=f"ceiling {ceiling * 100:.0f}%")
        peak_q = q3.idxmax()
        if q3[peak_q] < 97:
            y_peak = float(np.interp(q3[peak_q], oc["u_pct"], oc["queue_h"]))
            fig.add_scatter(x=[q3[peak_q]], y=[y_peak], mode="markers", name=f"{peak_q} plan",
                            marker=dict(size=12, color=RED, symbol="diamond"))
        else:
            fig.add_vline(x=97, line_color=RED, annotation_text=f"{peak_q} plan {q3[peak_q]:.0f}% (off the curve)")
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Loading %",
                          yaxis_title="Queue hours per EUV pass", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch", key="chart_2")
        st.caption("Kingman-shaped curve (queue ∝ u / (1 − u)) calibrated to the simulated EUV queue. "
                   "Queue time roughly doubles between 80% and 90% loading.")
    with c2:
        st.markdown("##### Wafer starts by program (2027 plan)")
        pf = fc.plan_frame(state)
        pf = pf[pf["site"].isin(sites())]
        pf["program"] = pf["program"].map(PROG)
        fig = px.bar(pf.groupby(["quarter", "program"], as_index=False)["wspw"].sum(), x="quarter", y="wspw",
                     color="program", labels={"wspw": "Wafer starts per week", "quarter": ""})
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, width="stretch", key="chart_3")

# ---------------------------------------------------------------- forecast
with t_fc:
    hist = data["wafer_start_history"]
    model = fm.fit(hist)
    fsum = fm.summary(hist, data["plan_2027"])
    st.markdown(f"""<div class="callout"><b>Requests are not demand.</b> Across the last three quarters, using raw
        requests as the forecast missed actual starts by <b>{fsum['overall_backtest_mape_raw_pct']}%</b> on average.
        Scaling each program's request by its historical realization rate cuts that to
        <b>{fsum['overall_backtest_mape_corrected_pct']}%</b>, and the number of at-risk 2027 cells falls from
        <b>{n_req}</b> to <b>{n_cor}</b>.</div>""", unsafe_allow_html=True)
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
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Wafer starts per week",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch", key="chart_4")
    with c2:
        st.markdown("##### Model by program")
        st.dataframe(model.drop(columns=["program"]).rename(columns={
            "program_name": "Program", "quarters": "Qtrs", "realization_rate": "Actual / requested",
            "avg_bias_pct": "Over-request %", "backtest_mape_raw_pct": "Error: raw %",
            "backtest_mape_corrected_pct": "Error: corrected %"}), hide_index=True, width="stretch")
        st.caption("Regression of actual on requested starts through the origin, one per program. "
                   f"Fitted on quarters up to {fm.TRAIN_END}; error (MAPE) measured on "
                   f"{', '.join(fsum['backtest_quarters'])}.")
        tot = pd.DataFrame(fsum["plan_2027_totals"])
        fig = go.Figure()
        fig.add_bar(x=tot["quarter"], y=tot["requested_wspw"], name="Requested", marker_color=ORANGE)
        fig.add_bar(x=tot["quarter"], y=tot["corrected_wspw"], name="Corrected", marker_color=GREEN)
        fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0), barmode="group",
                          yaxis_title="Total WSPW", legend=dict(orientation="h", y=1.2))
        st.plotly_chart(fig, width="stretch", key="chart_5")

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


ss = st.session_state
ss.setdefault("editor_v", 0)
ss.setdefault("changes", changes_df([]))
ss.setdefault("extraction", None)
ss.setdefault("scenario", None)
ss.setdefault("readout", None)

with t_scen:
    st.markdown("##### 1 · Read a planning note")
    st.caption("Paste a route-change notice, program request or equipment update. The AI turns it into structured "
               "changes; you check them before anything runs.")
    c1, c2 = st.columns([3, 2])
    with c1:
        ex = st.selectbox("Example notes", list(EXAMPLES))
        note = st.text_area("Planning note", EXAMPLES[ex], height=110, key=f"note_{ex}")
        b1, b2 = st.columns(2)
        if b1.button("Read note with AI", type="primary", width="stretch", disabled=not API_KEY):
            try:
                with st.spinner("Reading the note..."):
                    cs = ai.extract_changes(get_llm(MODEL, API_KEY), note)
                ss.changes = changes_df([c.model_dump() for c in cs.changes])
                ss.extraction = {"summary": cs.summary, "open_questions": cs.open_questions}
            except Exception as exc:
                ss.extraction = {"summary": f"AI extraction failed: {exc}", "open_questions": []}
            ss.editor_v += 1
            ss.scenario = ss.readout = None
        if b2.button("Fill example changes without AI", width="stretch"):
            ss.changes = changes_df(EXAMPLE_CHANGES[ex])
            ss.extraction = {"summary": "Pre-filled example changes (no AI used).", "open_questions": []}
            ss.editor_v += 1
            ss.scenario = ss.readout = None
        if not API_KEY:
            st.caption("Add OPENAI_API_KEY in the app secrets to enable AI reading.")
    with c2:
        if ss.extraction:
            st.markdown(f"**AI summary:** {ss.extraction['summary']}")
            if ss.extraction["open_questions"]:
                st.warning("**Confirm before running:**\n" + "\n".join(f"- {q}" for q in ss.extraction["open_questions"]))
            else:
                st.caption("No open questions flagged.")

    st.markdown("##### 2 · Check the changes")
    opt = lambda vals: list(vals)
    edited = st.data_editor(
        ss.changes, num_rows="dynamic", width="stretch", key=f"editor_{ss.editor_v}", hide_index=True,
        column_config={
            "type": st.column_config.SelectboxColumn("type", options=list(fc.ChangeType.__args__), required=True),
            "program": st.column_config.SelectboxColumn("program", options=opt(C.PROGRAM_ORDER)),
            "site": st.column_config.SelectboxColumn("site", options=opt(C.SITES)),
            "to_site": st.column_config.SelectboxColumn("to_site", options=opt(C.SITES)),
            "tool_group": st.column_config.SelectboxColumn("tool_group", options=opt(C.TOOL_GROUP_ORDER)),
            "from_quarter": st.column_config.SelectboxColumn("from_quarter", options=opt(C.PLAN_QUARTERS)),
            "to_quarter": st.column_config.SelectboxColumn("to_quarter", options=opt(C.PLAN_QUARTERS)),
            "source_text": st.column_config.TextColumn("source_text", width="medium"),
        })
    st.markdown("##### 3 · Run the capacity model")
    if st.button("Run scenario", type="primary"):
        changes, errors = df_to_changes(edited)
        for e in errors:
            st.error(e)
        if changes:
            ss.scenario = fc.run_scenario(data, changes, basis, ceiling)
            ss.readout = None
        elif not errors:
            st.info("Add at least one change.")

    res = ss.scenario
    if res:
        for w in res["warnings"]:
            st.warning(w)
        k = st.columns(4)
        card(k[0], "At-risk cells", f"{res['constraints_before']} → {res['constraints_after']}", "both fabs, 2027",
             GREEN if res["constraints_after"] <= res["constraints_before"] else RED)
        card(k[1], "New constraints", f"{len(res['new_constraints'])}", ", ".join(res["new_constraints"][:3]) or "none",
             RED if res["new_constraints"] else GREEN)
        card(k[2], "Cleared", f"{len(res['cleared_constraints'])}", ", ".join(res["cleared_constraints"][:3]) or "none", GREEN)
        card(k[3], "Changes applied", f"{len(res['applied_changes'])}", res["basis"] + " basis")
        st.markdown("**Applied:** " + " · ".join(res["applied_changes"]))
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("##### Scenario loading (change vs baseline in brackets)")
            delta = res["_tables"]["base"].merge(res["_tables"]["scenario"], on=["site", "tool_group", "quarter"],
                                                 suffixes=("_b", ""))
            delta["loading_change_pts"] = delta["loading_pct"] - delta["loading_pct_b"]
            st.plotly_chart(heatmap(res["_tables"]["scenario"], delta=delta), width="stretch", key="chart_6")
        with c2:
            st.markdown("##### 4 · Leadership readout")
            if st.button("Draft readout with AI", disabled=not API_KEY, width="stretch"):
                try:
                    with st.spinner("Drafting..."):
                        ss.readout = ai.draft_readout(get_llm(MODEL, API_KEY), res)
                except Exception as exc:
                    ss.readout = f"Readout failed: {exc}"
            if ss.readout:
                st.markdown(ss.readout)
                st.caption("Written only from the scenario results below.")
            with st.expander("Scenario results the readout uses"):
                st.json({k: v for k, v in res.items() if not k.startswith("_")}, expanded=False)

# ---------------------------------------------------------------- performance
with t_perf:
    w = data["sim_fab_weekly"]
    w = w[w["site"].isin(sites())]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Wafer starts and outs per week")
        ww = w.groupby("week_start", as_index=False)[["wafer_starts", "wafer_outs"]].sum()
        fig = go.Figure()
        fig.add_bar(x=ww["week_start"], y=ww["wafer_starts"], name="Starts", marker_color=BLUE)
        fig.add_scatter(x=ww["week_start"], y=ww["wafer_outs"], name="Outs", mode="lines+markers", line=dict(color=GREEN))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch", key="chart_7")
    with c2:
        st.markdown("##### Cycle time: hot vs normal lots (days)")
        ct = w.groupby("week_start", as_index=False)[["hot_cycle_time_days", "normal_cycle_time_days"]].mean()
        fig = px.line(ct.rename(columns={"hot_cycle_time_days": "Hot lots", "normal_cycle_time_days": "Normal lots"}),
                      x="week_start", y=["Hot lots", "Normal lots"], markers=True,
                      color_discrete_sequence=[RED, BLUE], labels={"value": "Days", "week_start": "", "variable": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch", key="chart_8")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Where cycle time goes (full-flow programs)")
        cp = fd.cycle_time_by_program(data, site)
        cp = cp[cp["program"] != "SHORT"]
        cp["lot type"] = cp["site"].map(SITE_LABEL) + " · " + cp["program"].map(PROG)
        m = cp.melt(id_vars="lot type", value_vars=["process", "queue", "transport", "hold"], var_name="Part", value_name="Days")
        m["Part"] = m["Part"].map({"process": "Processing", "queue": "Queue at tool", "transport": "Transport/dispatch",
                                   "hold": "Engineering hold"})
        fig = px.bar(m, y="lot type", x="Days", color="Part", orientation="h",
                     color_discrete_sequence=[GREEN, ORANGE, "#95a5a6", RED], labels={"lot type": ""})
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(fig, width="stretch", key="chart_9")
    with c2:
        st.markdown("##### Tool time by SEMI E10 state")
        tg = fd.tool_group_states(data, site)
        tg["tool group"] = row_label(tg)
        m = tg.melt(id_vars="tool group", value_vars=["productive_pct", "standby_pct", "engineering_pct",
                                                      "sched_down_pct", "unsched_down_pct"], var_name="State", value_name="%")
        m["State"] = m["State"].map({"productive_pct": "Productive", "standby_pct": "Standby", "engineering_pct": "Engineering",
                                     "sched_down_pct": "Scheduled down", "unsched_down_pct": "Unscheduled down"})
        fig = px.bar(m, y="tool group", x="%", color="State", orientation="h",
                     color_discrete_sequence=[GREEN, "#cfd8dc", BLUE, AMBER, RED], labels={"tool group": ""})
        fig.update_layout(height=320 + 8 * len(tg), margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=1.08, title=""))
        st.plotly_chart(fig, width="stretch", key="chart_10")

# ---------------------------------------------------------------- how it works
with t_how:
    st.markdown("##### How the AI fits into the planning cycle")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fa", fontname="Helvetica", fontsize=11];
  note [label="Planning note\\n(PCN, program request,\\nequipment update)", fillcolor="#fdf2e9"];
  ext [label="AI: extract\\nstructured changes", fillcolor="#e8f6ef"];
  human [label="Planner confirms\\nor edits changes", fillcolor="#fff7d6"];
  model [label="Capacity model\\n(deterministic)"];
  fc [label="Forecast model\\n(bias correction)"];
  read [label="AI: leadership readout\\n(numbers from results only)", fillcolor="#e8f6ef"];
  chat [label="AI copilot chat\\n(tool calling + SQL + document search)", fillcolor="#e8f6ef"];
  note -> ext -> human -> model -> read; fc -> model; chat -> model; chat -> fc;
}
""")
    st.markdown("""
**The rule: the AI never does the math.** It reads notes, chooses which model or query to run, and explains the
results. Every number comes from the capacity model, the forecast model, the simulation tables or a read-only SQL
query, and a person confirms extracted changes before the model runs.

**Capacity model** (per site, tool group and quarter)
- Required hours/week = Σ programs WSPW × passes × minutes per pass (× metrology sample rate) ÷ 60
- Available hours/week = tools × 168 × production availability (calendar time minus scheduled down,
  unscheduled down and engineering time)
- Loading = required ÷ available; tools needed = ⌈required ÷ (168 × availability × ceiling)⌉

**Forecast model**: actual starts = b × requested starts, fitted per program (least squares through the origin);
b is the realization rate. Backtested on the last three quarters.

**Fab performance data**: a SimPy discrete-event simulation of each fab for Q3 2026: lots of 25 wafers on
re-entrant routes, hot-lot priority, batch furnaces, sampled metrology, PM / breakdowns / engineering time,
transport and engineering holds.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Validation 1 · capacity model vs simulation (Q3 2026 loading %)**")
        st.dataframe(fc.model_vs_simulation(data), hide_index=True, width="stretch", height=300)
    with c2:
        st.markdown("**Validation 2 · Little's Law (WIP = throughput × cycle time)**")
        st.dataframe(fd.littles_law_check(data), hide_index=True, width="stretch")
        st.markdown("**Limits**")
        st.markdown("- Illustrative data, not any company's. Tool groups are pooled (no tool-level dedication).\n"
                    "- Routes are aggregated to passes per tool group, not full step lists.\n"
                    "- Loading is a weekly average; the simulation shows queues, the model does not.")
    with st.expander("Assumptions: tool groups"):
        st.dataframe(data["tool_groups"], hide_index=True, width="stretch")
    with st.expander("Assumptions: routes (passes per wafer)"):
        st.dataframe(data["routes"].pivot(index="tool_group", columns="program", values="passes")
                     .reindex(C.TOOL_GROUP_ORDER), width="stretch")
    with st.expander("Assumptions: installed tools and qualification"):
        st.dataframe(data["installed_tools"].pivot(index="tool_group", columns="site", values="tools")
                     .reindex(C.TOOL_GROUP_ORDER), width="stretch")
        st.dataframe(data["qualification"].pivot(index="program", columns="site", values="qualified"), width="stretch")

# ======================================================================
# Copilot chat
# ======================================================================
st.markdown("---")
h1, h2 = st.columns([6, 1])
h1.subheader("Planning copilot")
h1.caption("Ask in plain English. It runs the capacity model, forecast model, SQL and document search, "
           "and shows every step.")
if h2.button("Clear chat", width="stretch"):
    ss.messages = []
    st.rerun()
ss.setdefault("messages", [])

SUGGESTIONS = [
    "Which tool groups are constrained in 2027, and how does the bias-corrected forecast change that?",
    "Fab B etch goes over 100% in 2027. Would moving 25% of NAND Gen C starts to Fab A fix it without new constraints?",
    "What is driving cycle time at Fab B, and what did past planning reviews decide about it?",
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
    st.info("The copilot needs an OpenAI API key: add OPENAI_API_KEY to the app's Streamlit secrets "
            "(or a local .env file). Everything above works without it.")
else:
    cols = st.columns(len(SUGGESTIONS))
    clicked = None
    for col, q in zip(cols, SUGGESTIONS):
        if col.button(q, width="stretch"):
            clicked = q
    question = st.chat_input("Ask about loading, constraints, forecasts, scenarios or fab performance...") or clicked
    if question:
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            ctx = {"site": site, "basis": basis}
            try:
                tools = ai.make_tools(data, ctx, fd.DB_PATH, get_doc_search(API_KEY))
                with st.spinner("Running the models..."):
                    out = ai.answer_question(question, ss.messages[:-1], get_llm(MODEL, API_KEY), tools, ctx)
                answer, trace = out["answer"], out["trace"]
            except Exception as exc:
                answer, trace = f"Sorry, the copilot hit an error: `{exc}`", []
            st.markdown(answer)
            render_trace(trace)
        ss.messages.append({"role": "assistant", "content": answer, "trace": trace})

st.markdown("---")
st.caption("Portfolio project by Surya Vasanth · synthetic, illustrative data · not affiliated with any company")
