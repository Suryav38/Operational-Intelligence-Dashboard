"""
Capacity model and scenario engine for the 2027 plan.

For each site, tool group and quarter:
    required hours/week  = sum over programs of WSPW x passes x minutes per pass (x metrology sample) / 60
    available hours/week = tools x 168 x production availability
    loading              = required / available
    tools needed         = ceil(required / (168 x availability x ceiling))

A scenario is a list of PlanChange records (the same structure the AI fills in
from a route-change note). Each change edits one input; the model then reruns.
"""
import copy
import math
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

import fab_config as C
import forecast_model as fm

Q = C.PLAN_QUARTERS
Quarter = Literal[tuple(Q)]
Program = Literal[tuple(C.PROGRAM_ORDER)]
Site = Literal[tuple(C.SITES)]
ToolGroup = Literal[tuple(C.TOOL_GROUP_ORDER)]
ChangeType = Literal["route_change", "pull_in", "starts_change", "tool_delay", "tool_add",
                     "availability_change", "move_load"]


class PlanChange(BaseModel):
    """One change to the plan. Fill only the fields the change type needs."""
    type: ChangeType = Field(description=(
        "route_change: program, tool_group, passes_delta, from_quarter | "
        "pull_in: program, quarters (ramp reached N quarters earlier; negative = push out) | "
        "starts_change: program, site (optional), wspw_delta or pct, from_quarter/to_quarter (optional) | "
        "tool_delay: site, tool_group, quarters (planned arrival slips) | "
        "tool_add: site, tool_group, tools, from_quarter | "
        "availability_change: site, tool_group, availability_pts (negative = loss), from_quarter/to_quarter | "
        "move_load: program, site (from), to_site, pct of that program's starts, from_quarter"))
    program: Optional[Program] = None
    site: Optional[Site] = None
    to_site: Optional[Site] = None
    tool_group: Optional[ToolGroup] = None
    passes_delta: Optional[int] = Field(None, description="Change in passes per wafer at the tool group")
    wspw_delta: Optional[float] = Field(None, description="Change in wafer starts per week")
    pct: Optional[float] = Field(None, description="Percent, e.g. 10 for +10% or 20 for moving 20%")
    quarters: Optional[int] = None
    tools: Optional[int] = None
    availability_pts: Optional[float] = None
    from_quarter: Optional[Quarter] = None
    to_quarter: Optional[Quarter] = None
    source_text: Optional[str] = Field(None, description="Short quote from the note this change came from")


# ------------------------------------------------------------------ plan state
def base_state(data, basis="requested"):
    plan = data["plan_2027"]
    if basis == "corrected":
        plan = fm.corrected_plan(plan, fm.fit(data["wafer_start_history"]))
        col = "corrected_wspw"
    else:
        col = "requested_wspw"
    wspw = {s: {p: [0.0] * len(Q) for p in C.PROGRAM_ORDER} for s in C.SITES}
    for r in plan.itertuples():
        wspw[r.site][r.program][Q.index(r.quarter)] = float(getattr(r, col))
    passes = {p: {tg: [float(n)] * len(Q) for tg, n in C.PROGRAMS[p]["passes"].items()} for p in C.PROGRAM_ORDER}
    arrivals = [list(a) for a in C.TOOL_ARRIVALS]
    tools = {s: {tg: [float(n)] * len(Q) for tg, n in C.INSTALLED_TOOLS[s].items()} for s in C.SITES}
    avail = {s: {tg: [C.production_availability(tg)] * len(Q) for tg in C.TOOL_GROUP_ORDER} for s in C.SITES}
    return {"basis": basis, "wspw": wspw, "passes": passes, "arrivals": arrivals, "tools_base": tools, "avail": avail,
            "extra_tools": []}


def _tools(state):
    tools = copy.deepcopy(state["tools_base"])
    for site, tg, quarter, n in state["arrivals"] + state["extra_tools"]:
        if quarter in Q:
            for i in range(Q.index(quarter), len(Q)):
                tools[site][tg][i] += n
    return tools


def _range(ch):
    lo = Q.index(ch.from_quarter) if ch.from_quarter else 0
    hi = Q.index(ch.to_quarter) if ch.to_quarter else len(Q) - 1
    return range(lo, hi + 1)


def _shift_quarter(q, n):
    year, k = int(q[:4]), int(q[-1]) - 1 + n
    return f"{year + k // 4}Q{k % 4 + 1}"


def apply_changes(state, changes):
    """Apply PlanChange records; returns (new_state, applied descriptions, warnings)."""
    s = copy.deepcopy(state)
    applied, warnings = [], []
    for raw in changes:
        ch = raw if isinstance(raw, PlanChange) else PlanChange(**raw)
        try:
            t = ch.type
            if t == "route_change":
                _need(ch, "program", "tool_group", "passes_delta")
                for i in _range(ch):
                    s["passes"][ch.program][ch.tool_group][i] = max(0.0, s["passes"][ch.program][ch.tool_group][i] + ch.passes_delta)
                applied.append(f"{C.PROGRAMS[ch.program]['name']}: {ch.passes_delta:+d} {ch.tool_group} passes per wafer from {ch.from_quarter or Q[0]}")
            elif t == "pull_in":
                _need(ch, "program", "quarters")
                for site in ([ch.site] if ch.site else C.SITES):
                    old = s["wspw"][site][ch.program]
                    s["wspw"][site][ch.program] = [old[min(max(i + ch.quarters, 0), len(Q) - 1)] for i in range(len(Q))]
                applied.append(f"{C.PROGRAMS[ch.program]['name']}: ramp {'pulled in' if ch.quarters > 0 else 'pushed out'} {abs(ch.quarters)} quarter(s)")
            elif t == "starts_change":
                _need(ch, "program")
                if ch.wspw_delta is None and ch.pct is None:
                    raise ValueError("needs wspw_delta or pct")
                sites = [ch.site] if ch.site else [x for x in C.SITES if ch.program in C.QUALIFIED[x]]
                for site in sites:
                    for i in _range(ch):
                        v = s["wspw"][site][ch.program][i]
                        if v == 0 and ch.wspw_delta is None:
                            continue
                        s["wspw"][site][ch.program][i] = max(0.0, v * (1 + (ch.pct or 0) / 100) + (ch.wspw_delta or 0) / len(sites))
                what = f"{ch.wspw_delta:+.0f} WSPW" if ch.wspw_delta is not None else f"{ch.pct:+.0f}%"
                applied.append(f"{C.PROGRAMS[ch.program]['name']}: starts {what} ({', '.join(sites)}) {_qtxt(ch)}")
            elif t == "move_load":
                _need(ch, "program", "site", "to_site", "pct")
                if ch.program not in C.QUALIFIED[ch.to_site]:
                    raise ValueError(f"{ch.program} is not qualified at {ch.to_site}")
                for i in _range(ch):
                    moved = s["wspw"][ch.site][ch.program][i] * ch.pct / 100
                    s["wspw"][ch.site][ch.program][i] -= moved
                    s["wspw"][ch.to_site][ch.program][i] += moved
                applied.append(f"Move {ch.pct:.0f}% of {C.PROGRAMS[ch.program]['name']} starts {ch.site} -> {ch.to_site} {_qtxt(ch)}")
            elif t == "tool_delay":
                _need(ch, "site", "tool_group")
                n = ch.quarters or 1
                hit = [a for a in s["arrivals"] if a[0] == ch.site and a[1] == ch.tool_group]
                if not hit:
                    raise ValueError(f"no planned arrival for {ch.site} {ch.tool_group}")
                for a in hit:
                    a[2] = _shift_quarter(a[2], n)
                applied.append(f"{ch.site} {ch.tool_group} arrival delayed {n} quarter(s) to {hit[0][2]}")
            elif t == "tool_add":
                _need(ch, "site", "tool_group", "tools")
                q = ch.from_quarter or Q[0]
                s["extra_tools"].append([ch.site, ch.tool_group, q, ch.tools])
                applied.append(f"{ch.site} {ch.tool_group}: add {ch.tools} tool(s) from {q}")
            elif t == "availability_change":
                _need(ch, "site", "tool_group", "availability_pts")
                for i in _range(ch):
                    a = s["avail"][ch.site][ch.tool_group][i]
                    s["avail"][ch.site][ch.tool_group][i] = min(0.98, max(0.05, a + ch.availability_pts / 100))
                applied.append(f"{ch.site} {ch.tool_group}: availability {ch.availability_pts:+.1f} pts {_qtxt(ch)}")
        except (ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Skipped {ch.type}: {exc}")
    return s, applied, warnings


def _need(ch, *fields):
    missing = [f for f in fields if getattr(ch, f) is None]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")


def _qtxt(ch):
    if ch.from_quarter or ch.to_quarter:
        return f"({ch.from_quarter or Q[0]}-{ch.to_quarter or Q[-1]})"
    return "(all 2027)"


# ------------------------------------------------------------------ model
def _hours_per_wafer(state, program, tg, i):
    t, p = C.TOOL_GROUPS[tg], C.PROGRAMS[program]
    return state["passes"][program][tg][i] * t["min_per_wafer"] * p["time_mult"].get(tg, 1.0) * t.get("sample_rate", 1.0) / 60


def loading_table(state, ceiling=C.LOADING_CEILING) -> pd.DataFrame:
    tools = _tools(state)
    rows = []
    for site in C.SITES:
        for tg in C.TOOL_GROUP_ORDER:
            for i, q in enumerate(Q):
                req = sum(state["wspw"][site][p][i] * _hours_per_wafer(state, p, tg, i) for p in C.PROGRAM_ORDER)
                n = tools[site][tg][i]
                per_tool = C.HOURS_PER_WEEK * state["avail"][site][tg][i]
                if n == 0 and req == 0:
                    continue
                needed = math.ceil(req / (per_tool * ceiling) - 1e-9) if req > 0 else 0
                rows.append({"site": site, "tool_group": tg, "quarter": q, "tools": int(n),
                             "required_h_wk": round(req, 1), "available_h_wk": round(n * per_tool, 1),
                             "loading_pct": round(req / (n * per_tool) * 100, 1) if n else None,
                             "tools_needed_at_ceiling": needed, "tool_gap": max(0, needed - int(n))})
    return pd.DataFrame(rows)


def constraints(table, ceiling=C.LOADING_CEILING) -> pd.DataFrame:
    t = table[(table["loading_pct"].isna()) | (table["loading_pct"] >= ceiling * 100)].copy()
    t["status"] = t["loading_pct"].apply(lambda v: "Cannot run (over 100%)" if v is None or v > 100 else "At risk (above ceiling)")
    return t.sort_values(["site", "tool_group", "quarter"]).reset_index(drop=True)


def drivers(state, site, tg, quarter) -> list:
    i = Q.index(quarter)
    hrs = {p: state["wspw"][site][p][i] * _hours_per_wafer(state, p, tg, i) for p in C.PROGRAM_ORDER}
    total = sum(hrs.values()) or 1
    return [{"program": C.PROGRAMS[p]["name"], "share_pct": round(h / total * 100, 1)}
            for p, h in sorted(hrs.items(), key=lambda x: -x[1]) if h > 0]


def plan_frame(state) -> pd.DataFrame:
    return pd.DataFrame([{"site": s, "program": p, "quarter": q, "wspw": round(state["wspw"][s][p][i])}
                         for s in C.SITES for p in C.PROGRAM_ORDER for i, q in enumerate(Q)
                         if state["wspw"][s][p][i] > 0])


def model_vs_simulation(data) -> pd.DataFrame:
    """Validation: the capacity model's loading for Q3 2026 vs the simulated utilization."""
    sim = data["sim_tool_group_q3_2026"].set_index(["site", "tool_group"])
    rows = []
    for site, progs in C.ACTUAL_WSPW_2026Q3.items():
        for tg, n in C.INSTALLED_TOOLS[site].items():
            if n == 0:
                continue
            req = sum(w * C.tool_hours_per_wafer(p, tg) for p, w in progs.items())
            model = req / (n * C.HOURS_PER_WEEK * C.production_availability(tg)) * 100
            rows.append({"site": site, "tool_group": tg, "model_loading_pct": round(model, 1),
                         "simulated_loading_pct": sim.loc[(site, tg), "loading_pct"],
                         "difference_pts": round(sim.loc[(site, tg), "loading_pct"] - model, 1)})
    return pd.DataFrame(rows)


def run_scenario(data, changes, basis="requested", ceiling=C.LOADING_CEILING) -> dict:
    base = base_state(data, basis)
    scen, applied, warnings = apply_changes(base, changes)
    tb, ts = loading_table(base, ceiling), loading_table(scen, ceiling)
    m = tb.merge(ts, on=["site", "tool_group", "quarter"], suffixes=("_base", "_scen"))
    m["loading_change_pts"] = (m["loading_pct_scen"] - m["loading_pct_base"]).round(1)
    changed = m[m["loading_change_pts"].abs() >= 0.5]
    cb, cs = constraints(tb, ceiling), constraints(ts, ceiling)
    key = lambda d: set(zip(d["site"], d["tool_group"], d["quarter"]))
    new = key(cs) - key(cb)
    cleared = key(cb) - key(cs)
    worst = cs.sort_values("loading_pct", ascending=False).head(6)
    return {
        "basis": basis, "ceiling_pct": ceiling * 100, "applied_changes": applied, "warnings": warnings,
        "constraints_before": len(cb), "constraints_after": len(cs),
        "new_constraints": sorted(f"{s} {t} {q}" for s, t, q in new),
        "cleared_constraints": sorted(f"{s} {t} {q}" for s, t, q in cleared),
        "largest_changes": changed.reindex(changed["loading_change_pts"].abs().sort_values(ascending=False).index)
                                 .head(8)[["site", "tool_group", "quarter", "loading_pct_base", "loading_pct_scen",
                                           "loading_change_pts", "tool_gap_scen"]].to_dict(orient="records"),
        "worst_cells_after": [
            {**r, "drivers": drivers(scen, r["site"], r["tool_group"], r["quarter"])[:3]}
            for r in worst[["site", "tool_group", "quarter", "loading_pct", "tools", "tools_needed_at_ceiling",
                            "tool_gap"]].to_dict(orient="records")],
        "_tables": {"base": tb, "scenario": ts},
    }
