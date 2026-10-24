"""Deterministic review metrics. Display filters never remove model workload."""
from __future__ import annotations
import math
from datetime import timedelta
import pandas as pd
import fab_capacity as fc
import fab_config as C

FOCUS_GROUPS = ["EUV", "ETCH", "DEP"]
SITE_LABELS = {"FAB_A": "Fab A", "FAB_B": "Fab B"}
GROUP_LABELS = {"EUV": "EUV lithography", "DUV": "DUV lithography", "ETCH": "Etch",
                "DEP": "Deposition", "PVD": "Metal deposition", "CMP": "Planarization",
                "WET": "Wet clean", "FURN": "Furnace", "IMPL": "Ion implant", "METRO": "Inspection"}

def select_scope(table, site, quarter=None):
    out = table[table["site"] == site]
    return out[out["quarter"] == quarter].copy() if quarter is not None else out.copy()

def risk_label(load, ceiling=0.85):
    if pd.isna(load): return "No capacity"
    if load > 100: return "Over capacity"
    if load >= ceiling * 100: return "Above planning limit"
    return "Within planning limit"

def capacity_summary(table, state, site, quarter, ceiling=0.85):
    """Summarize ALL equipment groups in the chosen site and quarter."""
    t = select_scope(table, site, quarter)
    i = C.PLAN_QUARTERS.index(quarter)
    worst = t.assign(_sort=t["loading_pct"].fillna(float("inf"))).sort_values("_sort", ascending=False).iloc[0]
    return {"starts": sum(state["wspw"][site][p][i] for p in C.PROGRAM_ORDER),
            "at_risk": int((t["loading_pct"].isna() | (t["loading_pct"] >= ceiling * 100)).sum()),
            "over_capacity": int((t["loading_pct"].isna() | (t["loading_pct"] > 100)).sum()),
            "groups": len(t), "worst_group": worst["tool_group"], "worst_load": worst["loading_pct"]}

def program_workload(state, site, group, quarter):
    i = C.PLAN_QUARTERS.index(quarter)
    rows = []
    for p in C.PROGRAM_ORDER:
        starts = state["wspw"][site][p][i]
        if starts <= 0: continue
        tool = C.TOOL_GROUPS[group]
        passes = state["passes"][p][group][i]
        minutes = tool["min_per_wafer"] * C.PROGRAMS[p]["time_mult"].get(group, 1.0)
        sample = tool.get("sample_rate", 1.0)
        rows.append({"Program": C.PROGRAMS[p]["name"], "Starts / week": starts,
                     "Passes / wafer": passes, "Minutes / pass": minutes,
                     "Required h / week": starts * passes * minutes * sample / 60,
                     "Sample fraction": sample})
    result = pd.DataFrame(rows)
    total = result["Required h / week"].sum()
    result["Share %"] = result["Required h / week"] / total * 100 if total else 0.0
    return result.sort_values("Required h / week", ascending=False).reset_index(drop=True)

def compare_tables(baseline, scenario, ceiling=0.85):
    t = baseline.merge(scenario, on=["site", "tool_group", "quarter"], suffixes=("_base", "_scenario"))
    t["change_pp"] = t["loading_pct_scenario"] - t["loading_pct_base"]
    t["status"] = t["loading_pct_scenario"].map(lambda x: risk_label(x, ceiling))
    return t

def run_review(data, site, program, pct, first_quarter, last_quarter, ceiling=0.85,
               recovery_group=None, recovery_points=0.0):
    if program not in C.QUALIFIED[site]:
        raise ValueError("This program is not qualified at the selected site.")
    if not math.isfinite(pct) or not -100 <= pct <= 100:
        raise ValueError("Choose a wafer-start change between -100% and +100%.")
    if C.PLAN_QUARTERS.index(first_quarter) > C.PLAN_QUARTERS.index(last_quarter):
        raise ValueError("The end quarter must be on or after the start quarter.")
    changes = [fc.PlanChange(type="starts_change", site=site, program=program, pct=pct,
                            from_quarter=first_quarter, to_quarter=last_quarter)]
    if recovery_points:
        if recovery_group not in C.TOOL_GROUP_ORDER or C.INSTALLED_TOOLS[site][recovery_group] <= 0:
            raise ValueError("Choose an installed equipment group for availability recovery.")
        if not math.isfinite(recovery_points) or not 0 < recovery_points <= 15:
            raise ValueError("Availability recovery must be greater than 0 and no more than 15 percentage points.")
        if C.production_availability(recovery_group) + recovery_points / 100 > 0.98:
            raise ValueError("This recovery would exceed the model's 98% availability limit.")
        changes.append(fc.PlanChange(type="availability_change", site=site, tool_group=recovery_group,
                                     availability_pts=recovery_points, from_quarter=first_quarter,
                                     to_quarter=last_quarter))
    baseline = fc.base_state(data, "requested")
    scenario, applied, warnings = fc.apply_changes(baseline, changes)
    if warnings: raise ValueError("; ".join(warnings))
    tb, ts = fc.loading_table(baseline, ceiling), fc.loading_table(scenario, ceiling)
    return {"base_state": baseline, "scenario_state": scenario, "baseline": tb, "scenario": ts,
            "comparison": compare_tables(tb, ts, ceiling), "applied": applied,
            "changes": [c.model_dump(exclude_none=True) for c in changes],
            "site": site, "program": program, "pct": pct, "first_quarter": first_quarter,
            "last_quarter": last_quarter, "ceiling": ceiling}

def production_metrics(data, site):
    """Fixed 13-week window; completed-cohort yield; full-flow CT excludes SHORT."""
    weekly = data["sim_fab_weekly"].query("site == @site").sort_values("week_start").copy()
    lots = data["sim_lots"].query("site == @site").copy()
    completed = lots[lots["status"] == "COMPLETE"]
    full = completed[completed["program"] != "SHORT"]
    breakdown = full[["process_days", "queue_days", "transport_days", "hold_days"]].mean()
    by_program = completed.groupby("program").agg(
        completed_lots=("lot_id", "size"), cycle_days=("cycle_time_days", "mean"),
        process_days=("process_days", "mean"), queue_days=("queue_days", "mean"),
        transport_days=("transport_days", "mean"), hold_days=("hold_days", "mean")).reset_index()
    return {"weekly": weekly, "completed": completed, "full": full,
            "avg_outs": weekly["wafer_outs"].mean(), "avg_starts": weekly["wafer_starts"].mean(),
            "cycle_days": full["cycle_time_days"].mean(), "wip": weekly["wip_wafers"].mean(),
            "line_yield": completed["wafers_out"].sum() / completed["wafers_in"].sum() * 100,
            "breakdown": breakdown, "by_program": by_program,
            "window_start": pd.to_datetime(weekly["week_start"].min()).date(),
            "window_end": pd.to_datetime(weekly["week_start"].max()).date() + timedelta(days=6)}

def decision_brief(review, quarter):
    site, ceiling = review["site"], review["ceiling"]
    before = capacity_summary(review["baseline"], review["base_state"], site, quarter, ceiling)
    after = capacity_summary(review["scenario"], review["scenario_state"], site, quarter, ceiling)
    changed = select_scope(review["comparison"], site, quarter)
    biggest = changed.loc[changed["change_pp"].abs().idxmax()]
    if after["over_capacity"]:
        recommendation = "Revise the loading request before commitment. The scenario exceeds available production capacity."
    elif after["at_risk"]:
        recommendation = "Keep the request under review. Confirm an operational response for equipment above the planning limit."
    else:
        recommendation = "The scenario fits the modeled capacity limit. Confirm timing and operating assumptions before commitment."
    return {"recommendation": recommendation, "before": before, "after": after,
            "biggest": biggest.to_dict(),
            "owners": "Engineering: experiment timing · Operations: availability assumption · IE: loading review",
            "limit": "Quarter-average capacity screen. Lot due dates, queue-time impacts and cost savings are not predicted."}
