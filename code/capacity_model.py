"""
Capacity model: can each line meet the Q1 2025 demand forecast?

    Available hours/week = scheduled hours x availability x performance
    Required hours/week  = sum over SKUs of (demand / quality yield) x ideal cycle time
    Loading              = required / available
    Gap                  = required - available x ceiling   (positive = short)

Availability, performance and quality come from recent history (default: the
last 13 weeks of 2024). Demand comes from demand_forecast.csv. Scenario levers
change one of those inputs, so every result traces back to a formula.
"""
import math

import pandas as pd

from plant_metrics import oee_parts

SCHEDULED_SHIFTS_PER_WEEK = 15   # 3 shifts x 5 weekdays
SHIFT_HOURS = 8.0
MAX_EXTRA_SHIFTS = 6             # Saturday + Sunday, 3 shifts each
DEFAULT_CEILING = 0.85           # plan to stay below this loading
HISTORY_WEEKS = 13


def _history(data, weeks=HISTORY_WEEKS):
    prod = data["prod"]
    end = prod["date"].max()
    start = end - pd.Timedelta(weeks=weeks) + pd.Timedelta(days=1)
    return prod[prod["date"] >= start], data["events"][data["events"]["shift_date"] >= start], weeks


def machine_fix_gain(data, machine_id, weeks=HISTORY_WEEKS):
    """Availability points recovered if a machine's downtime fell to its line's median machine."""
    p, e, weeks = _history(data, weeks)
    m = data["machines"].set_index("machine_id")
    if machine_id not in m.index:
        raise ValueError(f"Unknown machine {machine_id}")
    line = m.loc[machine_id, "line_id"]
    per_machine = (e[e["line_id"] == line].groupby("machine_id")["duration_min"].sum()
                   .reindex(m[m["line_id"] == line].index, fill_value=0))
    excess_min = max(0.0, per_machine.get(machine_id, 0) - per_machine.median())
    scheduled_min = p[p["line_id"] == line]["planned_hours"].sum() * 60
    gain = float(excess_min / scheduled_min) if scheduled_min else 0.0
    return line, gain, float(excess_min / weeks / 60)


def _line_result(line, rates, demand_line, skus, growth, avail_gain, scrap_cut, extra_shifts, ceiling):
    a = float(min(0.99, rates["availability"] + avail_gain))
    q = min(0.999, rates["quality"] + scrap_cut)
    perf = rates["performance"]
    scheduled_h = (SCHEDULED_SHIFTS_PER_WEEK + extra_shifts) * SHIFT_HOURS
    available_h = scheduled_h * a * perf

    ct = skus.set_index("sku_id")["ideal_cycle_time_sec"]
    d = demand_line.assign(ct=demand_line["sku_id"].map(ct))
    d["req_h"] = d["demand_units"] * (1 + growth) / q * d["ct"] / 3600
    weekly_req = d.groupby("week_start")["req_h"].sum()
    required_h = float(weekly_req.mean())
    available_h = float(available_h)
    loading = required_h / available_h
    peak = float(weekly_req.max()) / available_h
    gap_h = required_h - available_h * ceiling
    # extra shifts needed to get under the ceiling (with this scenario's rates)
    per_shift_h = SHIFT_HOURS * a * perf * ceiling
    shifts_needed = max(0, math.ceil(gap_h / per_shift_h)) if gap_h > 0 else 0
    status = "Cannot meet demand" if loading > 1 else ("At risk (above ceiling)" if loading > ceiling else "OK")
    return {
        "line": line,
        "scheduled_h_wk": round(scheduled_h, 1),
        "availability_pct": round(a * 100, 1),
        "performance_pct": round(perf * 100, 1),
        "quality_pct": round(q * 100, 2),
        "available_h_wk": round(available_h, 1),
        "required_h_wk": round(required_h, 1),
        "avg_weekly_demand_units": int(demand_line.groupby("week_start")["demand_units"].sum().mean() * (1 + growth)),
        "loading_pct": round(loading * 100, 1),
        "peak_week_loading_pct": round(peak * 100, 1),
        "gap_h_wk": round(gap_h, 1),
        "extra_shifts_to_reach_ceiling": shifts_needed,
        "status": status,
    }


def run_scenario(data, line="All", demand_growth_pct=0.0, availability_gain_pts=0.0,
                 scrap_reduction_pts=0.0, extra_shifts_per_week=0, fix_machine=None,
                 ceiling=DEFAULT_CEILING, history_weeks=HISTORY_WEEKS) -> dict:
    """Run the capacity model for one or all lines under a scenario."""
    p_hist, _, _ = _history(data, history_weeks)
    lines = sorted(p_hist["line_id"].unique()) if line in (None, "All") else [line]
    extra_shifts = int(max(0, min(MAX_EXTRA_SHIFTS, extra_shifts_per_week)))
    fix_line, fix_gain, fix_h = (None, 0.0, 0.0)
    if fix_machine:
        fix_line, fix_gain, fix_h = machine_fix_gain(data, fix_machine, history_weeks)

    results, levers = [], {}
    for ln in lines:
        rates = oee_parts(p_hist[p_hist["line_id"] == ln])
        dem = data["demand"][data["demand"]["line_id"] == ln]
        gain = availability_gain_pts / 100 + (fix_gain if ln == fix_line else 0.0)
        args = dict(skus=data["skus"], growth=demand_growth_pct / 100, ceiling=ceiling)
        base = _line_result(ln, rates, dem, avail_gain=gain, scrap_cut=scrap_reduction_pts / 100,
                            extra_shifts=extra_shifts, **args)
        results.append(base)
        levers[ln] = _rank_levers(data, ln, rates, dem, base, gain, scrap_reduction_pts / 100,
                                  extra_shifts, history_weeks, args)

    return {
        "inputs": {"line": line, "demand_growth_pct": demand_growth_pct,
                   "availability_gain_pts": availability_gain_pts,
                   "scrap_reduction_pts": scrap_reduction_pts,
                   "extra_shifts_per_week": extra_shifts, "fix_machine": fix_machine,
                   "fix_machine_hours_recovered_per_week": round(fix_h, 1) if fix_machine else None,
                   "ceiling_pct": ceiling * 100,
                   "history": f"last {history_weeks} weeks of 2024",
                   "demand": "Q1 2025 forecast (13 weeks)"},
        "lines": results,
        "levers": levers,
    }


def _rank_levers(data, ln, rates, dem, base, gain, scrap_cut, extra_shifts, history_weeks, args):
    """Effect of each single lever on top of the current scenario, ranked by loading reduction."""
    p_hist, e_hist, _ = _history(data, history_weeks)
    e_line = e_hist[e_hist["line_id"] == ln]
    worst = e_line.groupby("machine_id")["duration_min"].sum().idxmax() if not e_line.empty else None
    best_q = max(oee_parts(p_hist[p_hist["line_id"] == x])["quality"] for x in p_hist["line_id"].unique())
    candidates = []
    if worst:
        _, g, h = machine_fix_gain(data, worst, history_weeks)
        candidates.append((f"Bring {worst} downtime to the line median (+{h:.1f} h/wk)", dict(avail=gain + g)))
    candidates.append(("Raise availability 2 points", dict(avail=gain + 0.02)))
    if best_q - rates["quality"] - scrap_cut > 0.001:
        candidates.append((f"Cut scrap to the plant's best line ({(1 - best_q) * 100:.1f}%)",
                           dict(scrap=best_q - rates["quality"])))
    if extra_shifts < MAX_EXTRA_SHIFTS:
        candidates.append(("Add one weekend shift", dict(shifts=extra_shifts + 1)))
    out = []
    for name, ch in candidates:
        r = _line_result(ln, rates, dem, avail_gain=ch.get("avail", gain),
                         scrap_cut=ch.get("scrap", scrap_cut),
                         extra_shifts=ch.get("shifts", extra_shifts), **args)
        out.append({"lever": name, "loading_pct": r["loading_pct"],
                    "loading_change_pts": round(r["loading_pct"] - base["loading_pct"], 1),
                    "status": r["status"]})
    return sorted(out, key=lambda x: x["loading_change_pts"])
