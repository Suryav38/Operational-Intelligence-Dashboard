"""
Plant metrics: data loading, filtering and every KPI the dashboard and the AI
assistant show. Both use these functions, so a number on a chart and a number
in a chat answer always come from the same calculation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "manrag"
STRUCTURED = DATA_DIR / "structured"
DB_PATH = DATA_DIR / "manrag_factory.db"
DOCS_DIR = DATA_DIR / "unstructured"

# Stops that count as equipment failures for MTBF/MTTR. Jams and changeovers are
# stoppages, not failures, so they are excluded.
FAILURE_CAUSES = ("MECH_FAILURE", "SENSOR_FAULT")

TARGETS = {
    "oee": 80.0,            # %
    "availability": 92.0,   # %
    "scrap_rate": 3.0,      # % (lower is better)
    "mtbf_line_h": 12.0,    # hours between failures that stop the line
    "mtbf_machine_h": 150.0,
    "mttr_min": 45.0,       # minutes (lower is better)
}


def load_data(structured_dir: Path = STRUCTURED) -> dict:
    d = structured_dir
    if not (d / "production_daily.csv").exists():
        raise FileNotFoundError("Data missing. Run `python code/data_generation.py` first.")
    data = {
        "prod": pd.read_csv(d / "production_daily.csv", parse_dates=["date"]),
        "events": pd.read_csv(d / "downtime_events.csv", parse_dates=["shift_date", "start_time", "end_time"]),
        "machines": pd.read_csv(d / "dim_machine.csv"),
        "skus": pd.read_csv(d / "dim_sku.csv"),
        "tickets": pd.read_csv(d / "maintenance_tickets.csv"),
        "demand": pd.read_csv(d / "demand_forecast.csv", parse_dates=["week_start"]),
    }
    data["events"]["is_failure"] = data["events"]["cause_code"].isin(FAILURE_CAUSES)
    return data


def date_bounds(data):
    return data["prod"]["date"].min().date(), data["prod"]["date"].max().date()


def filter_data(data, line="All", machine="All", start=None, end=None):
    """Production is recorded per line; downtime per machine.

    Returns production for the line scope, events for the line scope, and events
    for the machine scope (same as line scope when machine is 'All').
    """
    prod, ev = data["prod"], data["events"]
    lo = pd.Timestamp(start) if start else prod["date"].min()
    hi = pd.Timestamp(end) if end else prod["date"].max()
    p = prod[(prod["date"] >= lo) & (prod["date"] <= hi)]
    e = ev[(ev["shift_date"] >= lo) & (ev["shift_date"] <= hi)]
    if line and line != "All":
        p = p[p["line_id"] == line]
        e = e[e["line_id"] == line]
    e_m = e[e["machine_id"] == machine] if machine and machine != "All" else e
    return p, e, e_m


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def oee_parts(p: pd.DataFrame) -> dict:
    planned_h = p["planned_hours"].sum()
    run_h = p["runtime_hours"].sum()
    produced = p["produced_units"].sum()
    scrap = p["scrap_units"].sum()
    ideal_h = (p["produced_units"] * p["standard_cycle_time_sec"]).sum() / 3600
    a = _safe_div(run_h, planned_h)
    perf = min(1.0, _safe_div(ideal_h, run_h))
    q = _safe_div(produced - scrap, produced)
    return {"availability": a, "performance": perf, "quality": q, "oee": a * perf * q,
            "planned_h": planned_h, "runtime_h": run_h, "produced": int(produced),
            "scrap": int(scrap), "good": int(produced - scrap)}


def reliability(e: pd.DataFrame, line_runtime_h: float) -> dict:
    """MTBF = operating hours / number of failures; MTTR = mean failure repair time.

    For a single machine, operating hours = its line's runtime (a machine on a
    serial line runs when the line runs).
    """
    fails = e[e["is_failure"]]
    n = len(fails)
    return {
        "failures": n,
        "stops": len(e),
        "mtbf_h": _safe_div(line_runtime_h, n) if n else None,
        "mttr_min": float(fails["duration_min"].mean()) if n else None,
        "machine_downtime_h": e["duration_min"].sum() / 60,
    }


def compute_kpis(data, line="All", machine="All", start=None, end=None, downtime_cost_per_h=800.0) -> dict:
    p, e_line, e_m = filter_data(data, line, machine, start, end)
    parts = oee_parts(p)
    # Operating hours = summed line runtime. With all lines selected this gives
    # the average hours between failures on a line; with one machine selected it
    # gives that machine's MTBF (it runs whenever its line runs).
    rel = reliability(e_m, parts["runtime_h"])
    unit_cost = data["skus"].set_index("sku_id")["unit_cost"]
    scrap_cost = float((p["scrap_units"] * p["sku_id"].map(unit_cost)).sum())
    line_dt_h = p["downtime_minutes"].sum() / 60
    return {
        "scope": {"line": line, "machine": machine,
                  "start": str(p["date"].min().date()) if len(p) else None,
                  "end": str(p["date"].max().date()) if len(p) else None},
        "oee_pct": round(parts["oee"] * 100, 1),
        "availability_pct": round(parts["availability"] * 100, 1),
        "performance_pct": round(parts["performance"] * 100, 1),
        "quality_pct": round(parts["quality"] * 100, 2),
        "scrap_rate_pct": round((1 - parts["quality"]) * 100, 2) if parts["produced"] else 0.0,
        "produced_units": parts["produced"],
        "good_units": parts["good"],
        "planned_units": int(p["planned_units"].sum()),
        "plan_attainment_pct": round(_safe_div(parts["good"], p["planned_units"].sum()) * 100, 1),
        "line_downtime_h": round(line_dt_h, 1),
        "machine_scope_downtime_h": round(rel["machine_downtime_h"], 1),
        "stops": rel["stops"],
        "failures": rel["failures"],
        "mtbf_h": round(rel["mtbf_h"], 1) if rel["mtbf_h"] is not None else None,
        "mtbf_basis": "machine" if machine != "All" else "line",
        "mttr_min": round(rel["mttr_min"], 1) if rel["mttr_min"] is not None else None,
        "scrap_cost_usd": int(scrap_cost),
        "downtime_cost_usd_assumed": int(line_dt_h * downtime_cost_per_h),
    }


def daily_series(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("date").agg(produced=("produced_units", "sum"), scrap=("scrap_units", "sum"),
                              planned_units=("planned_units", "sum")).reset_index()
    g["good"] = g["produced"] - g["scrap"]
    g["attainment_pct"] = (g["good"] / g["planned_units"] * 100).round(1)
    return g


def weekly_oee(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week, grp in p.groupby(p["date"].dt.to_period("W")):
        parts = oee_parts(grp)
        rows.append({"week": week.start_time, "OEE": parts["oee"] * 100,
                     "Availability": parts["availability"] * 100,
                     "Performance": parts["performance"] * 100,
                     "Quality": parts["quality"] * 100})
    return pd.DataFrame(rows)


SCRAP_GROUPS = {
    "line": ["line_id"], "supplier": ["supplier_id"], "shift": ["shift_id"], "sku": ["sku_id"],
    "line_supplier": ["line_id", "supplier_id"], "line_shift": ["line_id", "shift_id"],
}


def scrap_breakdown(p: pd.DataFrame, group_by="line", unit_cost=None) -> pd.DataFrame:
    keys = SCRAP_GROUPS[group_by]
    g = p.groupby(keys).agg(produced=("produced_units", "sum"), scrap=("scrap_units", "sum")).reset_index()
    g["scrap_rate_pct"] = (g["scrap"] / g["produced"] * 100).round(2)
    if unit_cost is not None:
        cost = p.assign(c=p["scrap_units"] * p["sku_id"].map(unit_cost)).groupby(keys)["c"].sum()
        g = g.merge(cost.rename("scrap_cost_usd").round(0).reset_index(), on=keys)
    return g.sort_values("scrap_rate_pct", ascending=False).reset_index(drop=True)


DOWNTIME_GROUPS = {"machine": "machine_id", "cause": "cause_code", "shift": "shift_id", "line": "line_id"}


def downtime_breakdown(e: pd.DataFrame, group_by="machine", failures_only=False) -> pd.DataFrame:
    if failures_only:
        e = e[e["is_failure"]]
    key = DOWNTIME_GROUPS[group_by]
    g = (e.groupby(key).agg(stops=("event_id", "size"), downtime_min=("duration_min", "sum"),
                            failures=("is_failure", "sum"))
          .reset_index().sort_values("downtime_min", ascending=False).reset_index(drop=True))
    total = g["downtime_min"].sum()
    g["share_pct"] = (g["downtime_min"] / total * 100).round(1) if total else 0.0
    g["cumulative_pct"] = g["share_pct"].cumsum().round(1)
    return g


def machine_reliability_table(p: pd.DataFrame, e: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """MTBF/MTTR per machine (failures only), using its line's runtime as operating hours."""
    run_by_line = p.groupby("line_id")["runtime_hours"].sum()
    fails = e[e["is_failure"]].groupby("machine_id").agg(failures=("event_id", "size"),
                                                         mttr_min=("duration_min", "mean"))
    t = machines[["machine_id", "line_id", "machine_type"]].merge(fails, on="machine_id", how="left")
    t = t[t["line_id"].isin(run_by_line.index)]
    t["failures"] = t["failures"].fillna(0).astype(int)
    t["operating_h"] = t["line_id"].map(run_by_line)
    t["mtbf_h"] = np.where(t["failures"] > 0, t["operating_h"] / t["failures"].replace(0, np.nan), np.nan)
    t["mtbf_h"] = t["mtbf_h"].round(1)
    t["mttr_min"] = t["mttr_min"].round(1)
    return t.sort_values("mtbf_h").reset_index(drop=True)


def exceptions(p: pd.DataFrame, e: pd.DataFrame, min_units=20000) -> list:
    """Auto-flag segments that stand out from the rest of the plant (watch list)."""
    flags = []
    if p.empty:
        return flags
    plant_rate = p["scrap_units"].sum() / p["produced_units"].sum() * 100
    seen = set()
    for group in ["line_supplier", "line_shift", "supplier", "shift"]:
        b = scrap_breakdown(p, group)
        b = b[(b["produced"] >= min_units) & (b["scrap_rate_pct"] >= 1.4 * plant_rate)]
        for _, r in b.head(1).iterrows():
            label = " + ".join(str(r[k]) for k in SCRAP_GROUPS[group])
            if any(part in seen for part in label.split(" + ")) and group in ("supplier", "shift"):
                continue
            seen.update(label.split(" + "))
            flags.append({"type": "Quality", "text": f"Scrap on {label} is {r['scrap_rate_pct']:.1f}% "
                                                    f"vs {plant_rate:.1f}% across the selection."})
    if not e.empty:
        dm = e.groupby(["line_id", "machine_id"])["duration_min"].sum().reset_index()
        dm["line_median"] = dm.groupby("line_id")["duration_min"].transform("median")
        bad = dm[dm["duration_min"] >= 2.5 * dm["line_median"]].sort_values("duration_min", ascending=False)
        for _, r in bad.head(2).iterrows():
            flags.append({"type": "Reliability",
                          "text": f"{r['machine_id']} ({r['line_id']}) lost {r['duration_min']:,} min, "
                                  f"{r['duration_min'] / r['line_median']:.1f}x the typical machine on its line."})
    return flags
