"""Loads the generated tables and computes fab-performance metrics from the simulation."""
from pathlib import Path

import numpy as np
import pandas as pd

import fab_config as C

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fab"
TABLES = DATA / "tables"
DOCS = DATA / "documents"
DB_PATH = DATA / "td_fab.db"


def load() -> dict:
    if not (TABLES / "plan_2027.csv").exists():
        raise FileNotFoundError("Data missing. Run `python code/data_generation.py` first.")
    names = ["tool_groups", "routes", "installed_tools", "tool_arrivals", "qualification", "plan_2027",
             "wafer_start_history", "sim_lots", "sim_tool_group_q3_2026", "sim_fab_weekly", "sim_tool_group_weekly"]
    return {n: pd.read_csv(TABLES / f"{n}.csv") for n in names}


def _sites(site):
    return list(C.SITES) if site in (None, "All", "Both") else [site]


def performance_summary(data, site="Both") -> dict:
    """Q3 2026 fab performance from the simulation (13 weeks)."""
    w = data["sim_fab_weekly"]
    w = w[w["site"].isin(_sites(site))]
    lots = data["sim_lots"]
    lots = lots[lots["site"].isin(_sites(site)) & (lots["status"] == "COMPLETE")]
    full = lots[lots["program"] != "SHORT"]
    by_site_week = w.groupby("week_start")[["wafer_starts", "wafer_outs", "moves_per_day", "wip_wafers"]].sum()
    planned_wspw = sum(sum(C.ACTUAL_WSPW_2026Q3[s].values()) for s in _sites(site))
    layers = full["program"].map(lambda p: C.PROGRAMS[p]["mask_layers"])
    return {
        "site": site, "period": "Q3 2026 (13 simulated weeks)",
        "wafer_starts_per_week": int(by_site_week["wafer_starts"].mean()),
        "planned_starts_per_week": planned_wspw,
        "wafer_outs_per_week": int(by_site_week["wafer_outs"].mean()),
        "moves_per_day": int(by_site_week["moves_per_day"].mean()),
        "wip_wafers": int(by_site_week["wip_wafers"].mean()),
        "cycle_time_days_full_flow": round(full["cycle_time_days"].mean(), 1),
        "cycle_time_days_hot": round(full[full["priority"] == "HOT"]["cycle_time_days"].mean(), 1),
        "cycle_time_days_normal": round(full[full["priority"] == "NORMAL"]["cycle_time_days"].mean(), 1),
        "days_per_mask_layer": round((full["cycle_time_days"] / layers).mean(), 2),
        "x_factor": round((full["cycle_time_days"] * 24 / full["raw_process_h"]).mean(), 2),
        "line_yield_pct": round(lots["wafers_out"].sum() / lots["wafers_in"].sum() * 100, 2),
        "cycle_time_breakdown_days": {
            k.replace("_days", ""): round(full[k].mean(), 1)
            for k in ["process_days", "queue_days", "transport_days", "hold_days"]},
    }


def cycle_time_by_program(data, site="Both") -> pd.DataFrame:
    lots = data["sim_lots"]
    lots = lots[lots["site"].isin(_sites(site)) & (lots["status"] == "COMPLETE")]
    g = lots.groupby(["site", "program"]).agg(
        lots=("lot_id", "size"), cycle_time_days=("cycle_time_days", "mean"),
        process=("process_days", "mean"), queue=("queue_days", "mean"),
        transport=("transport_days", "mean"), hold=("hold_days", "mean")).reset_index()
    g["days_per_mask_layer"] = g.apply(lambda r: r["cycle_time_days"] / C.PROGRAMS[r["program"]]["mask_layers"], axis=1)
    return g.round(2)


def tool_group_states(data, site="Both") -> pd.DataFrame:
    t = data["sim_tool_group_q3_2026"]
    return t[t["site"].isin(_sites(site))].reset_index(drop=True)


def littles_law_check(data) -> pd.DataFrame:
    """WIP should equal throughput x cycle time (Little's Law) if the simulation is consistent."""
    rows = []
    for site in C.SITES:
        lots = data["sim_lots"]
        done = lots[(lots["site"] == site) & (lots["status"] == "COMPLETE")]
        w = data["sim_fab_weekly"]
        w = w[w["site"] == site]
        th_lots_per_day = len(done) / (len(w) * 7)
        ct = done["cycle_time_days"].mean()
        wip_lots = w["wip_wafers"].mean() / C.LOT_SIZE
        rows.append({"site": site, "throughput_lots_per_day": round(th_lots_per_day, 1),
                     "avg_cycle_time_days": round(ct, 1),
                     "predicted_wip_lots": round(th_lots_per_day * ct),
                     "simulated_wip_lots": round(wip_lots)})
    return pd.DataFrame(rows)


def operating_curve(data, site="FAB_A", tool_group="EUV"):
    """Queue time vs loading for a tool group: Kingman shape u/(1-u), calibrated to the simulated point."""
    t = data["sim_tool_group_q3_2026"]
    r = t[(t["site"] == site) & (t["tool_group"] == tool_group)].iloc[0]
    u0, q0, pt = r["loading_pct"] / 100, r["avg_queue_h"], r["avg_process_h"]
    k = q0 / (u0 / (1 - u0) * pt) if u0 < 1 and pt > 0 else 1.0
    u = np.linspace(0.30, 0.97, 60)
    return {"u_pct": (u * 100).round(1), "queue_h": (k * u / (1 - u) * pt).round(2),
            "sim_point": (round(u0 * 100, 1), q0), "k": round(k, 2), "process_h": pt}
