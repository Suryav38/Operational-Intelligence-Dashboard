"""
Generate all data for the TD Fab Capacity & Wafer Loading Planner.

    python code/data_generation.py

Steps:
  1. Run the SimPy fab simulation (fab_sim.py) for Q3 2026 at both sites.
  2. Write the planning tables: tool groups, routes, installed tools, arrivals,
     qualification, requested-vs-actual wafer-start history, the 2027 request.
  3. Write text documents for the assistant's search (planning-review minutes,
     past route-change notices, tool-group notes, planning guidelines).
  4. Load every table into a SQLite database the assistant can query read-only.

All data is synthetic and illustrative.
"""
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import fab_config as C
from fab_sim import simulate_all

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fab"
TABLES = DATA / "tables"
DOCS = DATA / "documents"
DB = DATA / "td_fab.db"
RNG = np.random.default_rng(2026)


def reset():
    shutil.rmtree(DATA, ignore_errors=True)
    shutil.rmtree(ROOT / "data" / "manrag", ignore_errors=True)   # old assembly-plant data
    for d in [TABLES, DOCS / "planning_reviews", DOCS / "route_changes", DOCS / "tool_notes", DOCS / "guidelines"]:
        d.mkdir(parents=True, exist_ok=True)


def config_tables():
    tg = pd.DataFrame([{"tool_group": k, "name": v["name"], "area": v["area"],
                        "min_per_wafer": v["min_per_wafer"], "sched_down_frac": v["pm"],
                        "unsched_down_frac": v["down"], "engineering_frac": v["eng"],
                        "production_availability": round(C.production_availability(k), 3),
                        "batch_lots": v.get("batch_lots"), "run_hours": v.get("run_hours"),
                        "sample_rate": v.get("sample_rate")} for k, v in C.TOOL_GROUPS.items()])
    routes = pd.DataFrame([{"program": p, "program_name": v["name"], "mask_layers": v["mask_layers"],
                            "tool_group": t, "passes": n, "time_mult": v["time_mult"].get(t, 1.0),
                            "tool_hours_per_wafer": round(C.tool_hours_per_wafer(p, t), 4)}
                           for p, v in C.PROGRAMS.items() for t, n in v["passes"].items()])
    installed = pd.DataFrame([{"site": s, "tool_group": t, "tools": n}
                              for s, d in C.INSTALLED_TOOLS.items() for t, n in d.items()])
    arrivals = pd.DataFrame(C.TOOL_ARRIVALS, columns=["site", "tool_group", "quarter", "tools"])
    qual = pd.DataFrame([{"site": s, "program": p, "qualified": p in C.QUALIFIED[s]}
                         for s in C.SITES for p in C.PROGRAMS])
    plan = pd.DataFrame([{"site": s, "program": p, "quarter": q, "requested_wspw": v[i]}
                         for s, d in C.REQUESTED_WSPW_2027.items() for p, v in d.items()
                         for i, q in enumerate(C.PLAN_QUARTERS)])
    return {"tool_groups": tg, "routes": routes, "installed_tools": installed,
            "tool_arrivals": arrivals, "qualification": qual, "plan_2027": plan}


def forecast_history():
    """Requested vs actual WSPW by quarter. Actual follows each program's ramp to its
    Q3 2026 level; requests are inflated or deflated by a program-specific bias."""
    rows = []
    qs = C.HISTORY_QUARTERS
    for site, progs in C.ACTUAL_WSPW_2026Q3.items():
        for p, final in progs.items():
            start = qs.index(C.PROGRAM_START[p])
            n = len(qs) - start
            for i, q in enumerate(qs):
                if i < start:
                    continue
                k = i - start
                if p == "DRAM_A":
                    level = final * (1.25 - 0.25 * k / max(n - 1, 1))      # winding down
                elif p in ("DRAM_B", "HBM_TV"):
                    level = final * (0.35 + 0.65 * k / max(n - 1, 1))      # new-program ramp
                elif p == "NAND_C":
                    level = final * (0.75 + 0.25 * k / max(n - 1, 1))
                else:
                    level = final * RNG.uniform(0.9, 1.05)
                actual = final if q == qs[-1] else level * RNG.normal(1, 0.04)
                requested = actual / (C.TRUE_REQUEST_BIAS[p] * RNG.normal(1, 0.035))
                rows.append({"quarter": q, "site": site, "program": p,
                             "requested_wspw": int(round(requested, -1)), "actual_wspw": int(round(actual))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ documents
def write_docs(hist, tg_perf):
    name = {p: v["name"] for p, v in C.PROGRAMS.items()}
    euv = tg_perf[(tg_perf["site"] == "FAB_A") & (tg_perf["tool_group"] == "EUV")].iloc[0]
    reviews = {
        "2025Q2": ["DRAM Node B ran 16% below its requested starts for the third quarter in a row; program "
                   "team asked to keep the request unchanged to protect EUV access.",
                   "Fab B etch loading reached 82% as NAND Gen C ramps. Flagged as a watch item.",
                   "Decision: keep loading ceiling at 85% for single-tool groups (EUV) because queue time "
                   "rises sharply above that level."],
        "2025Q4": ["HBM test vehicle started in Q3 2025; actual starts were about 20% below request.",
                   "Approved purchase of a second EUV scanner for Fab A, expected production release 2027Q4.",
                   "Short-loop experiments ran about 10% above request; teams submit them late in the quarter."],
        "2026Q1": ["Fab B deposition above 84%; three additional CVD/ALD chambers planned for 2027Q3.",
                   "Engineering holds account for roughly a third of DRAM Node B cycle time. Action: daily "
                   "hold review with Process Integration.",
                   "Fab A etch expansion (3 chambers) planned for 2027Q2."],
        "2026Q3": [f"EUV at Fab A averaged {euv['loading_pct']:.0f}% loading in Q3 with an average queue of "
                   f"{euv['avg_queue_h']:.0f} hours per pass, the largest single queue in the fab.",
                   "NAND Gen C requests for 2027 grow to 480 WSPW at Fab B by Q4. Fab B etch cannot absorb "
                   "this without help; options are a load move to Fab A (NAND Gen C is qualified there) or "
                   "additional etch chambers (lead time about 3 quarters).",
                   "Finance asked for a bias-adjusted view of 2027 requests before approving new tools."],
    }
    for q, bullets in reviews.items():
        h = hist[hist["quarter"] == q]
        tbl = "\n".join(f"- {r.site} {name[r.program]}: requested {r.requested_wspw} WSPW, actual {r.actual_wspw}"
                        for r in h.itertuples())
        text = (f"TD Quarterly Planning Review - {q}\nAttendees: TD IE Strategy, Process Integration, "
                f"Engineering, Operations, Finance\n\nWafer starts this quarter:\n{tbl}\n\nDiscussion and decisions:\n"
                + "\n".join(f"- {b}" for b in bullets) + "\n")
        (DOCS / "planning_reviews" / f"planning_review_{q}.txt").write_text(text)

    notices = [
        ("PCN-2025-008", "2025Q3", "DRAM Node B", "Adds one EUV contact layer, replacing a DUV double-patterning "
         "pair. Net litho passes unchanged at DUV minus two, EUV plus one. One extra etch and one extra clean."),
        ("PCN-2025-019", "2026Q1", "3D NAND Gen C", "Channel-hole etch recipe lengthened for higher aspect ratio; "
         "etch time per pass up about 10% on the high-aspect-ratio steps."),
        ("PCN-2026-004", "2026Q2", "HBM test vehicle", "TSV module adds two etch, two PVD and one CMP step."),
        ("PCN-2026-017", "2026Q3", "Short-loop experiments", "New gate-stack short loop adds furnace anneal; "
         "about one extra furnace pass per short-loop wafer."),
    ]
    for pid, q, prog, body in notices:
        text = (f"Process Change Notice {pid}\nOwner: Process Integration\nProgram: {prog}\nEffective: {q}\n\n"
                f"Change:\n{body}\n\nCapacity review: TD IE Strategy to assess tool-group impact before release.\n")
        (DOCS / "route_changes" / f"{pid}.txt").write_text(text)

    notes = {
        ("FAB_A", "EUV"): "Single EUV scanner. Engineering time about 15% for resist and dose experiments. "
                          "Second scanner approved, production release 2027Q4. Any slip moves the release to 2028Q1.",
        ("FAB_A", "ETCH"): "Three additional etch chambers planned for 2027Q2. NAND Gen C high-aspect-ratio etch "
                           "is qualified on Fab A chambers.",
        ("FAB_B", "ETCH"): "Thirteen chambers. NAND Gen C high-aspect-ratio etch takes about 2.2x the time of a "
                           "standard etch pass. New chambers have a lead time of about three quarters.",
        ("FAB_B", "DEP"): "Three CVD/ALD chambers arriving 2027Q3. NAND Gen C stack deposition is the main load.",
        ("FAB_A", "FURN"): "Batch furnaces run up to 4 lots; development lots average 3 lots per run. "
                           "Waiting to fill batches adds queue time.",
        ("FAB_B", "IMPL"): "Two implanters; the second is often in engineering use for new species qualification.",
    }
    perf = tg_perf.set_index(["site", "tool_group"])
    for (site, tg), body in notes.items():
        r = perf.loc[(site, tg)]
        text = (f"Tool group note - {C.SITES[site]} - {C.TOOL_GROUPS[tg]['name']}\nInstalled tools: {r['tools']}\n"
                f"Q3 2026: productive {r['productive_pct']}%, engineering {r['engineering_pct']}%, "
                f"unscheduled down {r['unsched_down_pct']}%, average queue {r['avg_queue_h']} h per lot step.\n\n{body}\n")
        (DOCS / "tool_notes" / f"tool_note_{site}_{tg}.txt").write_text(text)

    (DOCS / "guidelines" / "td_capacity_planning_guidelines.txt").write_text(
        "TD Capacity Planning Guidelines\n\n"
        "1. Loading = required tool hours / available production hours (calendar hours minus scheduled "
        "downtime, unscheduled downtime and engineering time).\n"
        "2. Plan tool groups at or below 85% loading. Above 85% queue time grows quickly; above 100% the "
        "plan cannot be run.\n"
        "3. Program teams submit wafer-start requests each quarter. IE Strategy compares requests with "
        "historical actuals before committing capacity.\n"
        "4. Moving a program to another site requires that the program is qualified at that site. "
        "New qualifications take about two quarters.\n"
        "5. Options for a constraint, in order: re-balance load across sites, adjust timing of experiments, "
        "recover availability, then add tools.\n"
        "6. Hot lots (about 10% of lots) get priority at every step; keep the share low so normal lots do "
        "not stall.\n")


def main():
    reset()
    print("Running fab simulation...")
    sim = simulate_all()
    tables = config_tables()
    hist = forecast_history()
    tables.update({
        "wafer_start_history": hist,
        "sim_lots": pd.DataFrame(sim["lots"]),
        "sim_tool_group_q3_2026": pd.DataFrame(sim["tool_groups"]),
        "sim_fab_weekly": pd.DataFrame(sim["weekly"]),
        "sim_tool_group_weekly": pd.DataFrame(sim["weekly_tg"]),
    })
    for name, df in tables.items():
        df.to_csv(TABLES / f"{name}.csv", index=False)
    write_docs(hist, tables["sim_tool_group_q3_2026"])
    with sqlite3.connect(DB) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
    n_docs = sum(1 for _ in DOCS.rglob("*.txt"))
    print(f"Wrote {len(tables)} tables, {n_docs} documents and {DB.name}.")


if __name__ == "__main__":
    main()
