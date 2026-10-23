"""
Synthetic data generator for the Operational Intelligence Dashboard.

Simulates a 3-line discrete assembly plant for calendar year 2024:
  * Schedule: Monday-Friday, 3 shifts x 8 h per line (120 scheduled hours/week).
  * Each shift runs 4 SKUs back to back in 2-hour blocks, so planned hours add
    up to 8 per shift (the old version gave every SKU the full 8 hours).
  * Machine downtime events are generated first; line downtime is derived from
    them (a stop on any machine stops the serial line), so the production table
    and the downtime log always agree.
  * A Q1 2025 weekly demand forecast drives the capacity-planning tab.

Planted problems the dashboard should surface:
  * M11 (Line 1) is unreliable: ~3x the stop rate and ~3x longer repairs.
  * Supplier SUP_B causes extra scrap on Line 2.
  * Night shift runs ~3 points slower with +1 point scrap.

All data is synthetic. Run from the repo root or the code/ folder:
    python code/data_generation.py
"""
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

DATE_START = "2024-01-01"
DATE_END = "2024-12-31"

LINES = ["L1", "L2", "L3"]
SHIFTS = [
    {"id": "MORNING", "start_hour": 6},
    {"id": "EVENING", "start_hour": 14},
    {"id": "NIGHT", "start_hour": 22},
]
SHIFT_MIN = 480
SKU_BLOCKS = 4                      # SKUs per shift
BLOCK_MIN = SHIFT_MIN // SKU_BLOCKS  # 120 minutes each
MACHINES_PER_LINE = 15
SKUS = [f"SKU{i:03d}" for i in range(1, 31)]
SKUS_PER_LINE = 8
SUPPLIERS = ["SUP_A", "SUP_B", "SUP_C", "SUP_D"]
SUPPLIER_P = [0.3, 0.3, 0.2, 0.2]
OPERATORS = ["Alex", "Jordan", "Riley", "Sam", "Taylor", "Morgan"]

PROBLEM_MACHINE = "M11"
PROBLEM_SUPPLIER = "SUP_B"
PROBLEM_SUPPLIER_LINE = "L2"

PLANNING_OEE = 0.85   # rate the scheduler plans to (sets planned_units)
FAILURE_CAUSES = ["MECH_FAILURE", "SENSOR_FAULT"]

# Q1 2025 demand relative to each line's 2024 Q4 output (line = 100% loaded today)
DEMAND_FACTOR = {"L1": 0.93, "L2": 1.10, "L3": 0.78}
FORECAST_WEEKS = 13


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def reset_dirs():
    """Delete old generated data so no stale files leak into the document index."""
    data_root = project_root() / "data" / "manrag"
    for sub in ["structured", "unstructured", "chroma_store"]:
        shutil.rmtree(data_root / sub, ignore_errors=True)
    db = data_root / "manrag_factory.db"
    if db.exists():
        db.unlink()
    structured = data_root / "structured"
    unstructured = data_root / "unstructured"
    structured.mkdir(parents=True, exist_ok=True)
    for sub in ["maintenance_logs", "shift_reports", "sop_docs", "weekly_summaries", "machine_notes"]:
        (unstructured / sub).mkdir(parents=True, exist_ok=True)
    return structured, unstructured


def workdays():
    days = pd.date_range(DATE_START, DATE_END, freq="D")
    return [d for d in days if d.weekday() < 5]


# ---------------------------------------------------------------- dimensions
def make_dimensions():
    dim_line = pd.DataFrame({"line_id": LINES,
                             "description": [f"Assembly Line {i + 1}" for i in range(len(LINES))]})
    dim_shift = pd.DataFrame({"shift_id": [s["id"] for s in SHIFTS],
                              "start_time": ["06:00", "14:00", "22:00"],
                              "end_time": ["14:00", "22:00", "06:00"]})
    rows, idx = [], 1
    for line in LINES:
        for _ in range(MACHINES_PER_LINE):
            rows.append({"machine_id": f"M{idx:02d}", "line_id": line,
                         "machine_type": RNG.choice(["Press", "Cutter", "Assembler", "Conveyor"]),
                         "commissioned_date": "2020-01-01"})
            idx += 1
    dim_machine = pd.DataFrame(rows)
    dim_sku = pd.DataFrame({
        "sku_id": SKUS,
        "name": [f"Product {i}" for i in range(1, len(SKUS) + 1)],
        "category": [RNG.choice(["Valve", "Controller", "Housing", "Bracket"]) for _ in SKUS],
        "unit_cost": RNG.uniform(10, 200, size=len(SKUS)).round(2),
        "ideal_cycle_time_sec": RNG.uniform(2.5, 8.0, size=len(SKUS)).round(1),
    })
    shuffled = RNG.permutation(SKUS).tolist()
    line_sku = pd.DataFrame([{"line_id": line, "sku_id": sku}
                             for i, line in enumerate(LINES)
                             for sku in shuffled[i * SKUS_PER_LINE:(i + 1) * SKUS_PER_LINE]])
    return dim_line, dim_shift, dim_machine, dim_sku, line_sku


# ---------------------------------------------------------------- downtime
def make_downtime_events(dim_machine, days):
    rows = []
    n_shifts = len(days) * len(SHIFTS)
    for _, m in dim_machine.iterrows():
        problem = m["machine_id"] == PROBLEM_MACHINE
        stops_per_month = 14 if problem else 4
        n_events = RNG.poisson(stops_per_month * 12)
        slots = RNG.choice(n_shifts, size=min(n_events, n_shifts), replace=False)
        for slot in slots:
            day = days[slot // len(SHIFTS)]
            shift = SHIFTS[slot % len(SHIFTS)]
            shift_start = datetime.combine(day.date(), datetime.min.time()) + timedelta(hours=shift["start_hour"])
            offset = int(RNG.integers(0, SHIFT_MIN - 5))
            if problem:
                duration = int(max(15, RNG.normal(100, 40)))
                cause = RNG.choice(["MECH_FAILURE", "MATERIAL_JAM", "SENSOR_FAULT", "CHANGEOVER_ISSUE"],
                                   p=[0.6, 0.2, 0.15, 0.05])
            else:
                duration = int(max(5, RNG.normal(35, 22)))
                cause = RNG.choice(["MECH_FAILURE", "MATERIAL_JAM", "SENSOR_FAULT", "CHANGEOVER_ISSUE"],
                                   p=[0.35, 0.3, 0.2, 0.15])
            duration = min(duration, SHIFT_MIN - offset)  # keep the event inside its shift
            start = shift_start + timedelta(minutes=offset)
            severity = "LOW" if duration < 20 else ("MEDIUM" if duration < 60 else "HIGH")
            rows.append({
                "machine_id": m["machine_id"], "line_id": m["line_id"],
                "shift_date": day.date().isoformat(), "shift_id": shift["id"],
                "start_time": start.isoformat(), "end_time": (start + timedelta(minutes=duration)).isoformat(),
                "start_offset_min": offset, "duration_min": duration,
                "cause_code": cause, "severity": severity,
                "operator_name": RNG.choice(OPERATORS),
            })
    df = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    df.insert(0, "event_id", range(1, len(df) + 1))
    return df


def line_stop_intervals(events):
    """Union of machine stop intervals per (line, date, shift), in minutes from shift start."""
    out = {}
    for key, grp in events.groupby(["line_id", "shift_date", "shift_id"]):
        ivs = sorted(zip(grp["start_offset_min"], grp["start_offset_min"] + grp["duration_min"]))
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[key] = merged
    return out


def overlap(intervals, lo, hi):
    return sum(max(0, min(e, hi) - max(s, lo)) for s, e in intervals)


# ---------------------------------------------------------------- production
def make_production(days, dim_sku, line_sku, stops):
    ct = dict(zip(dim_sku["sku_id"], dim_sku["ideal_cycle_time_sec"]))
    skus_by_line = line_sku.groupby("line_id")["sku_id"].apply(list).to_dict()
    rows = []
    for day in days:
        d = day.date().isoformat()
        for line in LINES:
            for shift in SHIFTS:
                intervals = stops.get((line, d, shift["id"]), [])
                for seq, sku in enumerate(RNG.choice(skus_by_line[line], size=SKU_BLOCKS, replace=False), start=1):
                    lo, hi = (seq - 1) * BLOCK_MIN, seq * BLOCK_MIN
                    dt_min = overlap(intervals, lo, hi)
                    runtime_h = (BLOCK_MIN - dt_min) / 60
                    perf = RNG.uniform(0.86, 0.97) - (0.03 if shift["id"] == "NIGHT" else 0)
                    produced = int(runtime_h * 3600 / ct[sku] * perf)
                    supplier = RNG.choice(SUPPLIERS, p=SUPPLIER_P)
                    scrap_rate = 0.02 + (0.01 if shift["id"] == "NIGHT" else 0)
                    if supplier == PROBLEM_SUPPLIER and line == PROBLEM_SUPPLIER_LINE:
                        scrap_rate += 0.03
                    scrap = int(RNG.binomial(produced, scrap_rate)) if produced > 0 else 0
                    rows.append({
                        "date": d, "line_id": line, "shift_id": shift["id"], "run_seq": seq,
                        "sku_id": sku, "supplier_id": supplier,
                        "planned_hours": BLOCK_MIN / 60, "runtime_hours": round(runtime_h, 3),
                        "downtime_minutes": int(dt_min),
                        "planned_units": int(BLOCK_MIN * 60 / ct[sku] * PLANNING_OEE),
                        "produced_units": produced, "scrap_units": scrap,
                        "standard_cycle_time_sec": ct[sku],
                    })
    return pd.DataFrame(rows)


def make_demand_forecast(prod):
    """Weekly Q1 2025 demand (good units) per line and SKU."""
    p = prod.copy()
    p["date"] = pd.to_datetime(p["date"])
    q4 = p[p["date"] >= "2024-10-01"]
    weeks_q4 = q4["date"].dt.to_period("W").nunique()
    base = (q4.assign(good=q4["produced_units"] - q4["scrap_units"])
              .groupby(["line_id", "sku_id"])["good"].sum() / weeks_q4).reset_index(name="base")
    first_monday = pd.Timestamp("2025-01-06")
    rows = []
    for w in range(FORECAST_WEEKS):
        week = first_monday + pd.Timedelta(weeks=w)
        for _, r in base.iterrows():
            factor = DEMAND_FACTOR[r["line_id"]] * (1 + RNG.normal(0, 0.04))
            rows.append({"week_start": week.date().isoformat(), "line_id": r["line_id"],
                         "sku_id": r["sku_id"], "demand_units": int(r["base"] * factor)})
    return pd.DataFrame(rows)


def make_maintenance_tickets(events):
    """One ticket per HIGH-severity failure, closed 0.5-5 days later (open if very recent)."""
    fails = events[(events["severity"] == "HIGH") & (events["cause_code"].isin(FAILURE_CAUSES))]
    rows = []
    for i, (_, e) in enumerate(fails.iterrows()):
        opened = datetime.fromisoformat(e["start_time"])
        closed = opened + timedelta(hours=float(RNG.uniform(12, 120)))
        status = "OPEN" if closed > datetime(2024, 12, 31, 23, 59) else "CLOSED"
        if e["machine_id"] == PROBLEM_MACHINE:
            issue, root = "Vibration", RNG.choice(["Misalignment", "Bearing Wear"], p=[0.8, 0.2])
        else:
            issue = "Sensor Errors" if e["cause_code"] == "SENSOR_FAULT" else RNG.choice(["Overheating", "Vibration", "Frequent Jams"])
            root = RNG.choice(["Bearing Wear", "Calibration Drift", "Material Quality", "Misalignment"])
        action = {
            "Misalignment": "Realigned mechanical assembly per SOP-MECH-ALIGN-001 and tightened fixtures.",
            "Bearing Wear": "Replaced worn bearings and recalibrated the station.",
            "Calibration Drift": "Adjusted sensor thresholds, replaced faulty sensor, and verified readings.",
            "Material Quality": "Escalated material issue to supplier and updated incoming inspection plan per SOP-MAT-QUAL-002.",
        }[root]
        rows.append({"ticket_id": 1001 + i, "machine_id": e["machine_id"], "line_id": e["line_id"],
                     "event_id": e["event_id"], "opened_at": opened.isoformat(),
                     "closed_at": closed.isoformat() if status == "CLOSED" else "",
                     "issue_type": issue, "root_cause": root, "action_taken": action, "status": status})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- text documents
def write_maintenance_logs(udir, tickets, events):
    """Ticket write-ups: all M11 tickets in Q4 plus up to 2 per other machine."""
    picks = pd.concat([
        tickets[(tickets["machine_id"] == PROBLEM_MACHINE) & (tickets["opened_at"] >= "2024-10-01")],
        tickets[tickets["machine_id"] != PROBLEM_MACHINE].groupby("machine_id").head(2),
    ])
    stops_per_machine = events.groupby("machine_id").size()
    for _, t in picks.iterrows():
        m = t["machine_id"]
        text = f"""Maintenance Ticket {t['ticket_id']}
Machine: {m} (Line {t['line_id']})
Opened: {t['opened_at'][:16].replace('T', ' ')}
Status: {t['status']}
Issue: {t['issue_type']}
Root cause: {t['root_cause']}

Summary:
Technician responded to a high-severity stop on {m}. Inspection pointed to {t['root_cause'].lower()}.
{m} logged {stops_per_machine.get(m, 0)} stops during 2024.

Corrective action:
{t['action_taken']}

Follow-up:
- Monitor {m} for noise, vibration, minor jams or sensor irregularities over the next two weeks.
- Escalate if unplanned downtime exceeds the normal range for this asset.
"""
        (udir / "maintenance_logs" / f"ticket_{t['ticket_id']}_{m}.txt").write_text(text)


def write_shift_reports(udir, prod, events):
    p = prod.copy()
    shift = (p.groupby(["date", "line_id", "shift_id"])
               .agg(produced=("produced_units", "sum"), scrap=("scrap_units", "sum"),
                    dt=("downtime_minutes", "sum")).reset_index())
    shift["scrap_rate"] = shift["scrap"] / shift["produced"]
    high_scrap = shift.sort_values("scrap_rate", ascending=False).groupby("line_id").head(20)
    high_dt = shift.sort_values("dt", ascending=False).groupby("line_id").head(10)
    for _, s in high_scrap.iterrows():
        blocks = p[(p["date"] == s["date"]) & (p["line_id"] == s["line_id"]) & (p["shift_id"] == s["shift_id"])].copy()
        blocks["rate"] = blocks["scrap_units"] / blocks["produced_units"]
        worst = blocks.sort_values("rate", ascending=False).iloc[0]
        text = f"""Shift Report - Line {s['line_id']} - {s['date']} - {s['shift_id']} shift

Quality:
Shift scrap rate was {s['scrap_rate'] * 100:.1f}% ({int(s['scrap'])} of {int(s['produced'])} units).
Worst run: {worst['sku_id']} with material from {worst['supplier_id']} at {worst['rate'] * 100:.1f}% scrap.
Operators reported dimensional defects and extra checks at the inspection station.

Actions taken:
- Held suspect material from {worst['supplier_id']} and notified quality per SOP-MAT-QUAL-002.
- Increased inspection frequency for the rest of the shift.

Notes for next shift:
- Watch scrap on {worst['sku_id']}; escalate to purchasing if the issue repeats with {worst['supplier_id']}.
"""
        (udir / "shift_reports" / f"shift_{s['line_id']}_{s['date']}_{s['shift_id']}_quality.txt").write_text(text)
    for _, s in high_dt.iterrows():
        ev = events[(events["shift_date"] == s["date"]) & (events["line_id"] == s["line_id"]) &
                    (events["shift_id"] == s["shift_id"])].sort_values("duration_min", ascending=False)
        lines = "\n".join(f"- {r.machine_id}: {r.cause_code} for {r.duration_min} min" for r in ev.itertuples())
        text = f"""Shift Report - Line {s['line_id']} - {s['date']} - {s['shift_id']} shift

Downtime:
The line lost {int(s['dt'])} minutes to stops this shift.
{lines}

Actions taken:
- Maintenance responded to the longest stop; machine returned to service after inspection.

Notes for next shift:
- Run the start-of-shift vibration and noise check on the machines above.
"""
        (udir / "shift_reports" / f"shift_{s['line_id']}_{s['date']}_{s['shift_id']}_downtime.txt").write_text(text)


def write_weekly_summaries(udir, prod, events):
    p = prod.copy()
    p["week"] = pd.to_datetime(p["date"]).dt.to_period("W").apply(lambda r: r.start_time.date())
    e = events.copy()
    e["week"] = pd.to_datetime(e["shift_date"]).dt.to_period("W").apply(lambda r: r.start_time.date())
    wk = (p.groupby(["week", "line_id"])
           .agg(produced=("produced_units", "sum"), scrap=("scrap_units", "sum"),
                planned=("planned_units", "sum"), dt=("downtime_minutes", "sum")).reset_index())
    for _, w in wk.iterrows():
        ev = e[(e["week"] == w["week"]) & (e["line_id"] == w["line_id"])]
        causes = ev.groupby("cause_code")["duration_min"].sum().sort_values(ascending=False).head(3)
        machines = ev.groupby("machine_id")["duration_min"].sum().sort_values(ascending=False).head(2)
        good = w["produced"] - w["scrap"]
        text = f"""Weekly Operations Summary - Line {w['line_id']} - week starting {w['week']}

Production: {good} good units against a plan of {w['planned']} ({good / w['planned'] * 100:.0f}% attainment).
Scrap: {w['scrap']} units ({w['scrap'] / w['produced'] * 100:.2f}%).
Downtime: {w['dt']} line minutes.
Top downtime causes: {', '.join(f'{c} ({m} min)' for c, m in causes.items()) or 'none'}.
Machines with most downtime: {', '.join(f'{m} ({v} min)' for m, v in machines.items()) or 'none'}.

Used in the weekly operations huddle to review performance and agree corrective actions.
"""
        (udir / "weekly_summaries" / f"weekly_summary_{w['line_id']}_{w['week']}.txt").write_text(text)


def write_machine_notes(udir, dim_machine):
    for _, m in dim_machine.iterrows():
        if m["machine_id"] == PROBLEM_MACHINE:
            body = """Operator Notes (High Attention Asset):
This machine has a known history of vibration and alignment issues, especially during extended runs
and right after mechanical changeovers.

Common observations:
- Increased noise and vibration at higher speeds.
- Minor jams at the infeed that escalate if not cleared promptly.
- Scrap rises (surface defects, dimensions out of spec) when alignment drifts.

Operating guidelines:
- Do a vibration and noise check at the start of each shift.
- Stop and inspect if minor stoppages occur more than 3 times in a 2-hour window.
- Log every abnormal event in the shift report, even if resolved quickly.

Escalation:
- If unplanned downtime exceeds 60 minutes in a shift, notify maintenance immediately.
- If alignment issues repeat in the same week, request a full mechanical inspection per SOP-MECH-ALIGN-001."""
        else:
            body = """Operator Notes:
This asset generally runs reliably under standard conditions.
Most issues are routine wear, minor jams or sensor calibration.

Operating guidelines:
- Follow the standard startup checklist, including safety and interlock checks.
- Keep the work area clear of excess material, tools and debris.
- Record recurring stoppages, defects or unusual noise in the shift report.

Escalation:
- If the same stoppage reason appears more than twice in a shift, log a maintenance request.
- If quality concerns are observed, hold suspect pallets and notify quality."""
        text = f"Machine: {m['machine_id']}\nLine: {m['line_id']}\nType: {m['machine_type']}\n\n{body}\n"
        (udir / "machine_notes" / f"machine_notes_{m['machine_id']}.txt").write_text(text)


SOP_ALIGNMENT = """Document Title: SOP - Mechanical Alignment for M-Series Machines
Document ID: SOP-MECH-ALIGN-001
Revision: 2.0
Owner: Maintenance Engineering
Effective Date: 2023-09-01

1. Purpose
Standard method for inspecting and correcting mechanical alignment on M-series machines
to minimize vibration, scrap and unplanned downtime.

2. Scope
All M-series machines on Lines L1, L2 and L3.

3. Responsibilities
- Maintenance Technicians: perform alignment inspections and adjustments.
- Line Supervisors: schedule planned alignment checks and verify completion.
- Operators: report abnormal noise, vibration or recurring jams.

4. Required PPE
Safety glasses, safety shoes, cut-resistant gloves, hearing protection in high-noise areas.

5. Procedure
5.1 Lockout/Tagout: stop the machine, apply site LOTO, verify zero energy.
5.2 Inspection: check belts, pulleys, couplings and mounts for wear or misalignment; check for loose fasteners or cracked mounts.
5.3 Alignment: verify shaft and belt alignment with approved tools; adjust to tolerance; torque mounting bolts to spec.
5.4 Verification: remove LOTO, run a short test batch, monitor vibration, noise and scrap. Escalate to the Maintenance Supervisor if abnormal conditions persist.

6. Documentation
Record alignment activities, measurements and replaced components in the maintenance system.
"""

SOP_MATERIAL = """Document Title: SOP - Handling Material Quality Issues
Document ID: SOP-MAT-QUAL-002
Revision: 1.3
Owner: Quality Assurance
Effective Date: 2023-06-15

1. Purpose
Standard response when material quality issues are suspected or confirmed, to protect product quality and prevent excess scrap.

2. Scope
All incoming raw materials used on Lines L1, L2 and L3.

3. Triggers for material quality review
- Sudden increase in scrap linked to a specific supplier or batch.
- Repeated jams or feeding issues with the same lot.
- Visual defects (surface damage, wrong dimensions, contamination).

4. Procedure
4.1 Immediate actions: stop the line if there is a safety or severe quality risk; isolate the suspect batch and tag it ON HOLD; record batch ID, supplier and time in the shift report.
4.2 Inspection: inspect and measure suspect material against approved specs and reference samples; document findings.
4.3 Disposition: if in spec, release for a controlled trial run; if out of spec, hold the batch and escalate to Quality and Purchasing; consider an alternate approved supplier.

5. Documentation
Log all material issues in the quality system with batch ID and supplier. Supplier scorecards are updated monthly.
"""


def main():
    structured, unstructured = reset_dirs()
    days = workdays()
    dim_line, dim_shift, dim_machine, dim_sku, line_sku = make_dimensions()
    events = make_downtime_events(dim_machine, days)
    stops = line_stop_intervals(events)
    prod = make_production(days, dim_sku, line_sku, stops)
    demand = make_demand_forecast(prod)
    tickets = make_maintenance_tickets(events)

    tables = {
        "dim_line": dim_line, "dim_shift": dim_shift, "dim_machine": dim_machine,
        "dim_sku": dim_sku, "dim_line_sku": line_sku,
        "production_daily": prod, "downtime_events": events.drop(columns="start_offset_min"),
        "maintenance_tickets": tickets, "demand_forecast": demand,
    }
    for name, df in tables.items():
        df.to_csv(structured / f"{name}.csv", index=False)

    write_maintenance_logs(unstructured, tickets, events)
    write_shift_reports(unstructured, prod, events)
    write_weekly_summaries(unstructured, prod, events)
    write_machine_notes(unstructured, dim_machine)
    (unstructured / "sop_docs" / "SOP_alignment_M_series.txt").write_text(SOP_ALIGNMENT)
    (unstructured / "sop_docs" / "SOP_material_quality.txt").write_text(SOP_MATERIAL)

    n_docs = sum(1 for _ in unstructured.rglob("*.txt"))
    print(f"Synthetic plant data generated: {len(prod)} production rows, {len(events)} downtime events, "
          f"{len(tickets)} tickets, {n_docs} text documents.")


if __name__ == "__main__":
    main()
