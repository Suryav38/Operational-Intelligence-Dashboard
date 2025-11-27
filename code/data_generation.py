import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# -----------------------------
# Global config & random seeds
# -----------------------------
RNG = np.random.default_rng(42)
random.seed(42)

# Full year of data
DATE_START = "2024-01-01"
DATE_END = "2024-12-31" 

LINES = ["L1", "L2", "L3"]
SHIFTS = [
    {"id": "MORNING", "start": "06:00", "end": "14:00"},
    {"id": "EVENING", "start": "14:00", "end": "22:00"},
    {"id": "NIGHT", "start": "22:00", "end": "06:00"},
]

MACHINES_PER_LINE = 15
SKUS = [f"SKU{i:03d}" for i in range(1, 31)]  # SKU001 .. SKU030

# Problem patterns
PROBLEM_MACHINE = "M11"  
PROBLEM_SUPPLIER = "SUP_B" 
SUPPLIERS = ["SUP_A", "SUP_B", "SUP_C", "SUP_D"]


def get_project_root() -> Path:
    # script is in project_root/code/, so root is parent of parent
    return Path(__file__).resolve().parents[1]


def ensure_dirs():
    root = get_project_root()
    data_root = root / "data" / "manrag"
    structured = data_root / "structured"
    unstructured = data_root / "unstructured"
    structured.mkdir(parents=True, exist_ok=True)
    (unstructured / "maintenance_logs").mkdir(parents=True, exist_ok=True)
    (unstructured / "shift_reports").mkdir(parents=True, exist_ok=True)
    (unstructured / "sop_docs").mkdir(parents=True, exist_ok=True)
    (unstructured / "weekly_summaries").mkdir(parents=True, exist_ok=True)
    (unstructured / "machine_notes").mkdir(parents=True, exist_ok=True)
    return structured, unstructured


# -----------------------------
# Dimension tables
# -----------------------------
def generate_dim_line():
    return pd.DataFrame(
        {
            "line_id": LINES,
            "description": [f"Assembly Line {i+1}" for i in range(len(LINES))],
        }
    )


def generate_dim_shift():
    return pd.DataFrame(
        {
            "shift_id": [s["id"] for s in SHIFTS],
            "start_time": [s["start"] for s in SHIFTS],
            "end_time": [s["end"] for s in SHIFTS],
        }
    )


def generate_dim_machine():
    rows = []
    machine_idx = 1
    for line in LINES:
        for _ in range(MACHINES_PER_LINE):
            rows.append(
                {
                    "machine_id": f"M{machine_idx:02d}",
                    "line_id": line,
                    "machine_type": RNG.choice(
                        ["Press", "Cutter", "Assembler", "Conveyor"]
                    ),
                    "commissioned_date": "2020-01-01",
                }
            )
            machine_idx += 1
    return pd.DataFrame(rows)


def generate_dim_sku():
    categories = ["Valve", "Controller", "Housing", "Bracket"]
    # NEW: Added ideal_cycle_time_sec to calculate Performance
    return pd.DataFrame(
        {
            "sku_id": SKUS,
            "name": [f"Product {i}" for i in range(1, len(SKUS) + 1)],
            "category": [RNG.choice(categories) for _ in SKUS],
            "unit_cost": RNG.uniform(10, 200, size=len(SKUS)).round(2),
            "ideal_cycle_time_sec": RNG.uniform(2.5, 8.0, size=len(SKUS)).round(1) 
        }
    )


# -----------------------------
# Fact tables
# -----------------------------
def generate_production_daily(dim_machine: pd.DataFrame, dim_sku: pd.DataFrame):
    """
    UPDATED: Calculates produced units based on runtime and ideal cycle time.
    Performance is now a derived result of (Actual / Ideal).
    """
    dates = pd.date_range(DATE_START, DATE_END, freq="D")
    rows = []

    # Create a lookup for cycle times
    sku_cycle_map = dict(zip(dim_sku["sku_id"], dim_sku["ideal_cycle_time_sec"]))

    # Assign SKUs to lines
    line_sku_map = {
        line: RNG.choice(SKUS, size=8, replace=False).tolist() for line in LINES
    }

    for date in dates:
        for line in LINES:
            for shift in SHIFTS:
                skus_for_line = line_sku_map[line]
                sku_sample = RNG.choice(skus_for_line, size=4, replace=False)
                
                for sku in sku_sample:
                    planned_hours = 8.0
                    
                    # 1. Calculate Downtime
                    downtime_minutes = max(0, int(RNG.normal(loc=25, scale=12)))
                    
                    # Problem Machine / Line logic simulation (indirectly affecting Line 2)
                    if line == "L2" and RNG.random() > 0.8:
                         downtime_minutes += int(RNG.uniform(10, 40))

                    runtime_hours = max(0.0, planned_hours - (downtime_minutes / 60.0))
                    runtime_seconds = runtime_hours * 3600
                    
                    # 2. Calculate Theoretical Max Production (100% Performance)
                    ideal_cycle = sku_cycle_map[sku]
                    max_possible_units = int(runtime_seconds / ideal_cycle) if ideal_cycle > 0 else 0
                    
                    # 3. Apply Performance Efficiency Factor
                    # Normal lines run between 85% and 98% speed efficiency
                    # Night shift is slightly slower (tired operators)
                    perf_efficiency = RNG.uniform(0.85, 0.98)
                    if shift["id"] == "NIGHT":
                        perf_efficiency -= 0.03
                    
                    produced_units = int(max_possible_units * perf_efficiency)

                    # 4. Calculate Scrap (Quality)
                    scrap_rate = 0.02
                    if shift["id"] == "NIGHT": scrap_rate += 0.01
                    
                    supplier = RNG.choice(SUPPLIERS, p=[0.3, 0.3, 0.2, 0.2])
                    if supplier == PROBLEM_SUPPLIER and line == "L2":
                        scrap_rate += 0.03
                    
                    # Add noise
                    scrap_units = int(max(0, (produced_units * scrap_rate) + RNG.normal(0, 5)))
                    
                    # Ensure produced units >= scrap units
                    if scrap_units > produced_units:
                        scrap_units = produced_units

                    rows.append(
                        {
                            "date": date.date().isoformat(),
                            "line_id": line,
                            "shift_id": shift["id"],
                            "sku_id": sku,
                            "supplier_id": supplier,
                            "produced_units": produced_units,
                            "scrap_units": scrap_units,
                            "planned_hours": planned_hours,
                            "runtime_hours": round(runtime_hours, 2),
                            "downtime_minutes": downtime_minutes,
                            # We store standard cycle time so dashboard can calculate Performance later
                            "standard_cycle_time_sec": ideal_cycle 
                        }
                    )

    return pd.DataFrame(rows)


def generate_downtime_events(dim_machine: pd.DataFrame):
    # (Same as before)
    dates = pd.date_range(DATE_START, DATE_END, freq="D")
    rows = []
    event_id = 1
    total_days = (dates[-1] - dates[0]).days + 1

    for _, m in dim_machine.iterrows():
        machine_id = m["machine_id"]
        line_id = m["line_id"]

        base_lambda_month = 5
        if machine_id == PROBLEM_MACHINE:
            base_lambda_month = 14 

        months = total_days / 30.0
        expected_events = RNG.poisson(base_lambda_month * months)

        if expected_events == 0:
            continue

        event_days = RNG.choice(dates, size=expected_events, replace=False)
        for day in event_days:
            shift = RNG.choice(SHIFTS)
            day_date = pd.to_datetime(day).date()
            start_hour = int(shift["start"].split(":")[0])
            start_time = datetime.combine(day_date, datetime.min.time()) + timedelta(hours=start_hour)
            offset_min = int(RNG.integers(0, 4 * 60))
            start_time = start_time + timedelta(minutes=offset_min)

            if machine_id == PROBLEM_MACHINE:
                duration_min = int(max(10, RNG.normal(loc=100, scale=45)))
            else:
                duration_min = int(max(5, RNG.normal(loc=35, scale=25)))

            end_time = start_time + timedelta(minutes=duration_min)
            cause_code = RNG.choice(
                ["MECH_FAILURE", "MATERIAL_JAM", "SENSOR_FAULT", "CHANGEOVER_ISSUE"],
                p=[0.4, 0.3, 0.2, 0.1],
            )
            severity = RNG.choice(["LOW", "MEDIUM", "HIGH"], p=[0.5, 0.3, 0.2])

            rows.append(
                {
                    "event_id": event_id,
                    "machine_id": machine_id,
                    "line_id": line_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_min": duration_min,
                    "cause_code": cause_code,
                    "severity": severity,
                    "shift_id": shift["id"],
                    "operator_name": RNG.choice(
                        ["Alex", "Jordan", "Riley", "Sam", "Taylor", "Morgan"]
                    ),
                }
            )
            event_id += 1

    return pd.DataFrame(rows)


def generate_maintenance_tickets(downtime_events: pd.DataFrame):
    # (Same as before)
    rows = []
    ticket_id = 1001

    grouped = downtime_events.groupby("machine_id")["duration_min"].sum().reset_index()
    for _, row in grouped.iterrows():
        machine_id = row["machine_id"]
        total_downtime = row["duration_min"]

        if total_downtime > 300 or machine_id == PROBLEM_MACHINE:
            machine_events = downtime_events[
                downtime_events["machine_id"] == machine_id
            ].sort_values("start_time")
            first_event = machine_events.iloc[0]
            last_event = machine_events.iloc[-1]

            issue_type = RNG.choice(["Vibration", "Overheating", "Frequent Jams", "Sensor Errors"])
            if machine_id == PROBLEM_MACHINE: issue_type = "Vibration"
            
            root_cause = RNG.choice(["Misalignment", "Bearing Wear", "Material Quality", "Calibration Drift"])
            if machine_id == PROBLEM_MACHINE: root_cause = "Misalignment"

            action_taken = RNG.choice([
                    "Replaced worn components and recalibrated line.",
                    "Realigned mechanical assembly and tightened fixtures.",
                    "Adjusted sensor thresholds, replaced faulty sensor, and verified readings.",
                    "Escalated recurring material quality issue to supplier and updated incoming inspection plan.",
            ])

            rows.append(
                {
                    "ticket_id": ticket_id,
                    "machine_id": machine_id,
                    "opened_at": first_event["start_time"],
                    "closed_at": last_event["end_time"],
                    "issue_type": issue_type,
                    "root_cause": root_cause,
                    "action_taken": action_taken,
                    "status": "CLOSED",
                }
            )
            ticket_id += 1

    return pd.DataFrame(rows)


# -----------------------------
# Unstructured text generation
# -----------------------------
def write_maintenance_logs(unstructured_dir: Path, maint_tickets: pd.DataFrame):
    # (Same as before)
    logs_dir = unstructured_dir / "maintenance_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for _, row in maint_tickets.iterrows():
        machine_id = row["machine_id"]
        ticket_id = row["ticket_id"]
        opened = str(row["opened_at"])[:10]
        issue_type = row["issue_type"]
        root_cause = row["root_cause"]
        action = row["action_taken"]

        text = f"""Ticket ID: {ticket_id}
Machine: {machine_id}
Opened: {opened}
Issue: {issue_type}

Summary:
Recurring issues were observed on machine {machine_id} across multiple shifts, triggering a maintenance review.
Based on downtime patterns, operator feedback, and inspection, the primary root cause was identified as {root_cause.lower()}.

Corrective Actions:
{action}

Follow-Up:
- Monitor {machine_id} closely over the next two weeks.
- Log any early symptoms such as noise, vibration, minor jams, or sensor irregularities.
- Escalate immediately if unplanned downtime exceeds the normal range for this asset.

This log is intended to support future root-cause analysis by providing historical context on how the issue was previously resolved.
"""
        file_path = logs_dir / f"ticket_{ticket_id}_{machine_id}.txt"
        file_path.write_text(text)


def write_shift_reports(unstructured_dir: Path, production_daily: pd.DataFrame):
    # (Same as before)
    reports_dir = unstructured_dir / "shift_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = production_daily.copy()
    df["scrap_rate"] = df["scrap_units"] / df["produced_units"].replace(0, np.nan)
    high_scrap = df[df["scrap_rate"] > 0.05].head(80)

    for _, row in high_scrap.iterrows():
        date = row["date"]
        line_id = row["line_id"]
        shift_id = row["shift_id"]
        scrap_rate = round(row["scrap_rate"] * 100, 2)
        sku = row["sku_id"]
        supplier = row["supplier_id"]

        text = f"""Line: {line_id}
Date: {date}
Shift: {shift_id}

Summary:
Elevated scrap rate of {scrap_rate}% was observed on SKU {sku} during this shift.
Material was sourced from {supplier}, and operators reported more frequent jams and additional quality checks at the inspection station.

Actions Taken:
- Cleared accumulated material from feeders and guides.
- Performed additional visual inspections on incoming material from {supplier}.
- Logged a potential quality concern for supplier {supplier} for follow-up with the purchasing and quality teams.

Recommendations:
- Pre-screen a small sample lot from {supplier} before full changeover.
- Increase inspection frequency for {supplier} deliveries until scrap levels return to baseline.
- If high scrap persists across multiple shifts, consider temporary switch to an alternate supplier or material spec.
"""
        file_path = reports_dir / f"shift_{line_id}_{date}_{shift_id}.txt"
        file_path.write_text(text)


def write_weekly_line_summaries(unstructured_dir: Path, production_daily: pd.DataFrame, downtime_events: pd.DataFrame):
    # (Same as before)
    summaries_dir = unstructured_dir / "weekly_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prod = production_daily.copy()
    prod["date"] = pd.to_datetime(prod["date"])
    prod["week"] = prod["date"].dt.to_period("W").apply(lambda r: r.start_time.date())

    weekly = (
        prod.groupby(["week", "line_id"])
        .agg(
            total_units=("produced_units", "sum"),
            total_scrap=("scrap_units", "sum"),
            avg_scrap_rate=(
                "scrap_units",
                lambda x: x.sum()
                / max(1, prod.loc[x.index, "produced_units"].sum()),
            ),
            total_downtime_min=("downtime_minutes", "sum"),
        )
        .reset_index()
    )

    dt = downtime_events.copy()
    dt["start_dt"] = pd.to_datetime(dt["start_time"])
    dt["week"] = dt["start_dt"].dt.to_period("W").apply(lambda r: r.start_time.date())

    cause_counts = (
        dt.groupby(["week", "line_id", "cause_code"])
        .size()
        .reset_index(name="cnt")
    )

    for _, row in weekly.iterrows():
        week_start = row["week"]
        line_id = row["line_id"]
        total_units = int(row["total_units"])
        total_scrap = int(row["total_scrap"])
        scrap_rate_pct = round(row["avg_scrap_rate"] * 100, 2)
        total_dt = int(row["total_downtime_min"])

        subset_causes = cause_counts[
            (cause_counts["week"] == week_start) & (cause_counts["line_id"] == line_id)
        ].sort_values("cnt", ascending=False)

        top_causes_txt = ""
        for _, c in subset_causes.head(3).iterrows():
            top_causes_txt += f"- {c['cause_code']} occurred {c['cnt']} time(s)\n"

        text = f"""Weekly Operations Summary
Week starting: {week_start}
Line: {line_id}

Production:
- Total units produced: {total_units}
- Total scrap units: {total_scrap}
- Average scrap rate: {scrap_rate_pct}%

Downtime:
- Total downtime: {total_dt} minutes
"""
        if top_causes_txt:
            text += "Top downtime causes:\n" + top_causes_txt + "\n"
        else:
            text += "Top downtime causes:\n- No significant downtime recorded this week.\n\n"

        text += """Usage:
This summary is used during weekly operations huddles to:
- Review overall performance for the line
- Identify recurring scrap or downtime drivers
- Decide if corrective actions, trials, or maintenance interventions are required.
"""
        file_path = summaries_dir / f"weekly_summary_{line_id}_{week_start}.txt"
        file_path.write_text(text)


def write_machine_operator_notes(unstructured_dir: Path, dim_machine: pd.DataFrame):
    # (Same as before)
    notes_dir = unstructured_dir / "machine_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    for _, row in dim_machine.iterrows():
        machine_id = row["machine_id"]
        line_id = row["line_id"]
        machine_type = row["machine_type"]

        if machine_id == PROBLEM_MACHINE:
            content = f"""Machine: {machine_id}
Line: {line_id}
Type: {machine_type}

Operator Notes (High Attention Asset):
This machine has a known history of vibration and alignment issues, especially during extended production runs
and immediately after mechanical changeovers.

Common Observations:
- Increased noise and vibration at higher speeds.
- Minor jams at the infeed section that can escalate if not cleared promptly.
- Gradual rise in scrap rate (surface defects, dimension out of spec) when alignment drifts.

Operating Guidelines:
- Perform a quick vibration and noise check at the start of each shift.
- Stop and inspect if minor stoppages occur more than 3 times within a 2-hour window.
- Document all abnormal events in the shift log, even if resolved quickly.

Escalation:
- If unplanned downtime exceeds 60 minutes in a shift, notify maintenance immediately.
- If repeated alignment issues occur in the same week, request a full mechanical inspection.
"""
        else:
            content = f"""Machine: {machine_id}
Line: {line_id}
Type: {machine_type}

Operator Notes:
This asset generally runs reliably under standard operating conditions.
Most issues are related to routine wear, minor jams, or sensor calibration.

Operating Guidelines:
- Follow the standard startup checklist, including safety and interlock checks.
- Keep the work area clear of excess material, tools, and debris.
- Record any recurring stoppages, product defects, or unusual noise patterns in the shift report.

Escalation:
- If the same stoppage reason appears more than twice in a shift, log a maintenance request.
- If product quality concerns are observed, hold suspect pallets and notify quality.
"""
        file_path = notes_dir / f"machine_notes_{machine_id}.txt"
        file_path.write_text(content)


def write_sop_docs(unstructured_dir: Path):
    # (Same as before)
    sop_dir = unstructured_dir / "sop_docs"
    sop_dir.mkdir(parents=True, exist_ok=True)

    sop_alignment = """Document Title: SOP - Mechanical Alignment for M-Series Machines
Document ID: SOP-MECH-ALIGN-001
Revision: 2.0
Owner: Maintenance Engineering
Effective Date: 2023-09-01

1. Purpose
This procedure defines the standard method for inspecting and correcting mechanical alignment
on M-series machines to minimize vibration, scrap, and unplanned downtime.

2. Scope
Applies to all M-series machines operating on Lines L1, L2, and L3.

3. Responsibilities
- Maintenance Technicians: Perform alignment inspections and adjustments.
- Line Supervisors: Schedule planned alignment checks and verify completion.
- Operators: Report abnormal noise, vibration, or recurring jams.

4. Required PPE
- Safety glasses
- Safety shoes
- Cut-resistant gloves
- Hearing protection (in high-noise areas)

5. Procedure
5.1 Lockout/Tagout
    a. Stop the machine and follow site lockout/tagout procedures.
    b. Verify zero energy state before proceeding.

5.2 Inspection
    a. Inspect belts, pulleys, couplings, and mounting points for visible wear or misalignment.
    b. Check for loose fasteners, cracked mounts, or abnormal wear patterns.

5.3 Alignment
    a. Use approved alignment tools to verify shaft and belt alignment.
    b. Adjust as necessary until alignment is within specified tolerances.
    c. Tighten all mounting bolts to specified torque.

5.4 Verification
    a. Remove lockout/tagout and restart the machine.
    b. Run a short test batch and monitor vibration, noise, and scrap levels.
    c. If abnormal conditions persist, escalate to the Maintenance Supervisor.

6. Documentation
- Record all alignment activities, measurements, and replaced components in the maintenance system.
- Tag any components that must be monitored in future inspections.
"""
    sop_material = """Document Title: SOP - Handling Material Quality Issues
Document ID: SOP-MAT-QUAL-002
Revision: 1.3
Owner: Quality Assurance
Effective Date: 2023-06-15

1. Purpose
To define the standard response when material quality issues are suspected or confirmed,
in order to protect product quality and prevent excessive scrap.

2. Scope
Applies to all incoming raw materials used on Lines L1, L2, and L3.

3. Responsibilities
- Operators: Identify and report visible defects, jams, or abnormal behavior linked to material.
- Quality Technicians: Inspect suspect material, perform checks, and make disposition decisions.
- Purchasing: Coordinate with suppliers on quality concerns and corrective actions.

4. Triggers for Material Quality Review
- Sudden increase in scrap rate associated with a specific supplier or batch.
- Repeated jams or feeding issues with the same material lot.
- Visual defects (surface damage, incorrect dimensions, contamination).

5. Procedure
5.1 Immediate Actions
    a. Stop production on the affected line if safety or severe quality risk is present.
    b. Isolate the suspect material batch and label it as "ON HOLD".
    c. Record batch ID, supplier, and time of issue in the shift report.

5.2 Inspection
    a. Perform visual inspection and basic measurements on suspect material.
    b. Compare against approved specifications and reference samples.
    c. Document findings and, if required, collect photo evidence.

5.3 Disposition
    a. If material is within spec, release for controlled trial run with close monitoring.
    b. If material is out of spec, hold the batch and escalate to Quality and Purchasing.
    c. Consider switching to an alternate approved supplier or batch if available.

6. Documentation
- All material issues must be logged in the quality system with batch ID and supplier.
- Supplier performance metrics are updated monthly based on number and severity of issues.
"""
    (sop_dir / "SOP_alignment_M_series.txt").write_text(sop_alignment)
    (sop_dir / "SOP_material_quality.txt").write_text(sop_material)


# -----------------------------
# Main orchestration
# -----------------------------
def main():
    structured_dir, unstructured_dir = ensure_dirs()

    # Dimensions
    dim_line = generate_dim_line()
    dim_shift = generate_dim_shift()
    dim_machine = generate_dim_machine()
    # Pass dim_machine here? No, dim_sku is independent.
    dim_sku = generate_dim_sku()

    # Facts
    # UPDATED: Pass dim_sku to production_daily
    production_daily = generate_production_daily(dim_machine, dim_sku)
    downtime_events = generate_downtime_events(dim_machine)
    maint_tickets = generate_maintenance_tickets(downtime_events)

    # Save structured data
    dim_line.to_csv(structured_dir / "dim_line.csv", index=False)
    dim_shift.to_csv(structured_dir / "dim_shift.csv", index=False)
    dim_machine.to_csv(structured_dir / "dim_machine.csv", index=False)
    dim_sku.to_csv(structured_dir / "dim_sku.csv", index=False)

    production_daily.to_csv(structured_dir / "production_daily.csv", index=False)
    downtime_events.to_csv(structured_dir / "downtime_events.csv", index=False)
    maint_tickets.to_csv(structured_dir / "maintenance_tickets.csv", index=False)

    # Unstructured data
    write_maintenance_logs(unstructured_dir, maint_tickets)
    write_shift_reports(unstructured_dir, production_daily)
    write_sop_docs(unstructured_dir)
    write_weekly_line_summaries(unstructured_dir, production_daily, downtime_events)
    write_machine_operator_notes(unstructured_dir, dim_machine)

    print("Synthetic MANRAG dataset generated successfully.")


if __name__ == "__main__":
    main()