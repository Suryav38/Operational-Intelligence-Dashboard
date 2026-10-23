"""
Offline self-check (no API key needed): runs the metrics, the capacity model
and every data tool the assistant can call, and prints the headline numbers.
    python code/self_check.py
"""
import json

import assistant as ai
import capacity_model as cm
import plant_metrics as pm

data = pm.load_data()
k = pm.compute_kpis(data)
print(f"2024 plant: OEE {k['oee_pct']}% | availability {k['availability_pct']}% | scrap {k['scrap_rate_pct']}% | "
      f"MTBF {k['mtbf_h']} h | MTTR {k['mttr_min']} min")
p, e, _ = pm.filter_data(data)
for f in pm.exceptions(p, e):
    print("Watch list:", f["text"])
for line in cm.run_scenario(data)["lines"]:
    print(f"Q1 2025 {line['line']}: loading {line['loading_pct']}% -> {line['status']}")

scope = {"line": "All", "machine": "All", "start": "2024-10-01", "end": "2024-12-31"}
tools = {t.name: t for t in ai.make_tools(data, scope, pm.DB_PATH)}
checks = {
    "get_kpis": {"line": "L1", "machine": "M11"},
    "downtime_breakdown": {"group_by": "machine", "top_n": 3},
    "scrap_breakdown": {"group_by": "line_supplier", "top_n": 3},
    "machine_reliability": {"top_n": 3},
    "run_capacity_scenario": {"line": "L2", "extra_shifts_per_week": 2},
    "run_sql": {"query": "SELECT shift_id, ROUND(100.0*SUM(scrap_units)/SUM(produced_units),2) AS scrap_pct "
                         "FROM production_daily GROUP BY shift_id"},
}
for name, args in checks.items():
    out = json.loads(tools[name].invoke(args))
    print(f"OK {name}: {json.dumps(out)[:120]}...")
print("All checks passed.")
