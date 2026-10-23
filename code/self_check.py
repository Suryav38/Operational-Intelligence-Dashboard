"""
Offline self-check (no API key needed): runs the models, the validations and
every data tool the copilot can call, and prints the headline numbers.
    python code/self_check.py
"""
import json

import assistant as ai
import fab_capacity as fc
import fab_data as fd
import forecast_model as fm

data = fd.load()
p = fd.performance_summary(data)
print(f"Q3 2026: {p['wafer_starts_per_week']} WSPW, cycle time {p['cycle_time_days_full_flow']} d, "
      f"DPML {p['days_per_mask_layer']}, X-factor {p['x_factor']}, line yield {p['line_yield_pct']}%")
f = fm.summary(data["wafer_start_history"], data["plan_2027"])
print(f"Forecast backtest error: raw {f['overall_backtest_mape_raw_pct']}% -> corrected {f['overall_backtest_mape_corrected_pct']}%")
for basis in ("requested", "corrected"):
    print(f"2027 at-risk cells ({basis}):", len(fc.constraints(fc.loading_table(fc.base_state(data, basis)))))
mv = fc.model_vs_simulation(data)
print(f"Model vs simulation: mean absolute difference {mv['difference_pts'].abs().mean():.1f} pts")
print(fd.littles_law_check(data).to_string(index=False))

tools = {t.name: t for t in ai.make_tools(data, {"site": "Both", "basis": "corrected"})}
checks = {
    "fab_performance": {"site": "FAB_B"},
    "capacity_plan": {"basis": "corrected"},
    "forecast_accuracy": {},
    "run_scenario": {"changes": [{"type": "move_load", "program": "NAND_C", "site": "FAB_B", "to_site": "FAB_A", "pct": 25}]},
    "run_sql": {"query": "SELECT site, tool_group, loading_pct FROM sim_tool_group_q3_2026 ORDER BY loading_pct DESC LIMIT 3"},
}
for name, args in checks.items():
    out = json.loads(tools[name].invoke(args))
    print(f"OK {name}: {json.dumps(out)[:110]}...")
print("All checks passed.")
