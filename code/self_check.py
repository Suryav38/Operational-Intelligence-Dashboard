"""Offline validation of the focused dashboard's models; no API key required."""
import fab_capacity as fc
import fab_data as fd
import review_model as rm

data = fd.load()
state = fc.base_state(data)
table = fc.loading_table(state)
euv = rm.select_scope(table, 'FAB_A', '2027Q1').query("tool_group == 'EUV'").iloc[0]
assert abs(euv.loading_pct - 86.3) < .11
assert abs(euv.available_h_wk - 117.6) < .01
review = rm.run_review(data, 'FAB_A', 'DRAM_B', 10, '2027Q2', '2027Q2')
assert abs(review['scenario_state']['wspw']['FAB_A']['DRAM_B'][1] - 858) < 1e-8
assert review['scenario_state']['wspw']['FAB_A']['DRAM_B'][0] == 700
for site in ('FAB_A', 'FAB_B'):
    s = rm.capacity_summary(table, state, site, '2027Q2')
    p = rm.production_metrics(data, site)
    assert len(p['weekly']) == 13
    assert 0 <= p['line_yield'] <= 100
    assert abs(p['breakdown'].sum() - p['cycle_days']) < .03
    print(f"{site}: {s['starts']:.0f} requested WSPW; {s['at_risk']}/{s['groups']} groups at risk; "
          f"historical throughput {p['avg_outs']:.0f} wafers/week; cycle time {p['cycle_days']:.1f} days")
print('Offline capacity, scenario and historical metric checks passed.')
