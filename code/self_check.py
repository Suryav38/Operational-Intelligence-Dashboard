"""Fast offline integrity checks; no AI key or simulation regeneration required."""
import numpy as np
import fab_data as fd
import fab_config as C
import fab_demo as D
import launch_model as L
import scenario_requests as R
import wafer_analyst as A

data=fd.load()
assert len(data['sim_fab_weekly'].week_start.unique())==52
assert data['sim_fab_weekly'].flow_balance_error.eq(0).all()
assert data['sim_fab_weekly_q4'].flow_balance_error.eq(0).all()
assert sorted(data['sim_fab_weekly_q4'].week_start.unique())==['2026-09-28','2026-10-05','2026-10-12','2026-10-19']
# The July run-rate plan keeps the 75% DRAM wafer-start mix the pilot study was generated with.
july=[(p,v[i]) for s in C.JULY_RUN_RATE_WSPW_2027.values() for p,v in s.items() for i in range(4)]
assert abs(sum(v for p,v in july if D.TECHNOLOGY[p]=='DRAM')/sum(v for _,v in july)-.75)<.005
# Hand check: EUV at Fab A, 2027Q2, October requested plan.
state,table=L.capacity(data,'baseline');row=table.query("site=='FAB_A' and tool_group=='EUV' and quarter=='2027Q2'").iloc[0]
A_=C.REQUESTED_WSPW_2027['FAB_A'];expected=100*(A_['DRAM_B'][1]+A_['HBM_TV'][1])*7/60/(168*.7)
assert abs(row.loading_pct-round(expected,1))<.01
review=R.run(data,'Demand growth','FAB_A',program='DRAM_B',pct=10)
after=review['comparison'].query("site=='FAB_A' and tool_group=='EUV' and quarter=='2027Q2'").loading_pct_scenario.iloc[0]
assert abs(after-round(100*(1.1*A_['DRAM_B'][1]+A_['HBM_TV'][1])*7/60/(168*.7),1))<.01
# Analyst hand check: Node B run rate 825 + HBM 180 fills EUV in 2027Q1, so +100 has no room.
w=A.quarter_weeks('2027Q1')
r=A.analyze(data,dict(site='FAB_A',program='DRAM_B',quarter='2027Q1',requested_wspw=925,protected=['HBM_TV'],first_week=w[0],last_week=w[-1]))
assert r['fits_under_ceiling']==0 and r['recommended'] is None
for case in ['introduction','slower']:
    summary=L.study_summary(data,case)
    assert summary['matched_lots']==summary['initial_lots'] and summary['unfinished_lots']==0
print(f'EUV 2027Q2 loading {row.loading_pct}% → {after}% with +10% Node B; analyst hand check passed.')
print('Flow balances (history + Q4 to date), plan mix, capacity and analyst checks passed.')
