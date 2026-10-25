"""Fast offline integrity checks; no AI key or simulation regeneration required."""
import numpy as np
import fab_data as fd
import fab_config as C
import fab_demo as D
import launch_model as L
import scenario_requests as R
import execution_model as em

data=fd.load()
assert len(data['sim_fab_weekly'].week_start.unique())==52
assert data['sim_fab_weekly'].flow_balance_error.eq(0).all()
assert data['launch_weekly'].flow_balance_error.eq(0).all()
plan=data['plan_2027'].assign(technology=lambda f:f.program.map(D.TECHNOLOGY))
ratio=plan[plan.technology=='DRAM'].groupby('quarter').requested_wspw.sum()/plan.groupby('quarter').requested_wspw.sum()
assert np.allclose(ratio,.75)
state,table=L.capacity(data,'baseline');row=table.query("site=='FAB_A' and tool_group=='EUV' and quarter=='2027Q2'").iloc[0]
assert abs(row.loading_pct-99.2)<.01
review=R.run(data,'Demand growth','FAB_A',program='DRAM_B',pct=10)
assert review['comparison'].query("site=='FAB_A' and tool_group=='EUV' and quarter=='2027Q2'").loading_pct_scenario.iloc[0]==107.4
for case in ['introduction','slower']:
    summary=L.study_summary(data,case)
    assert summary['matched_lots']==summary['initial_lots'] and summary['unfinished_lots']==0
    print(case,'additional WIP',round(summary['additional_wip']),'mean existing-lot shift',round(summary['mean_delay_days'],2),'days')
print('52-week flow balances, 75% DRAM target, capacity calculations and paired pilot cohorts passed.')
