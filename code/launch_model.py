"""Planning overlay and matched-cohort summaries for one explicit pilot program."""
import copy,math
import numpy as np
import pandas as pd
import fab_config as C
import fab_capacity as fc
import fab_demo as D
import review_model as rm

def planning_state(data,case='baseline'):
    state=fc.base_state(data)
    rows=data['pilot_plan_quarterly'].query('case==@case').set_index('quarter')
    for site in C.SITES:
        state['wspw'][site][D.PILOT]=[float(rows.loc[q,'requested_wspw']) if site=='FAB_A' else 0. for q in C.PLAN_QUARTERS]
    return state

def capacity(data,case='baseline',ceiling=.85):
    state=planning_state(data,case)
    return state,loading_state(state,ceiling)

def loading_state(state,ceiling=.85):
    table=fc.loading_table(state,ceiling);tools=fc._tools(state)
    for idx,row in table.iterrows():
        i=C.PLAN_QUARTERS.index(row.quarter)
        required=sum(state['wspw'][row.site].get(p,[0.]*4)[i]*D.hours_per_wafer(p,row.tool_group) for p in D.PROGRAMS)
        n=tools[row.site][row.tool_group][i];per=168*state['avail'][row.site][row.tool_group][i]
        needed=math.ceil(required/(per*ceiling)-1e-9) if required else 0
        table.loc[idx,['required_h_wk','loading_pct','tools_needed_at_ceiling','tool_gap']]=[
            round(required,1),round(100*required/(n*per),1) if n else np.nan,needed,max(0,needed-n)]
    return table

def workload(data,case,site,group,quarter):
    state=planning_state(data,case);i=C.PLAN_QUARTERS.index(quarter);rows=[]
    for program,spec in D.PROGRAMS.items():
        starts=state['wspw'][site][program][i]
        if not starts:continue
        rows.append({'Program':spec['name'],'Starts / week':starts,'Passes / wafer':spec['passes'][group],
            'Minutes / pass':C.TOOL_GROUPS[group]['min_per_wafer']*spec['time_mult'].get(group,1.),
            'Required h / week':starts*D.hours_per_wafer(program,group),'Sample fraction':C.TOOL_GROUPS[group].get('sample_rate',1.)})
    frame=pd.DataFrame(rows);total=frame['Required h / week'].sum()
    frame['Share %']=100*frame['Required h / week']/total if total else 0.
    return frame.sort_values('Required h / week',ascending=False)

def study_summary(data,case):
    rows=data['launch_summary'].query('case==@case')
    baseline=data['launch_summary'].query("case=='baseline'")
    pairs=data['launch_lot_comparison'].query('case==@case')
    matched=pairs[pairs.matched.astype(str).str.lower().eq('true')]
    return dict(case=case,replications=int(rows.seed.nunique()),launch_date=D.LAUNCH_DATE,
        baseline_post_launch_wip=float(baseline.post_launch_avg_wip.mean()),post_launch_wip=float(rows.post_launch_avg_wip.mean()),
        additional_wip=float(rows.post_launch_avg_wip.mean()-baseline.post_launch_avg_wip.mean()),
        peak_wip=float(rows.post_launch_peak_wip.max()),mean_delay_days=float(matched.additional_days.mean()),
        p90_delay_days=float(matched.additional_days.quantile(.9)),matched_lots=len(matched),
        initial_lots=int(rows.initial_existing_lots.sum()),unfinished_lots=int(rows.unfinished_existing_lots.sum()),
        pilot_completed_lots_mean=float(rows.pilot_completed_lots.mean()),
        limits='Synthetic matched existing-lot cohort; incomplete lots excluded from completion deltas and counted separately. Replicate ranges are simulation variability, not real-world predictive confidence.')
