"""Small, validated capacity requests: demand shock, unavailable tools, transfer."""
import math
import pandas as pd
import fab_config as C
import fab_capacity as fc
import review_model as rm
import weekly_scenarios as W
import launch_model as L

OPERATIONS=['Demand growth','Tool unavailability','Qualified load transfer']

def run(data,operation,site,program='DRAM_B',pct=10.,group='EUV',unavailable=1,to_site='FAB_B',
        first='2027Q2',last='2027Q2',timing='quarters',ceiling=.85,recovery_group=None,recovery_points=0.,target_wspw=None,pilot_case='baseline'):
    if operation not in OPERATIONS or site not in C.SITES:raise ValueError('Choose a supported operation and fab.')
    if timing=='weeks':
        calendar=W.calendar().set_index('week_start')
        if first not in calendar.index or last not in calendar.index or first>last:raise ValueError('The end week must follow the start week within 2027.')
        first_q=W.quarter_of(calendar.loc[first,'period_start']);last_q=W.quarter_of(calendar.loc[last,'period_end']-pd.Timedelta(days=1))
    else:
        if first not in C.PLAN_QUARTERS or last not in C.PLAN_QUARTERS or first>last:raise ValueError('The end quarter must be on or after the start quarter.')
        first_q,last_q=first,last
    if not math.isfinite(pct) or not -100<=pct<=100:raise ValueError('Choose a finite percentage from -100 to 100.')
    base=L.planning_state(data,pilot_case)
    if operation=='Demand growth':
        review=rm.run_review(data,site,program,pct,first_q,last_q,ceiling,recovery_group,recovery_points,target_wspw=target_wspw)
        applied=review['applied'];changes=review['changes']
        changed,_,warnings=fc.apply_changes(base,changes)
        if warnings:raise ValueError('; '.join(warnings))
    elif operation=='Qualified load transfer':
        if to_site==site or to_site not in C.SITES:raise ValueError('Choose a different receiving fab.')
        if program not in C.QUALIFIED[site] or program not in C.QUALIFIED[to_site]:raise ValueError('The route is not qualified at both fabs; the transfer is blocked.')
        if not 0<=pct<=100:raise ValueError('Transfer share must be from 0 to 100%.')
        changes=[fc.PlanChange(type='move_load',site=site,to_site=to_site,program=program,pct=pct,from_quarter=first_q,to_quarter=last_q)]
        changed,applied,warnings=fc.apply_changes(base,changes)
        if warnings:raise ValueError('; '.join(warnings))
        changes=[c.model_dump(exclude_none=True) for c in changes]
    else:
        if group not in C.TOOL_GROUPS or not isinstance(unavailable,int) or unavailable<1:raise ValueError('Choose an equipment group and a positive whole-tool quantity.')
        changed,_,_=fc.apply_changes(base,[]);tools=fc._tools(base)
        for i,q in enumerate(C.PLAN_QUARTERS):
            if first_q<=q<=last_q:
                if unavailable>tools[site][group][i]:raise ValueError('Unavailable quantity exceeds released tools in the requested period.')
                changed['tools_base'][site][group][i]-=unavailable
        applied=[f'{site}: {unavailable} {group} tools unavailable in the selected window'];changes=[]
    b=L.loading_state(base,ceiling);s=L.loading_state(changed,ceiling)
    review=dict(base_state=base,scenario_state=changed,baseline=b,scenario=s,comparison=rm.compare_tables(b,s,ceiling),
        applied=applied,changes=changes,site=site,program=program,pct=pct,first_quarter=first_q,last_quarter=last_q,
        ceiling=ceiling,target_wspw=target_wspw,timing_mode=timing,operation=operation,pilot_case=pilot_case,to_site=to_site if operation=='Qualified load transfer' else None)
    if timing=='weeks':
        wb,sb=W.week_table(base,pilot_case=pilot_case);ws,ss=W.week_table(base,changed,first,last,pilot_case=pilot_case)
        keys=['site','tool_group','week_start','period_start','period_end','period_days','week_label']
        comparison=wb.merge(ws,on=keys,how='outer',suffixes=('_base','_scenario'))
        comparison['change_pp']=comparison.loading_pct_scenario-comparison.loading_pct_base
        comparison['status']=comparison.loading_pct_scenario.map(lambda x:rm.risk_label(x,ceiling))
        comparison['inside_request_window']=comparison.week_start.between(first,last)
        review.update(first_week=first,last_week=last,weekly_baseline=wb,weekly_scenario=ws,weekly_comparison=comparison,starts_before=sb,starts_after=ss)
        review['applied']=[f'{item} · applied only to weeks {first} through {last}' for item in applied]
    return review
