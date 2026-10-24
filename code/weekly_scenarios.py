"""Weekly capacity screens from quarterly rates, with exact calendar overlap.

This is not a dispatch simulation. HOT is request metadata; it never creates
capacity or changes the historical lot cohorts.
"""
import pandas as pd
import plotly.graph_objects as go
import fab_config as C
import fab_capacity as fc
import review_model as rm
import executive_charts as ec

def calendar():
    start=pd.Timestamp('2027-01-01');end=pd.Timestamp('2028-01-01')
    monday=start-pd.Timedelta(days=start.weekday());rows=[]
    while monday<end:
        lo=max(start,monday);hi=min(end,monday+pd.Timedelta(days=7));days=(hi-lo).days
        rows.append(dict(week_start=monday.strftime('%Y-%m-%d'),period_start=lo,period_end=hi,
            period_days=days,label=lo.strftime('%d %b')+'–'+(hi-pd.Timedelta(days=1)).strftime('%d %b')+(' · partial week' if days<7 else '')))
        monday+=pd.Timedelta(days=7)
    return pd.DataFrame(rows)

def quarter_of(date):return f'{date.year}Q{(date.month-1)//3+1}'

def segments(lo,hi):
    out=[]
    for i,q in enumerate(C.PLAN_QUARTERS):
        start=pd.Timestamp(int(q[:4]),3*(int(q[-1])-1)+1,1)
        end=start+pd.DateOffset(months=3)
        days=max(0,(min(hi,end)-max(lo,start)).days)
        if days:out.append((i,days/7))
    return out

def week_table(base,changed=None,first_week=None,last_week=None):
    base_tools=fc._tools(base);changed_tools=fc._tools(changed) if changed else base_tools
    records=[];starts=[]
    for w in calendar().itertuples():
        apply=changed is not None and first_week<=w.week_start<=last_week
        state=changed if apply else base;tools=changed_tools if apply else base_tools
        parts=segments(w.period_start,w.period_end)
        for site in C.SITES:
            volume=sum(sum(state['wspw'][site][p][i] for p in C.PROGRAM_ORDER)*fraction for i,fraction in parts)
            starts.append(dict(site=site,week_start=w.week_start,starts_in_period=volume,starts_weekly_equiv=volume*7/w.period_days))
            for group in C.TOOL_GROUP_ORDER:
                required=sum(sum(state['wspw'][site][p][i]*fc._hours_per_wafer(state,p,group,i) for p in C.PROGRAM_ORDER)*fraction for i,fraction in parts)
                available=sum(tools[site][group][i]*168*state['avail'][site][group][i]*fraction for i,fraction in parts)
                if required==0 and available==0:continue
                records.append(dict(site=site,tool_group=group,week_start=w.week_start,period_start=w.period_start,
                    period_end=w.period_end,period_days=w.period_days,week_label=w.label,
                    required_h_period=required,available_h_period=available,
                    loading_pct=round(100*required/available,1) if available else float('nan')))
    return pd.DataFrame(records),pd.DataFrame(starts)

def run(data,site,program,pct,first_week,last_week,ceiling=.85,recovery_group=None,recovery_points=0.,target_wspw=None):
    weeks=calendar().set_index('week_start')
    if first_week not in weeks.index or last_week not in weeks.index or first_week>last_week:
        raise ValueError('The end week must be on or after the start week, within the 2027 planning calendar.')
    lo=weeks.loc[first_week,'period_start'];hi=weeks.loc[last_week,'period_end']
    review=rm.run_review(data,site,program,pct,quarter_of(lo),quarter_of(hi-pd.Timedelta(days=1)),
        ceiling,recovery_group,recovery_points,target_wspw=target_wspw)
    before,starts_before=week_table(review['base_state'])
    after,starts_after=week_table(review['base_state'],review['scenario_state'],first_week,last_week)
    keys=['site','tool_group','week_start','period_start','period_end','period_days','week_label']
    comparison=before.merge(after,on=keys,how='outer',suffixes=('_base','_scenario'))
    comparison['change_pp']=comparison.loading_pct_scenario-comparison.loading_pct_base
    comparison['status']=comparison.loading_pct_scenario.map(lambda x:rm.risk_label(x,ceiling))
    comparison['inside_request_window']=comparison.week_start.between(first_week,last_week)
    change=f'target {target_wspw:g} WSPW' if target_wspw is not None else f'{pct:+g}% starts'
    review['applied']=[f"{C.PROGRAMS[program]['name']}: {change} at {rm.SITE_LABELS[site]}, {lo:%d %b %Y}–{hi-pd.Timedelta(days=1):%d %b %Y} only"]
    if recovery_group and recovery_points:
        review['applied'].append(f'{rm.GROUP_LABELS[recovery_group]}: +{recovery_points:g} availability points in the same weekly window only')
    review.update(timing_mode='weeks',first_week=first_week,last_week=last_week,
        weekly_baseline=before,weekly_scenario=after,weekly_comparison=comparison,
        starts_before=starts_before,starts_after=starts_after)
    return review

def summary(review,site,week,after=False):
    table=review['weekly_scenario' if after else 'weekly_baseline']
    rows=table[table.week_start==week]
    starts=review['starts_after' if after else 'starts_before']
    starts=starts[starts.week_start==week]
    if site!='Both':rows=rows[rows.site==site];starts=starts[starts.site==site]
    worst=rows.assign(sort=rows.loading_pct.fillna(float('inf'))).sort_values('sort',ascending=False).iloc[0]
    return dict(starts=starts.starts_weekly_equiv.sum(),at_risk=int((rows.loading_pct.ge(100*review['ceiling'])|rows.loading_pct.isna()).sum()),
        over_capacity=int((rows.loading_pct.gt(100)|rows.loading_pct.isna()).sum()),groups=len(rows),
        worst_load=worst.loading_pct,worst_site=worst.site,worst_group=worst.tool_group)

def equipment_chart(comparison,scope,ceiling):
    frame=rm.select_scope(comparison,scope)
    frame=frame[frame.tool_group.isin(rm.FOCUS_GROUPS)|frame.loading_pct_base.ge(ceiling*100)|frame.loading_pct_scenario.ge(ceiling*100)|frame.loading_pct_scenario.isna()]
    frame=frame.sort_values('loading_pct_scenario',ascending=False)
    labels=frame.site.map(rm.SITE_LABELS)+' · '+frame.tool_group.map(rm.GROUP_LABELS)
    fig=go.Figure()
    for col,name,color in [('loading_pct_base','Baseline','#bac9dd'),('loading_pct_scenario','Scenario',ec.BLUE)]:
        fig.add_bar(y=labels,x=frame[col],orientation='h',name=name,marker_color=color,
            text=frame[col].map(lambda x:'No capacity' if pd.isna(x) else f'{x:.1f}%'),textposition='outside')
    ec.style(fig,max(300,len(frame)*45+90));fig.update_layout(barmode='group')
    fig.add_vline(x=ceiling*100,line_dash='dash',line_color=ec.AMBER);fig.add_vline(x=100,line_dash='dot',line_color=ec.RED)
    maximum=frame[['loading_pct_base','loading_pct_scenario']].max().max()
    fig.update_xaxes(title='Equipment loading (%)',range=[0,max(115,maximum*1.18)])
    fig.update_yaxes(autorange='reversed');return fig

def risk_chart(review,scope,ceiling):
    fig=go.Figure()
    for key,name,color in [('weekly_baseline','Baseline','#98aac2'),('weekly_scenario','Scenario',ec.BLUE)]:
        frame=rm.select_scope(review[key],scope)
        counts=frame.assign(risk=frame.loading_pct.ge(100*ceiling)|frame.loading_pct.isna()).groupby('period_start').risk.sum()
        fig.add_scatter(x=counts.index,y=counts,name=name,mode='lines',line=dict(color=color,width=2,shape='hv'))
    ec.style(fig,320);fig.update_yaxes(title='At-risk equipment groups',dtick=1,rangemode='tozero');fig.update_xaxes(tickformat='%d %b')
    weeks=calendar().set_index('week_start')
    fig.add_vrect(x0=weeks.loc[review['first_week'],'period_start'],x1=weeks.loc[review['last_week'],'period_end'],fillcolor=ec.BLUE,opacity=.05,line_width=0)
    return fig
