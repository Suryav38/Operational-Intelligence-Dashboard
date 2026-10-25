"""Execution charts use exact filtered lot cohorts; equipment states stay separate."""
import pandas as pd
import plotly.graph_objects as go
import fab_config as C
import review_model as rm
import executive_charts as ec

COMPONENTS=[('process_days','Processing',ec.BLUE),('queue_days','Queues',ec.AMBER),
            ('transport_days','Dispatch / transport','#a6b5cc'),('hold_days','Engineering holds',ec.TEAL)]

def weekly_flow(p):
    w=p['weekly'];fig=go.Figure()
    for col,name,color in [('wafer_starts','Starts',ec.BLUE),('wafer_outs','Completions',ec.TEAL)]:
        fig.add_bar(x=w.week_start,y=w[col],name=name,marker_color=color,
            hovertemplate='%{x}<br>%{y:,.0f} wafers<extra>'+name+'</extra>')
    ec.style(fig,300);fig.update_layout(barmode='group');fig.update_yaxes(title='Wafers / week')
    fig.update_xaxes(type='date',tickformat='%d %b');return fig

def weekly_trend(p,metric):
    fig=go.Figure();frame=p['by_site_week']
    for s in frame.site.unique():
        rows=frame[frame.site==s]
        fig.add_scatter(x=rows.week_start,y=rows[metric],name=rm.SITE_LABELS[s],mode='lines+markers',
            line=dict(color=ec.SITE_COLORS[s],width=2),marker=dict(size=5))
    if len(frame.site.unique())>1:
        fig.add_scatter(x=p['weekly'].week_start,y=p['weekly'][metric],name='Combined',mode='lines',
            line=dict(color='#24374f',dash='dot',width=2))
    ec.style(fig,300);fig.update_xaxes(type='date',tickformat='%d %b')
    fig.update_yaxes(title='Input-wafer equivalents' if metric=='wip_wafers' else 'Days / completed lot',rangemode='tozero')
    return fig

def cycle_by_program(p):
    fig=go.Figure();frame=p['cycles'];programs=[q for q in C.PROGRAM_ORDER if q in set(frame.program)]
    for s in p['by_site_week'].site.unique():
        ct=frame[frame.site==s].groupby('program').cycle_time_days.mean().reindex(programs)
        fig.add_bar(y=[ec.PROGRAM_SHORT[q] for q in programs],x=ct,orientation='h',name=rm.SITE_LABELS[s],marker_color=ec.SITE_COLORS[s])
    ec.style(fig,310);fig.update_layout(barmode='group');fig.update_xaxes(title='Mean cycle time (days)');fig.update_yaxes(autorange='reversed');return fig

def cycle_breakdown(p,group='site'):
    fig=go.Figure();frame=p['cycles'].groupby(group)[[c[0] for c in COMPONENTS]].mean()
    if group=='program':frame=frame.reindex([q for q in C.PROGRAM_ORDER if q in frame.index])
    labels=frame.index.map(rm.SITE_LABELS if group=='site' else ec.PROGRAM_SHORT)
    for col,name,color in COMPONENTS:
        fig.add_bar(y=labels,x=frame[col],orientation='h',name=name,marker_color=color,
            hovertemplate='%{y}<br>%{x:.2f} days<extra>'+name+'</extra>')
    ec.style(fig,310);fig.update_layout(barmode='stack',margin=dict(l=0,r=12,t=12,b=100),legend=dict(y=-.34))
    fig.update_xaxes(title='Mean days / completed lot');fig.update_yaxes(autorange='reversed');return fig

def priority_compare(p):
    fig=go.Figure();frame=p['cycles']
    programs=[q for q in C.PROGRAM_ORDER if q in set(frame.program)]
    for priority,color in [('HOT',ec.TEAL),('NORMAL',ec.BLUE)]:
        group=frame[frame.priority==priority].groupby('program').cycle_time_days.mean().reindex(programs)
        fig.add_bar(y=[ec.PROGRAM_SHORT[q] for q in programs],x=group,orientation='h',name=priority,marker_color=color)
    ec.style(fig,310);fig.update_layout(barmode='group');fig.update_xaxes(title='Mean cycle time (days)');fig.update_yaxes(autorange='reversed');return fig

def cycle_distribution(p):
    fig=go.Figure()
    for s in p['by_site_week'].site.unique():
        frame=p['cycles'].query('site==@s')
        fig.add_box(y=frame.program.map(ec.PROGRAM_SHORT),x=frame.cycle_time_days,orientation='h',
            name=rm.SITE_LABELS[s],marker_color=ec.SITE_COLORS[s],boxpoints=False)
    ec.style(fig,310);fig.update_layout(boxmode='group');fig.update_xaxes(title='Cycle time distribution (days)')
    fig.update_yaxes(categoryorder='array',categoryarray=[ec.PROGRAM_SHORT[q] for q in reversed(C.PROGRAM_ORDER) if q in set(p['cycles'].program)])
    return fig

def wip_age(p):
    frame=p['wip_lots'].copy()
    frame['age_band']=pd.cut(frame.age_days,bins=[0,7,14,28,56,float('inf')],labels=['0–7 d','7–14 d','14–28 d','28–56 d','56+ d'],right=False)
    fig=go.Figure()
    for s in p['by_site_week'].site.unique():
        counts=frame[frame.site==s].groupby('age_band',observed=False).wafers_in.sum()
        fig.add_bar(x=counts.index.astype(str),y=counts,name=rm.SITE_LABELS[s],marker_color=ec.SITE_COLORS[s])
    ec.style(fig,310);fig.update_layout(barmode='group');fig.update_yaxes(title='WIP (input-wafer equivalents)');return fig

def equipment_states(frame):
    fig=go.Figure();labels=frame.site.map(rm.SITE_LABELS)+' · '+frame.tool_group.map(rm.GROUP_LABELS)
    for col,name,color in [('productive_pct','Productive',ec.BLUE),('standby_pct','Standby','#bac8da'),
            ('engineering_pct','Engineering',ec.TEAL),('sched_down_pct','Scheduled down',ec.AMBER),('unsched_down_pct','Unscheduled down',ec.RED)]:
        fig.add_bar(y=labels,x=frame[col],orientation='h',name=name,marker_color=color,
            hovertemplate='%{y}<br>%{x:.1f}% of calendar time<extra>'+name+'</extra>')
    ec.style(fig,max(320,len(frame)*30+130));fig.update_layout(barmode='stack',margin=dict(l=0,r=12,t=12,b=100),legend=dict(y=-.23))
    fig.update_xaxes(title='Equipment time allocation (%)',range=[0,100.2]);fig.update_yaxes(autorange='reversed');return fig

def queue_trend(data,site,groups,first=None,last=None):
    frame=data['sim_tool_group_weekly']
    if first is not None:frame=frame[frame.week_start.between(first,last)]
    if site!='Both':frame=frame[frame.site==site]
    frame=frame[frame.tool_group.isin(groups)]
    fig=go.Figure()
    for (s,g),rows in frame.groupby(['site','tool_group'],sort=False):
        fig.add_scatter(x=rows.week_start,y=rows.avg_queue_h,name=rm.SITE_LABELS[s]+' · '+rm.GROUP_LABELS[g],
            mode='lines+markers',marker=dict(size=3),line=dict(color=ec.PROGRAM_COLORS[C.TOOL_GROUP_ORDER.index(g)%len(ec.PROGRAM_COLORS)],dash='solid' if s=='FAB_A' else 'dot'))
    ec.style(fig,340);fig.update_xaxes(type='date',tickformat='%d %b');fig.update_yaxes(title='Queue per visit (h)');return fig

def attainment(frame):
    fig=go.Figure()
    for col,name,color in [('requested_wspw','Requested',ec.BLUE),('actual_wspw','Actual',ec.TEAL)]:
        fig.add_scatter(x=frame.quarter,y=frame[col],name=name,mode='lines+markers',line=dict(color=color,width=2))
    ec.style(fig,320);fig.update_yaxes(title='Wafer starts / week');return fig
