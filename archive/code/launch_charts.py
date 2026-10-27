"""Compact charts for route, inventory and matched new-program impacts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fab_config as C
import fab_demo as D
import launch_model as L
import review_model as rm
import executive_charts as ec
import weekly_scenarios as W

COLORS={'baseline':'#8fa3bd','introduction':ec.BLUE,'slower':ec.TEAL}

def mark_launch(fig):
    fig.add_shape(type='line',xref='x',yref='paper',x0=D.LAUNCH_DATE,x1=D.LAUNCH_DATE,y0=0,y1=1,line=dict(color=ec.AMBER,dash='dot'))
    fig.add_annotation(x=D.LAUNCH_DATE,y=1.04,xref='x',yref='paper',text='Pilot release',showarrow=False,font=dict(size=10,color=ec.AMBER))

def inventory(data,site):
    frame=data['installed_tools'].pivot(index='tool_group',columns='site',values='tools').reindex(C.TOOL_GROUP_ORDER)
    columns=list(C.SITES) if site=='Both' else [site];frame=frame[columns]
    fig=go.Figure(go.Heatmap(z=frame.values,x=[rm.SITE_LABELS[s] for s in columns],y=[rm.GROUP_LABELS[g] for g in frame.index],text=frame.values,texttemplate='%{text:.0f}',showscale=False,xgap=3,ygap=3,colorscale=[[0,'#eff3fa'],[1,'#2d6cc7']],hovertemplate='%{x} · %{y}<br>%{z:.0f} tools<extra></extra>'))
    ec.style(fig,400);fig.update_xaxes(side='top',showgrid=False);fig.update_yaxes(autorange='reversed',showgrid=False);return fig

def wip(data,case):
    frame=data['launch_weekly'];fig=go.Figure()
    for key in D.CASES:
        rows=frame[frame.case==key];stats=rows.groupby('week_start').wip_wafers.agg(['mean','min','max'])
        if key==case and key!='baseline':
            fig.add_scatter(x=stats.index,y=stats['max'],mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip')
            fig.add_scatter(x=stats.index,y=stats['min'],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(24,90,219,.09)',showlegend=False,hoverinfo='skip')
        fig.add_scatter(x=stats.index,y=stats['mean'],mode='lines',name=D.CASES[key],line=dict(color=COLORS[key],width=2))
    ec.style(fig,335);mark_launch(fig);fig.update_xaxes(type='date',tickformat='%d %b');fig.update_yaxes(title='WIP (input-wafer equivalents)',rangemode='tozero');return fig

def existing_delays(data,case):
    frame=data['launch_lot_comparison'].query('case==@case').dropna(subset=['additional_days'])
    stats=frame.groupby('program').additional_days.agg(mean='mean',p90=lambda x:x.quantile(.9)).reindex([p for p in C.PROGRAM_ORDER if p in set(frame.program)])
    labels=[D.NAMES[p] for p in stats.index];fig=go.Figure()
    fig.add_bar(y=labels,x=stats['mean'],name='Mean additional time',orientation='h',marker_color=ec.BLUE)
    fig.add_scatter(y=labels,x=stats.p90,name='90th percentile',mode='markers',marker=dict(color=ec.AMBER,size=9,symbol='diamond'))
    ec.style(fig,335);fig.update_xaxes(title='Completion shift vs baseline (days)',rangemode='tozero');fig.update_yaxes(autorange='reversed');return fig

def queues(data,case):
    frame=data['launch_equipment'];post=frame[frame.week_start>=D.LAUNCH_DATE]
    active=post[post.case==case].groupby('tool_group').apply(lambda g:np.average(g.avg_queue_h,weights=g.lot_steps) if g.lot_steps.sum() else 0,include_groups=False)
    groups=list(active.sort_values(ascending=False).head(6).index);fig=go.Figure()
    for key,name,color in [('baseline','Pre-pilot','#bac9dd'),(case,D.CASES[case],ec.BLUE)]:
        rows=post[post.case==key];values=[]
        for group in groups:
            g=rows[rows.tool_group==group];values.append(np.average(g.avg_queue_h,weights=g.lot_steps) if g.lot_steps.sum() else 0)
        fig.add_bar(y=[rm.GROUP_LABELS[g] for g in groups],x=values,name=name,orientation='h',marker_color=color)
    ec.style(fig,335);fig.update_layout(barmode='group');fig.update_xaxes(title='Mean queue per equipment visit (h)');fig.update_yaxes(autorange='reversed');return fig

def pilot_flow(data,case):
    frame=data['launch_program_weekly'].query('case==@case and program==@D.PILOT')
    totals=frame.groupby(['seed','week_start'])[['wafer_starts','wafer_outs']].sum().groupby('week_start').mean();fig=go.Figure()
    for col,name,color in [('wafer_starts','Pilot starts',ec.BLUE),('wafer_outs','Pilot completions',ec.TEAL)]:
        fig.add_bar(x=totals.index,y=totals[col],name=name,marker_color=color)
    ec.style(fig,335);fig.update_layout(barmode='group');mark_launch(fig);fig.update_xaxes(type='date',tickformat='%d %b');fig.update_yaxes(title='Mean wafers / week',rangemode='tozero');return fig

def workload(data,case,site,quarter,table,ceiling,programs,all_groups=False):
    selected=ec.focus_table(table,site,ceiling,all_groups);groups=list(dict.fromkeys(zip(selected.site,selected.tool_group)))
    records=[L.workload(data,case,s,g,quarter) for s,g in groups];fig=go.Figure()
    labels=[rm.SITE_LABELS[s]+' · '+rm.GROUP_LABELS[g] for s,g in groups]
    visible=list(programs)
    for i,p in enumerate(D.PROGRAMS):
        if p not in visible:continue
        values=[r.loc[r.Program==D.NAMES[p],'Share %'].sum() for r in records]
        fig.add_bar(y=labels,x=values,orientation='h',name=D.NAMES[p],marker_color=ec.PROGRAM_COLORS[i])
    other=[r.loc[~r.Program.isin([D.NAMES[p] for p in visible]),'Share %'].sum() for r in records]
    if any(other):fig.add_bar(y=labels,x=other,orientation='h',name='Other programs',marker_color='#d3dce8')
    ec.style(fig,max(310,len(groups)*32+105));fig.update_layout(barmode='stack',legend=dict(y=-.3),margin=dict(b=110))
    fig.update_xaxes(title='Share of required equipment hours (%)',range=[0,100]);fig.update_yaxes(autorange='reversed');return fig

def route_visits(frame):
    counts=frame.groupby('tool_group').size().reindex(C.TOOL_GROUP_ORDER,fill_value=0)
    fig=go.Figure(go.Bar(y=[rm.GROUP_LABELS[g] for g in counts.index],x=counts,orientation='h',marker_color=ec.BLUE,text=counts,textposition='outside'))
    ec.style(fig,370);fig.update_xaxes(title='Visits per full-route wafer');fig.update_yaxes(autorange='reversed');return fig

def ordered_route(frame):
    fig=go.Figure(go.Scatter(x=frame.step_no,y=frame.tool_group.map(rm.GROUP_LABELS),mode='lines+markers',line=dict(color='#b9c8dc',width=1),marker=dict(color=ec.BLUE,size=5),customdata=frame[['layer','recipe_id']],hovertemplate='Step %{x} · %{y}<br>Layer %{customdata[0]} · %{customdata[1]}<extra></extra>'))
    ec.style(fig,370);fig.update_xaxes(title='Ordered operation number');fig.update_yaxes(categoryorder='array',categoryarray=[rm.GROUP_LABELS[g] for g in reversed(C.TOOL_GROUP_ORDER)]);return fig

def weekly_loading(data,case,group,ceiling):
    state=L.planning_state(data,'baseline');fig=go.Figure()
    for key,name,color in [('baseline','Pre-pilot','#8fa3bd'),(case,D.CASES[case],ec.BLUE)]:
        frame,_=W.week_table(state,pilot_case=key)
        rows=frame[(frame.site=='FAB_A')&(frame.tool_group==group)]
        fig.add_scatter(x=rows.period_start,y=rows.loading_pct,name=name,mode='lines',line=dict(color=color,width=2,shape='hv'))
    ec.style(fig,300);mark_launch(fig)
    fig.add_hline(y=ceiling*100,line_dash='dash',line_color=ec.AMBER);fig.add_hline(y=100,line_dash='dot',line_color=ec.RED)
    fig.update_xaxes(tickformat='%b %Y');fig.update_yaxes(title='Requested-route loading (%)',rangemode='tozero');return fig
