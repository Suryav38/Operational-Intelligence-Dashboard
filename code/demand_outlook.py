"""2027 start-demand views and a time-separated request-realization backtest.

Expected starts calibrate supplied requests; they are not a market-demand
forecast. Program coefficients pool fabs. Site filters change aggregation,
not the population used to fit the model.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fab_config as C
import forecast_model as fm
import execution_model as em
import executive_charts as ec
import review_model as rm
import fab_demo as D

BREAKDOWNS={'Technology family':'technology_family','Program family':'family','Program name':'program','Route technology':'route_technology'}
MEASURES={'Requested plan':'requested_wspw','Expected starts':'expected_wspw'}

def plan(data,site='Both',programs=None,pilot_case='baseline'):
    model=fm.fit(data['wafer_start_history'])
    frame=fm.corrected_plan(data['plan_2027'],model).rename(columns={'corrected_wspw':'expected_wspw'})
    frame=frame.merge(model[['program','realization_rate']],on='program',how='left',validate='many_to_one')
    frame['family']=frame.program.map(em.FAMILIES)
    frame['technology_family']=frame.program.map(D.TECHNOLOGY)
    frame['route_technology']=frame.program.map(lambda p:'Includes EUV' if C.PROGRAMS[p]['passes']['EUV'] else 'DUV only')
    if pilot_case!='baseline':
        extra=data['pilot_plan_quarterly'].query('case==@pilot_case').copy()
        extra['expected_wspw']=extra.requested_wspw.map(lambda n:float('nan') if n>0 else 0.)
        extra['realization_rate']=float('nan');extra['family']='DRAM';extra['technology_family']='DRAM';extra['route_technology']='Includes EUV'
        frame=pd.concat([frame,extra.drop(columns='case')],ignore_index=True)
    if site!='Both':frame=frame[frame.site==site]
    if programs is not None:frame=frame[frame.program.isin(programs)]
    return frame.sort_values(['quarter','site','program']).reset_index(drop=True)

def quarterly(frame):
    totals=frame.groupby('quarter')[['requested_wspw','expected_wspw']].sum().reindex(C.PLAN_QUARTERS,fill_value=0)
    unknown=frame[frame.expected_wspw.isna()&frame.requested_wspw.gt(0)].quarter.unique()
    totals.loc[unknown,'expected_wspw']=float('nan')
    return totals

def validation(data,site='Both',programs=None):
    history=data['wafer_start_history'].copy()
    train=history[history.quarter<=fm.TRAIN_END]
    coefficients={}
    for p in C.PROGRAM_ORDER:
        rows=train[train.program==p]
        x=rows.requested_wspw.to_numpy(float);y=rows.actual_wspw.to_numpy(float)
        coefficients[p]=float(np.dot(x,y)/np.dot(x,x)) if len(x) and np.dot(x,x)>0 else 1.
    held=history[history.quarter>fm.TRAIN_END].copy()
    held['expected_wspw']=held.requested_wspw*held.program.map(coefficients)
    if site!='Both':held=held[held.site==site]
    if programs is not None:held=held[held.program.isin(programs)]
    if held.empty:return held,dict(raw_mape=float('nan'),expected_mape=float('nan'),rows=0)
    actual=held.actual_wspw.replace(0,np.nan)
    metrics=dict(raw_mape=float((100*(held.requested_wspw-actual).abs()/actual).mean()),
                 expected_mape=float((100*(held.expected_wspw-actual).abs()/actual).mean()),rows=len(held))
    return held,metrics

def mix_chart(frame,group,measure):
    dimension=BREAKDOWNS[group];col=MEASURES[measure]
    pivot=frame.pivot_table(index='quarter',columns=dimension,values=col,aggfunc='sum',fill_value=0).reindex(C.PLAN_QUARTERS,fill_value=0)
    orders={'program':list(D.PROGRAMS),'family':['DRAM','NAND','HBM','Experiments'],'technology_family':['DRAM','NAND','Experiments'],'route_technology':['Includes EUV','DUV only']}
    pivot=pivot.reindex(columns=[name for name in orders[dimension] if name in pivot.columns])
    colors={p:color for p,color in zip(D.PROGRAMS,ec.PROGRAM_COLORS)} if dimension=='program' else (
        {'DRAM':ec.BLUE,'NAND':ec.TEAL,'HBM':'#9b8cb9','Experiments':'#b8c4d3'} if dimension in ('family','technology_family') else {'Includes EUV':ec.BLUE,'DUV only':ec.TEAL})
    fig=go.Figure()
    for i,name in enumerate(pivot.columns):
        label=ec.PROGRAM_SHORT[name] if dimension=='program' else name
        fig.add_bar(x=pivot.index,y=pivot[name],name=label,marker_color=colors[name],
            hovertemplate='%{x}<br>%{y:,.0f} wafers/week<extra>'+label+'</extra>')
    ec.style(fig,335);fig.update_layout(barmode='stack',margin=dict(l=0,r=12,t=15,b=95),legend=dict(y=-.25))
    fig.update_yaxes(title='Wafer starts / week');return fig

def fab_chart(frame,measure):
    col=MEASURES[measure];fig=go.Figure()
    for site in C.SITES:
        rows=frame[frame.site==site]
        if rows.empty:continue
        totals=rows.groupby('quarter')[col].sum().reindex(C.PLAN_QUARTERS,fill_value=0)
        fig.add_bar(x=totals.index,y=totals,name=rm.SITE_LABELS[site],marker_color=ec.SITE_COLORS[site],
            text=totals.map(lambda x:f'{x:,.0f}'),textposition='outside')
    ec.style(fig,335);fig.update_layout(barmode='group');fig.update_yaxes(title='Wafer starts / week',rangemode='tozero');return fig

def ramp_chart(frame,measure,quarter):
    col=MEASURES[measure];frame=frame.copy()
    frame['label']=frame.site.map(rm.SITE_LABELS)+' · '+frame.program.map(ec.PROGRAM_SHORT)
    pivot=frame.pivot_table(index='label',columns='quarter',values=col,aggfunc='sum',fill_value=0).reindex(columns=C.PLAN_QUARTERS,fill_value=0)
    order=[rm.SITE_LABELS[s]+' · '+ec.PROGRAM_SHORT[p] for s in C.SITES for p in D.PROGRAMS]
    pivot=pivot.reindex([x for x in order if x in pivot.index])
    fig=go.Figure(go.Heatmap(x=pivot.columns,y=pivot.index,z=pivot.values,
        text=pivot.map(lambda v:f'{v:,.0f}').values,texttemplate='%{text}',textfont=dict(size=11),
        colorscale=[[0,'#edf3fa'],[1,'#2867c8']],showscale=False,xgap=3,ygap=3,
        hovertemplate='%{y} · %{x}<br>%{z:,.0f} wafers/week<extra></extra>'))
    ec.style(fig,max(300,len(pivot)*32+80));fig.update_xaxes(side='top',showgrid=False);fig.update_yaxes(autorange='reversed',showgrid=False)
    i=C.PLAN_QUARTERS.index(quarter)
    fig.add_shape(type='rect',xref='x',yref='paper',x0=i-.49,x1=i+.49,y0=0,y1=1,line=dict(color='#17375d',width=2),fillcolor='rgba(0,0,0,0)')
    return fig

def plan_comparison(frame):
    totals=quarterly(frame);fig=go.Figure()
    for col,name,color in [('requested_wspw','Requested plan',ec.BLUE),('expected_wspw','Expected starts',ec.TEAL)]:
        fig.add_scatter(x=totals.index,y=totals[col],name=name,mode='lines+markers',line=dict(color=color,width=2),
            hovertemplate='%{x}<br>%{y:,.0f} wafers/week<extra>'+name+'</extra>')
    ec.style(fig,335);fig.update_yaxes(title='Wafer starts / week',rangemode='tozero');return fig

def backtest_chart(held):
    cols=['requested_wspw','actual_wspw','expected_wspw']
    totals=held.groupby('quarter')[cols].sum();fig=go.Figure()
    for col,name,color in [('actual_wspw','Actual starts','#24374f'),('requested_wspw','Raw request',ec.BLUE),('expected_wspw','Adjusted estimate',ec.TEAL)]:
        fig.add_scatter(x=totals.index,y=totals[col],name=name,mode='lines+markers',line=dict(color=color,width=2))
    ec.style(fig,320);fig.update_yaxes(title='Wafer starts / week',rangemode='tozero');return fig
