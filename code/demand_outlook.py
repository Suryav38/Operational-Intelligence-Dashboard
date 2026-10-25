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
COMPARISON_QUARTERS=[f'{year}Q{q}' for year in (2026,2027) for q in range(1,5)]


def comparison_frame(data,site='Both',programs=None,pilot_case='baseline'):
    """Separate observed synthetic history, user planning bridge and future plan.

    A missing prior-period observation stays missing. The bridge never enters
    the model fit; future expectations still use only the supplied history.
    """
    history=data['wafer_start_history'].copy()
    history=history[history.quarter.str.startswith('2026')]
    history['record_kind']='historical';history['expected_wspw']=np.nan
    history['allocation_basis']='Generated reporting-quarter program records'
    bridge=data.get('demand_bridge_2026q4',pd.DataFrame()).copy()
    bridge['expected_wspw']=np.nan
    future=plan(data,site,programs,pilot_case).copy()
    future['actual_wspw']=np.nan;future['record_kind']='future_plan'
    future['period_basis']='Calendar-quarter average requested weekly rate; expected starts are model estimates'
    future['allocation_basis']='Supplied 2027 program plan and selected pilot ramp'
    frame=pd.concat([history,bridge,future],ignore_index=True)
    if site!='Both':frame=frame[frame.site==site]
    if programs is not None:frame=frame[frame.program.isin(programs)]
    frame['family']=frame.program.map(D.FAMILIES)
    frame['technology_family']=frame.program.map(D.TECHNOLOGY)
    frame['program_name']=frame.program.map(D.NAMES)
    frame['route_technology']=frame.program.map(lambda p:'Includes EUV' if D.PROGRAMS[p]['passes']['EUV'] else 'DUV only')
    return frame.sort_values(['quarter','site','program']).reset_index(drop=True)


def comparison_totals(frame):
    totals=frame.groupby('quarter')[['requested_wspw','actual_wspw','expected_wspw']].sum(min_count=1).reindex(COMPARISON_QUARTERS)
    unknown=frame[frame.expected_wspw.isna()&frame.requested_wspw.gt(0)&frame.record_kind.eq('future_plan')].quarter.unique()
    totals.loc[unknown,'expected_wspw']=np.nan
    return totals


def comparison_changes(frame,quarter):
    totals=comparison_totals(frame)
    previous=quarter.replace('2027','2026')
    prior=totals.loc[previous,'requested_wspw'];current=totals.loc[quarter,'requested_wspw']
    end=totals.loc['2026Q4','requested_wspw'];start=totals.loc['2027Q1','requested_wspw']
    pct=lambda a,b:float(100*(a/b-1)) if pd.notna(b) and b>0 and pd.notna(a) else np.nan
    return dict(prior_quarter=previous,prior_requested=prior,current_requested=current,
        yoy_pct=pct(current,prior),prior_q4_requested=end,next_q1_requested=start,
        transition_delta=start-end,transition_pct=pct(start,end))


def _chart_scopes(frame):
    scopes=[('All selected programs',frame)]
    for label,mask in [('DRAM incl. HBM',frame.technology_family.eq('DRAM')),
        ('NAND',frame.technology_family.eq('NAND')),('HBM only',frame.family.eq('HBM')),
        ('Experiments',frame.family.eq('Experiments'))]:
        if mask.any():scopes.append((label,frame[mask]))
    for p in D.PROGRAMS:
        rows=frame[frame.program==p]
        if not rows.empty:scopes.append((D.NAMES[p],rows))
    return scopes


def _chart_filter(fig,frame,series):
    buttons=[dict(label=label,method='update',args=[{'y':series(rows)},{'yaxis.autorange':True}]) for label,rows in _chart_scopes(frame)]
    fig.update_layout(updatemenus=[dict(type='dropdown',direction='down',x=0,y=1.21,xanchor='left',yanchor='top',
        buttons=buttons,showactive=True,font=dict(size=10),bgcolor='white',bordercolor='#b8c4d3')],
        margin=dict(l=0,r=12,t=65,b=85),legend=dict(y=-.24,itemclick='toggle',itemdoubleclick='toggleothers'))
    return fig


def year_comparison_chart(frame):
    def series(rows):
        t=comparison_totals(rows)
        return [t.loc[[f'{year}Q{q}' for q in range(1,5)],'requested_wspw'].tolist() for year in (2026,2027)]
    fig=go.Figure();ys=series(frame)
    for i,(year,color) in enumerate([(2026,'#aebfd6'),(2027,ec.BLUE)]):
        fig.add_bar(x=['Q1','Q2','Q3','Q4'],y=ys[i],name=f'{year} requested',marker_color=color,
            marker_pattern_shape=['','','','/'] if year==2026 else '',
            customdata=['Historical request']*3+['Q4 planning reference; assumed allocation'] if year==2026 else ['Future requested plan']*4,
            hovertemplate=str(year)+'%{x}<br>%{y:,.1f} WSPW<br>%{customdata}<extra></extra>')
    ec.style(fig,390);fig.update_layout(barmode='group')
    fig.update_yaxes(title='Requested wafer starts / week',rangemode='tozero')
    return _chart_filter(fig,frame,series)


def history_outlook_chart(frame):
    def series(rows):
        t=comparison_totals(rows)
        return [t[col].tolist() for col in ('requested_wspw','actual_wspw','expected_wspw')]
    fig=go.Figure();ys=series(frame)
    basis=['Historical reporting-quarter record']*3+['Q4 planning reference; assumed allocation']+['2027 future plan']*4
    for i,(label,color,dash) in enumerate([('Requested',ec.BLUE,'solid'),('Historical actual',ec.TEAL,'solid'),('Model expected',ec.AMBER,'dash')]):
        fig.add_scatter(x=COMPARISON_QUARTERS,y=ys[i],name=label,mode='lines+markers',connectgaps=False,
            line=dict(color=color,width=2,dash=dash),marker=dict(symbol=['circle']*3+['diamond']+['circle']*4),
            customdata=basis,hovertemplate='%{x}<br>%{y:,.1f} WSPW<br>%{customdata}<extra>'+label+'</extra>')
    ec.style(fig,390);fig.update_yaxes(title='Wafer starts / week',rangemode='tozero')
    fig.update_xaxes(type='category',tickangle=-35,categoryorder='array',categoryarray=COMPARISON_QUARTERS)
    fig.add_shape(type='line',xref='x',yref='paper',x0=3.5,x1=3.5,y0=0,y1=1,line=dict(color='#b8c4d3',dash='dot'))
    return _chart_filter(fig,frame,series)

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
