"""Presentation charts; scope changes display, never the underlying workload."""
import plotly.graph_objects as go
import pandas as pd
import fab_config as C
import review_model as rm
import fab_demo as D

BLUE,TEAL,AMBER,RED='#2f5d8a','#6f8f7a','#c4841d','#b8412f'
INK,MUTED,GRID='#1d2329','#6b737c','#eceef0'
FONT='IBM Plex Sans, Helvetica Neue, Arial, sans-serif'
SITE_COLORS={'FAB_A':BLUE,'FAB_B':'#9db4c9','Both':INK}
PROGRAM_SHORT=dict(D.NAMES)
# Order follows fab_config.PROGRAM_ORDER + pilot: DRAM A, DRAM B, NAND, HBM, short loops, pilot
PROGRAM_COLORS=['#2f5d8a','#7fa1c3','#6f8f7a','#b08d4a','#c9ced4','#8a5a83']

def style(fig,height=300,title_x=None):
    fig.update_layout(template='plotly_white',height=height,
        font=dict(family=FONT,size=12,color=INK),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=12,t=10,b=40),
        legend=dict(orientation='h',y=1.02,yanchor='bottom',x=0,font=dict(size=11,color=MUTED),traceorder='normal',title_text=''),
        hoverlabel=dict(bgcolor='white',bordercolor=GRID,font=dict(family=FONT,size=12,color=INK)),bargap=.28)
    fig.update_xaxes(gridcolor=GRID,zeroline=False,linecolor='#d5d9dd',tickfont=dict(color=MUTED,size=11),title_font=dict(color=MUTED,size=11))
    fig.update_yaxes(gridcolor=GRID,zeroline=False,tickfont=dict(color=MUTED,size=11),title_font=dict(color=MUTED,size=11))
    return fig

def sites(scope):return list(C.SITES) if scope=='Both' else [scope]

def focus_table(table,scope,ceiling,all_groups=False):
    t=rm.select_scope(table,scope)
    if all_groups:return t
    risky=set(zip(t.loc[t.loading_pct.ge(ceiling*100)|t.loading_pct.isna(),'site'],t.loc[t.loading_pct.ge(ceiling*100)|t.loading_pct.isna(),'tool_group']))
    return t[[g in rm.FOCUS_GROUPS or (s,g) in risky for s,g in zip(t.site,t.tool_group)]]

def loading_heatmap(table,scope,quarter,ceiling,all_groups=False):
    t=focus_table(table,scope,ceiling,all_groups).copy()
    t['row']=t.site.map(rm.SITE_LABELS)+' · '+t.tool_group.map(rm.GROUP_LABELS)
    p=t.pivot(index='row',columns='quarter',values='loading_pct')
    order=[rm.SITE_LABELS[s]+' · '+rm.GROUP_LABELS[g] for s in sites(scope) for g in C.TOOL_GROUP_ORDER]
    p=p.reindex([x for x in order if x in p.index],columns=C.PLAN_QUARTERS)
    fig=go.Figure(go.Heatmap(z=p.map(lambda x:2 if pd.isna(x) or x>100 else (1 if x>=ceiling*100 else 0)).values,
        x=p.columns,y=p.index,text=p.map(lambda x:'No capacity' if pd.isna(x) else f'{x:.1f}%').values,
        texttemplate='%{text}',textfont=dict(size=12,color='#14243b'),zmin=0,zmax=2,showscale=False,xgap=3,ygap=3,
        colorscale=[[0,'#dcece7'],[.2499,'#dcece7'],[.25,'#f5e3c3'],[.7499,'#f5e3c3'],[.75,'#f4d7d7'],[1,'#f4d7d7']],
        hovertemplate='%{y} · %{x}<br>Loading %{text}<extra></extra>'))
    qi=C.PLAN_QUARTERS.index(quarter)
    fig.add_shape(type='rect',xref='x',yref='paper',x0=qi-.49,x1=qi+.49,y0=0,y1=1,line=dict(color=BLUE,width=2),fillcolor='rgba(0,0,0,0)')
    style(fig,max(260,len(p)*32+60))
    fig.update_xaxes(side='top',showgrid=False)
    fig.update_yaxes(autorange='reversed',showgrid=False)
    return fig

def equipment_compare(table,scope,quarter,ceiling,all_groups=False):
    selected=focus_table(table,scope,ceiling,all_groups)
    groups=[g for g in C.TOOL_GROUP_ORDER if g in set(selected.tool_group)]
    t=rm.select_scope(table,scope,quarter)
    t=t[t.tool_group.isin(groups)]
    fig=go.Figure()
    for s in sites(scope):
        p=t[t.site==s].set_index('tool_group').reindex(groups)
        fig.add_bar(y=[rm.GROUP_LABELS[g] for g in groups],x=p.loading_pct,orientation='h',name=rm.SITE_LABELS[s],
                    marker_color=SITE_COLORS[s],text=p.loading_pct.map(lambda x:'' if pd.isna(x) else f'{x:.1f}%'),textposition='outside',
                    hovertemplate='%{y}<br>%{x:.1f}% loading<extra>'+rm.SITE_LABELS[s]+'</extra>')
    style(fig,max(260,len(groups)*38+100))
    fig.update_layout(barmode='group'); fig.add_vline(x=ceiling*100,line_dash='dash',line_color=AMBER)
    fig.add_vline(x=100,line_dash='dot',line_color=RED)
    fig.update_xaxes(title='Equipment loading (%)',range=[0,max(115,t.loading_pct.max()*1.18)])
    fig.update_yaxes(autorange='reversed')
    return fig

def starts_mix(state,scope,quarter,programs=None):
    fig=go.Figure();i=C.PLAN_QUARTERS.index(quarter)
    programs=C.PROGRAM_ORDER if programs is None else programs
    for s in sites(scope):
        fig.add_bar(y=[PROGRAM_SHORT[p] for p in programs],x=[state['wspw'][s][p][i] for p in programs],
                    orientation='h',name=rm.SITE_LABELS[s],marker_color=SITE_COLORS[s],hovertemplate='%{y}<br>%{x:,.0f} wafers/week<extra>'+rm.SITE_LABELS[s]+'</extra>')
    style(fig,300);fig.update_layout(barmode='group');fig.update_xaxes(title='Requested wafer starts / week');fig.update_yaxes(autorange='reversed')
    return fig

def workload_drivers(state,scope,table,ceiling,programs=None):
    t=focus_table(table,scope,ceiling);groups=list(dict.fromkeys(zip(t.site,t.tool_group)))
    quarter=t.quarter.iloc[0] if len(t) else C.PLAN_QUARTERS[0]
    labels=[rm.SITE_LABELS[s]+' · '+rm.GROUP_LABELS[g] for s,g in groups]
    records=[rm.program_workload(state,s,g,quarter) for s,g in groups]
    fig=go.Figure()
    selected=set(C.PROGRAM_ORDER if programs is None else programs)
    for p,color in zip(C.PROGRAM_ORDER,PROGRAM_COLORS):
        if p not in selected:continue
        name=C.PROGRAMS[p]['name'];values=[]
        for r in records:
            m=r[r.Program==name]; values.append(float(m['Share %'].iloc[0]) if len(m) else 0)
        fig.add_bar(y=labels,x=values,orientation='h',name=PROGRAM_SHORT[p],marker_color=color,
                    hovertemplate='%{y}<br>%{x:.1f}% of required hours<extra>'+PROGRAM_SHORT[p]+'</extra>')
    if selected!=set(C.PROGRAM_ORDER):
        other=[sum(r.loc[~r.Program.isin([C.PROGRAMS[p]['name'] for p in selected]),'Share %']) for r in records]
        fig.add_bar(y=labels,x=other,orientation='h',name='Other programs',marker_color='#d3dce8')
    style(fig,max(300,len(groups)*30+110));fig.update_layout(barmode='stack',margin=dict(l=0,r=12,t=12,b=95),legend=dict(y=-.32))
    fig.update_xaxes(title='Share of required tool-hours (%)',range=[0,100]);fig.update_yaxes(autorange='reversed')
    return fig

def scenario_loads(review,scope,quarter,ceiling):
    c=rm.select_scope(review['comparison'],scope,quarter)
    c=c[(c.tool_group.isin(rm.FOCUS_GROUPS))|(c.loading_pct_scenario>=ceiling*100)]
    c=c.sort_values('loading_pct_scenario',ascending=False)
    labels=c.site.map(rm.SITE_LABELS)+' · '+c.tool_group.map(rm.GROUP_LABELS)
    fig=go.Figure()
    for col,name,color in [('loading_pct_base','Baseline','#bac9dd'),('loading_pct_scenario','Scenario',BLUE)]:
        fig.add_bar(y=labels,x=c[col],orientation='h',name=name,marker_color=color,
                    text=c[col].map(lambda x:f'{x:.1f}%'),textposition='outside',hovertemplate='%{y}<br>%{x:.1f}%<extra>'+name+'</extra>')
    style(fig,max(300,len(c)*45+90));fig.update_layout(barmode='group')
    fig.add_vline(x=ceiling*100,line_dash='dash',line_color=AMBER);fig.add_vline(x=100,line_dash='dot',line_color=RED)
    fig.update_xaxes(title='Equipment loading (%)',range=[0,max(115,c[['loading_pct_base','loading_pct_scenario']].max().max()*1.18)])
    fig.update_yaxes(autorange='reversed');return fig

def scenario_risks(review,scope,ceiling):
    fig=go.Figure()
    for key,label,color in [('baseline','Baseline','#bac9dd'),('scenario','Scenario',BLUE)]:
        t=rm.select_scope(review[key],scope)
        counts=t.assign(risk=t.loading_pct.ge(ceiling*100)|t.loading_pct.isna()).groupby('quarter').risk.sum().reindex(C.PLAN_QUARTERS)
        fig.add_bar(x=counts.index,y=counts.values,name=label,marker_color=color,text=counts.values,textposition='outside')
    style(fig,300);fig.update_layout(barmode='group');fig.update_yaxes(title='At-risk fab / equipment combinations',dtick=1,rangemode='tozero');return fig
