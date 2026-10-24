"""Stakeholder review: execution first, explicit cohorts, capacity and lot detail."""
from __future__ import annotations
import html
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import executive_charts as ec
import performance_charts as pc
import execution_model as em
import oee_model as om
import fab_capacity as fc
import fab_config as C
import fab_data as fd
import review_model as rm
import demand_outlook as demand
import weekly_scenarios as weekly

st.set_page_config(page_title='TD Fab | Performance & Planning',page_icon='◫',layout='wide')
st.markdown('<style>'+Path(__file__).with_name('dashboard.css').read_text()+'</style>',unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def load_data():return fd.load()
try:data=load_data()
except Exception as exc:st.error(f'Project data could not be loaded: {exc}');st.stop()
ss=st.session_state

def card(col,label,value,sub='',tone=''):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value {tone}">{html.escape(str(value))}</div><div class="kpi-sub">{html.escape(sub)}</div></div>',unsafe_allow_html=True)
def number(value,decimals=1,suffix=''):
    return '—' if pd.isna(value) else f'{value:,.{decimals}f}'+suffix

def chart(fig,key):st.plotly_chart(fig,width='stretch',config={'displayModeBar':False},key=key)
def panel(col,name):
    box=col.container(border=True);box.subheader(name);return box

def title(name,period):
    st.markdown('<div class="topline"><div class="eyebrow">TD / STRATEGY & OPERATIONS</div><span class="badge">SYNTHETIC DEMO</span></div>',unsafe_allow_html=True)
    st.title(name);st.caption(rm.SITE_LABELS[site]+' · '+period)
def table_view(frame):
    out=frame.copy();out['Site']=out.site.map(rm.SITE_LABELS);out['Equipment']=out.tool_group.map(rm.GROUP_LABELS);return out

def export_table(frame,labels=None):
    return frame.rename(columns=labels or {}).to_csv(index=False)

st.sidebar.markdown('<div class="brand"><div class="brand-mark">TD</div><div><div class="brand-title">Fab performance & planning</div><div class="brand-sub">Stakeholder review</div></div></div>',unsafe_allow_html=True)
PAGES=['Fab performance','Demand outlook','Capacity & scenarios','Lot flow & cycle time']
if ss.get('page') not in PAGES:ss['page']=PAGES[0]
page=st.sidebar.radio('Workspace',PAGES,key='page')
st.sidebar.divider()
site=st.sidebar.selectbox('Site',['Both','FAB_A','FAB_B'],format_func=rm.SITE_LABELS.get,key='site')
family=st.sidebar.selectbox('Program family',['All families','DRAM','NAND','HBM','Experiments'],key='family')
options=em.program_options(site,family)
if ss.get('_program_scope')!=(site,family):
    ss['_program_scope']=(site,family);ss['programs']=options
programs=st.sidebar.multiselect('Program names',options,format_func=lambda p:C.PROGRAMS[p]['name'],key='programs')
if not options:st.sidebar.caption('This family has no qualified program at this fab.')
if page in ['Capacity & scenarios','Demand outlook']:
    ss.setdefault('quarter',ss.get('_planning_quarter','2027Q2'))
    quarter=st.sidebar.selectbox('Review quarter',C.PLAN_QUARTERS,key='quarter');ss['_planning_quarter']=quarter
    ceiling=ss.get('ceiling_pct',85)/100
    if page=='Capacity & scenarios':
        with st.sidebar.expander('Planning assumptions'):
            ceiling=st.slider('Planning limit (%)',75,95,85,5,key='ceiling_pct')/100
            st.caption('85% is a demo planning policy. Equipment releases contribute from the start of their stated quarter.')
    priority='All priorities';first=last=None;include_short=False
else:
    quarter=ss.get('_planning_quarter','2027Q2');ceiling=ss.get('ceiling_pct',85)/100
    priority=st.sidebar.selectbox('Lot priority',['All priorities','HOT','NORMAL'],key='priority')
    weeks=sorted(data['sim_fab_weekly'].week_start.unique())
    first,last=st.sidebar.select_slider('Reporting weeks',weeks,value=(weeks[0],weeks[-1]),format_func=lambda w:pd.Timestamp(w).strftime('%d %b'),key='report_weeks')
    include_short=st.sidebar.checkbox('Include short loops in cycle time',value=False,key='include_short')
    st.sidebar.caption('Weeks begin Monday. WIP is measured at each week end.')
st.sidebar.divider()
st.sidebar.caption('5 programs · 2 fabs · 13 simulated operating weeks')
state=fc.base_state(data);capacity=fc.loading_table(state,ceiling)


def overview(p):
    cols=st.columns(4)
    card(cols[0],'Completions / week',number(p['avg_outs'],0),'Selected programs & priority')
    card(cols[1],'Cycle time',number(p['cycle_days'],1,' d'),'All selected programs' if include_short else 'Full-flow completed lots')
    card(cols[2],'Average WIP',number(p['avg_wip'],0),'Input-wafer equivalents')
    card(cols[3],'Line yield',number(p['yield_pct'],2,'%'),'Completed-cohort wafer survival')
    st.write('')
    a,b=st.columns(2,gap='large')
    with panel(a,'Weekly wafer flow'):chart(pc.weekly_flow(p),'weekly_flow')
    with panel(b,'WIP by fab'):chart(pc.weekly_trend(p,'wip_wafers'),'weekly_wip')
    a,b=st.columns(2,gap='large')
    with panel(a,'Weekly cycle time'):chart(pc.weekly_trend(p,'avg_cycle_time_days'),'weekly_ct')
    with panel(b,'Cycle time by program'):chart(pc.cycle_by_program(p),'cycle_program')
    with st.expander('Program scorecard & metric definitions'):
        summary=p['completed'].groupby(['site','program']).agg(completed_lots=('lot_id','size'),wafers_in=('wafers_in','sum'),wafers_out=('wafers_out','sum'),mean_cycle_days=('cycle_time_days','mean')).reset_index()
        summary['Yield (%)']=100*summary.wafers_out/summary.wafers_in
        summary['Site']=summary.site.map(rm.SITE_LABELS);summary['Program']=summary.program.map(ec.PROGRAM_SHORT)
        st.dataframe(summary[['Site','Program','completed_lots','wafers_out','mean_cycle_days','Yield (%)']].rename(columns={'completed_lots':'Completed lots','wafers_out':'Wafers out','mean_cycle_days':'Mean cycle time (d)'}).round(2),hide_index=True,width='stretch')
        st.caption('Flow, WIP and yield include all selected programs. Cycle-time charts follow the short-loop checkbox. Cycle time is completed-lot weighted; yield uses input and output from the same completed cohorts. Scorecard cycle times include each listed program. Filtered WIP is reconstructed from lot timestamps and original input quantities. Empty cohorts display —.')
        st.caption('Little’s Law is a diagnostic relationship, not an enforced equality between these finite-window KPIs. The weekly snapshots and completed cohorts have different boundary effects.')
    st.download_button('Download filtered weekly performance',export_table(p['weekly']),'filtered_weekly_performance.csv','text/csv')


def equipment_page():
    st.caption('Q3 2026 · full-quarter shared equipment groups · site filter applies; program, priority and week filters do not.')
    available=[g for g in C.TOOL_GROUP_ORDER if any(C.INSTALLED_TOOLS[s][g] for s in ec.sites(site))]
    selected=st.multiselect('Equipment groups',available,default=[g for g in rm.FOCUS_GROUPS if g in available],format_func=rm.GROUP_LABELS.get,key='health_groups_'+site)
    if not selected:st.info('Select an equipment group to compare.');return
    frame=em.equipment_health(data,site,selected);cols=st.columns(4)
    worst=frame.sort_values('loading_pct',ascending=False).iloc[0]
    card(cols[0],'Installed machines',number(frame.tools.sum(),0),'Selected equipment groups')
    card(cols[1],'Peak production loading',number(worst.loading_pct,1,'%'),rm.SITE_LABELS[worst.site]+' · '+rm.GROUP_LABELS[worst.tool_group])
    card(cols[2],'Longest equipment queue',number(frame.avg_queue_h.max(),2,' h'),'Mean wait per equipment visit')
    card(cols[3],'Highest downtime',number(frame.downtime_pct.max(),1,'%'),'Scheduled + unscheduled')
    st.write('');a,b=st.columns([1.2,1],gap='large')
    with panel(a,'Equipment time allocation'):chart(pc.equipment_states(frame),'equipment_states')
    with panel(b,'Weekly equipment queues'):chart(pc.queue_trend(data,site,selected),'equipment_queues')
    with st.expander('Equipment scorecard & definitions'):
        view=table_view(frame)
        cols=['Site','Equipment','tools','productive_pct','standby_pct','engineering_pct','sched_down_pct','unsched_down_pct','loading_pct','avg_queue_h']
        labels={'tools':'Machines','productive_pct':'Productive (%)','standby_pct':'Standby (%)','engineering_pct':'Engineering (%)','sched_down_pct':'Scheduled down (%)','unsched_down_pct':'Unscheduled down (%)','loading_pct':'Production loading (%)','avg_queue_h':'Queue / visit (h)'}
        st.dataframe(view[cols].rename(columns=labels).round(2),hide_index=True,width='stretch')
        st.caption('State percentages share a calendar-time denominator and may total 99.9–100.1% after rounding. Production loading uses productive time divided by time available after engineering and downtime. These are group measurements, not individual-machine OEE.')
    st.download_button('Download equipment health',export_table(frame),'Q3_2026_equipment_health.csv','text/csv')


def attainment_page():
    st.caption('2024Q1–2026Q3 · historical weekly start rates by quarter · site and program filters apply; lot priority and reporting-week filters do not.')
    frame=em.historical_attainment(data,site,programs)
    if frame.empty:st.info('No historical starts for this selection.');return
    latest=frame.iloc[-1];cols=st.columns(4)
    card(cols[0],'Latest requested WSPW',number(latest.requested_wspw,0),latest.quarter)
    card(cols[1],'Latest actual WSPW',number(latest.actual_wspw,0),latest.quarter)
    card(cols[2],'Latest attainment',number(latest.attainment_pct,1,'%'),'Actual starts / requested starts')
    card(cols[3],'Latest variance',number(latest.actual_wspw-latest.requested_wspw,0),'Wafers / week')
    st.write('')
    a,b=st.columns([1.25,1],gap='large')
    with panel(a,'Requested vs actual wafer starts'):chart(pc.attainment(frame),'attainment_trend')
    with panel(b,'Attainment by quarter'):
        fig=go.Figure(go.Bar(x=frame.quarter,y=frame.attainment_pct,marker_color=[ec.TEAL if x>=100 else ec.BLUE for x in frame.attainment_pct]))
        ec.style(fig,320);fig.add_hline(y=100,line_dash='dot',line_color=ec.AMBER);fig.update_yaxes(title='Actual / requested (%)',rangemode='tozero');chart(fig,'attainment_ratio')
    with st.expander('Historical source values'):
        st.dataframe(frame.rename(columns={'quarter':'Quarter','requested_wspw':'Requested WSPW','actual_wspw':'Actual WSPW','attainment_pct':'Attainment (%)'}).round(1),hide_index=True,width='stretch')
        st.caption('Attainment measures wafer starts, not output, delivery performance or forecast accuracy. Historical under-realization does not guarantee spare capacity against future commitments.')
    st.download_button('Download historical attainment',export_table(frame),'historical_start_attainment.csv','text/csv')


def machine_oee_page():
    st.caption('Machine OEE requires its own time, speed and first-pass quality records. The fab simulation contains group time states only.')
    mode=st.radio('OEE inputs',['Worked example','Upload machine records'],horizontal=True,key='oee_source')
    if mode=='Worked example':
        st.info('Illustrative calculator · EXAMPLE-01 is independent of Fab A / Fab B. Dashboard filters do not apply.')
        with st.expander('Edit example inputs',expanded=False):
            a,b,c=st.columns(3)
            planned=a.number_input('Planned production time (min)',min_value=1.,value=480.,key='oee_planned')
            runtime=b.number_input('Run time (min)',min_value=0.,value=360.,key='oee_runtime')
            ideal=c.number_input('Ideal cycle time (seconds / unit)',min_value=.001,value=60.,key='oee_ideal')
            a,b=st.columns(2)
            total=a.number_input('Total processed units',min_value=0,value=300,step=1,key='oee_total')
            good=b.number_input('First-pass good units',min_value=0,value=294,step=1,key='oee_good')
        records=om.example();records.loc[0,['planned_minutes','run_minutes','ideal_cycle_seconds','total_count','good_count']]=[planned,runtime,ideal,total,good]
    else:
        st.caption('One single-recipe record per machine and non-overlapping period. Uploaded records use their own scope; dashboard filters do not apply.')
        st.download_button('Download machine OEE template',pd.DataFrame(columns=om.COLUMNS).to_csv(index=False),'machine_oee_template.csv','text/csv')
        uploaded=st.file_uploader('Machine-period CSV',type=['csv'],key='machine_oee_upload')
        if uploaded is None:return
        if uploaded.size>5*1024*1024:st.error('Use a CSV smaller than 5 MB.');return
        try:records=pd.read_csv(uploaded)
        except Exception:st.error('The file could not be read as a CSV.');return
    try:out=om.calculate(records)
    except ValueError as exc:st.error(str(exc));return
    if ss.get('oee_record') not in out.index:ss['oee_record']=out.index[0]
    selection=st.selectbox('Machine / period',list(out.index),format_func=lambda i:str(out.loc[i,'machine_id'])+' · '+str(out.loc[i,'period']),key='oee_record')
    row=out.loc[selection];cols=st.columns(4)
    for col,label,value in zip(cols,['Availability','Performance','Quality','OEE'],[row.availability_pct,row.performance_pct,row.quality_pct,row.oee_pct]):
        card(col,label,number(value,2,'%'),'Selected machine / period')
    st.write('')
    a,b=st.columns(2,gap='large')
    with panel(a,'OEE factors'):
        fig=go.Figure(go.Bar(y=['Availability','Performance','Quality','OEE'],x=[row.availability_pct,row.performance_pct,row.quality_pct,row.oee_pct],orientation='h',marker_color=[ec.BLUE,ec.TEAL,ec.AMBER,'#24374f']))
        ec.style(fig,290);fig.update_xaxes(range=[0,100],title='Percent');fig.update_yaxes(autorange='reversed');chart(fig,'oee_factors')
    with panel(b,'Machine-period comparison'):
        fig=go.Figure(go.Bar(y=out.machine_id.astype(str)+' · '+out.period.astype(str),x=out.oee_pct,orientation='h',marker_color=ec.BLUE))
        ec.style(fig,max(290,min(600,len(out)*30+85)));fig.update_xaxes(range=[0,100],title='OEE (%)');fig.update_yaxes(autorange='reversed');chart(fig,'oee_machines')
    with st.expander('OEE formulas & source records'):
        st.caption('Availability = run / planned time. Performance = ideal cycle time × total count / run time. Quality = first-pass good / total count. OEE = A × P × Q. Planned production time must follow a declared scheduling policy. Split mixed recipes into separate periods with their own ideal cycle standards. No unweighted average of machine OEE is shown. With no processed units, quality is undefined; OEE is zero.')
        st.dataframe(out.round(2),hide_index=True,width='stretch')
        st.markdown('[Calculation reference: Vorne OEE](https://www.oee.com/calculating-oee/)')
    st.download_button('Download calculated machine OEE',export_table(out),'machine_oee_results.csv','text/csv')


def performance_page():
    title('Fab performance',pd.Timestamp(first).strftime('%d %b')+'–'+(pd.Timestamp(last)+pd.Timedelta(days=6)).strftime('%d %b %Y')+' · execution review')
    tabs=st.tabs(['Production overview','Equipment health','Historical attainment','Machine OEE'])
    with tabs[0]:
        if not programs:st.info('Select at least one program in the sidebar.')
        else:
            p=em.execution(data,site,programs,priority,first,last,include_short)
            overview(p)
    with tabs[1]:equipment_page()
    with tabs[2]:attainment_page()
    with tabs[3]:machine_oee_page()


def capacity_outlook():
    m=rm.capacity_summary(capacity,state,site,quarter,ceiling);cols=st.columns(4)
    card(cols[0],'Wafer starts / week',number(m['starts'],0),'All programs in selected fab(s)')
    card(cols[1],'At-risk equipment',f"{m['at_risk']} / {m['groups']}",f'At or above {ceiling*100:.0f}%','alert' if m['at_risk'] else '')
    card(cols[2],'Peak equipment load',number(m['worst_load'],1,'%'),rm.SITE_LABELS[m['worst_site']]+' · '+rm.GROUP_LABELS[m['worst_group']],'danger' if m['worst_load']>100 else 'alert')
    card(cols[3],'Additional tools at limit',str(int(rm.select_scope(capacity,site,quarter).tool_gap.sum())),'Model requirement · before qualification / cost review')
    st.write('');a,b=st.columns([1.25,1],gap='large')
    with panel(a,'Loading across quarters'):
        all_groups=st.toggle('Show all equipment groups',value=False,key='all_groups')
        chart(ec.loading_heatmap(capacity,site,quarter,ceiling,all_groups),'capacity_heatmap')
    with panel(b,quarter+' · equipment comparison'):chart(ec.equipment_compare(capacity,site,quarter,ceiling),'equipment_comparison')
    st.caption(f'Green <{ceiling*100:.0f}% · amber {ceiling*100:.0f}–100% · red >100%. Network loading retains all programs; program filters focus the charts below.')
    if programs:
        a,b=st.columns(2,gap='large')
        with panel(a,'Requested starts · selected programs'):chart(ec.starts_mix(state,site,quarter,programs),'starts_mix')
        with panel(b,'Share of equipment workload'):chart(ec.workload_drivers(state,site,rm.select_scope(capacity,site,quarter),ceiling,programs),'load_drivers')
    with st.expander('Equipment gaps, releases & calculation details'):
        current=table_view(rm.select_scope(capacity,site,quarter));current['Status']=current.loading_pct.map(lambda x:rm.risk_label(x,ceiling))
        st.dataframe(current[['Site','Equipment','tools','tools_needed_at_ceiling','tool_gap','loading_pct','required_h_wk','available_h_wk','Status']].rename(columns={'tools':'Installed / released','tools_needed_at_ceiling':'Needed at limit','tool_gap':'Additional tools','loading_pct':'Loading (%)','required_h_wk':'Required h/week','available_h_wk':'Available h/week'}).round(1),hide_index=True,width='stretch')
        arrivals=rm.select_scope(data['tool_arrivals'],site);st.dataframe(arrivals,hide_index=True,width='stretch')
        choices=list(zip(current.site,current.tool_group))
        selected=st.selectbox('Trace equipment',choices,format_func=lambda k:rm.SITE_LABELS[k[0]]+' · '+rm.GROUP_LABELS[k[1]],key='trace_equipment')
        st.dataframe(rm.program_workload(state,*selected,quarter).round(2),hide_index=True,width='stretch')
        st.caption('Required h/week = Σ starts × passes × minutes/pass × sampling ÷ 60. Available h/week = resources × 168 × production availability. This is an aggregate capacity screen; capital costs and ROI are not included.')
    st.download_button('Download capacity review',export_table(rm.select_scope(capacity,site).assign(planning_limit_pct=ceiling*100,demand_basis='requested')),f'{site}_2027_capacity_review.csv','text/csv')


def demand_page():
    title('2027 demand outlook','2027Q1–Q4 · '+quarter+' focus · wafer-start planning')
    if not programs:st.info('Select at least one program in the sidebar.');return
    frame=demand.plan(data,site,programs);totals=demand.quarterly(frame)
    if frame.empty:st.info('No 2027 demand for the selected fab and programs.');return
    tabs=st.tabs(['Next-year demand','Forecast validation'])
    with tabs[0]:
        current=totals.loc[quarter];q1=totals.loc['2027Q1','requested_wspw'];q4=totals.loc['2027Q4','requested_wspw']
        growth=100*(q4/q1-1) if q1 else float('nan');cols=st.columns(4)
        card(cols[0],quarter+' requested WSPW',number(current.requested_wspw,0),'Selected fab(s) & programs')
        card(cols[1],quarter+' expected WSPW',number(current.expected_wspw,0),'Historical request-realization model')
        card(cols[2],'2027 average request / week',number(totals.requested_wspw.mean(),0),'Mean of four quarterly weekly rates')
        card(cols[3],'Q4 vs Q1 requested growth',number(growth,1,'%'),number(q1,0)+' → '+number(q4,0)+' WSPW')
        st.write('');a,b=st.columns(2)
        breakdown=a.selectbox('Break down demand by',list(demand.BREAKDOWNS),key='demand_breakdown')
        measure=b.radio('Chart basis',list(demand.MEASURES),horizontal=True,key='demand_measure')
        st.caption('Each quarter shows a weekly start rate. Expected starts calibrate requests using historical program realization; capacity commitments continue to use the requested plan.')
        a,b=st.columns([1.15,1],gap='large')
        with panel(a,'2027 · '+measure.lower()+' mix'):chart(demand.mix_chart(frame,breakdown,measure),'demand_mix')
        with panel(b,'2027 · demand by fab'):chart(demand.fab_chart(frame,measure),'demand_fabs')
        a,b=st.columns([1.15,1],gap='large')
        with panel(a,'Quarterly ramp · fab & program'):chart(demand.ramp_chart(frame,measure,quarter),'demand_ramp')
        with panel(b,'Requested plan vs expected starts'):chart(demand.plan_comparison(frame),'demand_comparison')
        with st.expander('Quarter / fab / program detail & assumptions'):
            view=frame.copy();view['Fab']=view.site.map(rm.SITE_LABELS);view['Program']=view.program.map(ec.PROGRAM_SHORT)
            st.dataframe(view[['quarter','Fab','family','Program','route_technology','requested_wspw','expected_wspw','realization_rate']].rename(columns={'quarter':'Quarter','family':'Family','route_technology':'Route technology','requested_wspw':'Requested WSPW','expected_wspw':'Expected WSPW','realization_rate':'Realization factor'}),hide_index=True,width='stretch')
            st.caption('Expected WSPW = requested WSPW × program realization factor, rounded per fab/program/quarter. The model fits actual = factor × requested using history through 2026Q3 and pools the two fabs for each program. This estimates start realization conditional on the supplied plan; it does not independently forecast market demand or future orders. Factors above 1 are allowed when history shows under-requesting.')
            st.caption('Route technology is derived from the configured EUV pass count. It is not a physical wafer specification. Diameter, substrate, wafer grade, product SKU and detailed recipe history are absent from the source files. No confidence interval or annual wafer-volume total is inferred from quarterly weekly rates.')
        export=frame.assign(program_name=frame.program.map(ec.PROGRAM_SHORT),estimate_basis='historical program request realization; pooled fabs')
        st.download_button('Download 2027 demand outlook',export_table(export),'2027_demand_outlook.csv','text/csv')
    with tabs[1]:
        held,metrics=demand.validation(data,site,programs)
        st.caption('Fit: 2024Q1–2025Q4 · held-out evaluation: 2026Q1–Q3 · site and program filters apply.')
        cols=st.columns(3)
        card(cols[0],'Raw-request backtest MAPE',number(metrics['raw_mape'],1,'%'),'Mean error across held-out fab/program/quarter rows')
        card(cols[1],'Adjusted backtest MAPE',number(metrics['expected_mape'],1,'%'),'Lower is better · same held-out rows')
        card(cols[2],'Held-out observations',str(metrics['rows']),'Historical validation, not 2027 accuracy')
        st.write('')
        if held.empty:st.info('No held-out history is available for these programs.');return
        a,b=st.columns([1.15,1],gap='large')
        with panel(a,'Held-out actual starts vs estimates'):chart(demand.backtest_chart(held),'demand_backtest')
        with panel(b,'Program realization factors'):
            import forecast_model as fm
            model=fm.fit(data['wafer_start_history']);model=model[model.program.isin(programs)]
            fig=go.Figure(go.Bar(y=model.program_name,x=100*model.realization_rate,orientation='h',marker_color=ec.BLUE))
            ec.style(fig,320);fig.add_vline(x=100,line_dash='dot',line_color=ec.AMBER);fig.update_xaxes(title='Actual starts / requested starts (%)',rangemode='tozero');fig.update_yaxes(autorange='reversed');chart(fig,'demand_realization')
            st.caption('2027 factors use all history through 2026Q3, pooled across fabs. Backtest predictions use only the earlier training period.')
        with st.expander('Held-out source rows'):
            st.dataframe(held.round(2),hide_index=True,width='stretch')
            st.caption('MAPE is the mean absolute percentage error across individual fab/program/quarter records; the chart sums those records for visual comparison. All data is synthetic, with limited history. Backtest performance does not establish future forecasting accuracy.')


def scenario_page():
    if ss.get('_scenario_scope')!=(site,quarter):
        ss['_scenario_scope']=(site,quarter);ss['review']=None
        ss['scenario_first']=quarter;ss['scenario_last']=quarter
    a,b,c=st.columns([.9,1.5,.9])
    request_site=a.selectbox('Fab receiving the change',['FAB_A','FAB_B'],format_func=rm.SITE_LABELS.get,key='scenario_site') if site=='Both' else site
    if site!='Both':a.markdown('**Fab receiving the change**');a.write(rm.SITE_LABELS[site])
    eligible=[p for p in programs if p in C.QUALIFIED[request_site]]
    if not eligible:st.info('Select a program qualified at the fab receiving the demand change.');return
    if ss.get('scenario_program') not in eligible:ss['scenario_program']='DRAM_B' if 'DRAM_B' in eligible else eligible[0]
    program=b.selectbox('Program',eligible,format_func=ec.PROGRAM_SHORT.get,key='scenario_program')
    request_priority=c.selectbox('Request priority',['NORMAL','HOT'],key='scenario_priority')
    timing=st.radio('Scenario timing',['Quarter range','Week range'],horizontal=True,key='scenario_timing')
    first_week=last_week=None
    if timing=='Quarter range':
        a,b=st.columns(2)
        if 'scenario_first' not in ss:ss['scenario_first']=quarter
        if 'scenario_last' not in ss:ss['scenario_last']=quarter
        first_q=a.selectbox('From quarter',C.PLAN_QUARTERS,key='scenario_first')
        last_q=b.selectbox('Through quarter',C.PLAN_QUARTERS,key='scenario_last')
        period_label=first_q+'–'+last_q
    else:
        calendar=weekly.calendar().set_index('week_start');week_options=list(calendar.index)
        q_start=pd.Timestamp(int(quarter[:4]),3*(int(quarter[-1])-1)+1,1);q_end=q_start+pd.DateOffset(months=3)
        defaults=calendar[(calendar.period_start>=q_start)&(calendar.period_end<=q_end)]
        a,b=st.columns(2)
        first_week=a.selectbox('Start week',week_options,index=week_options.index(defaults.index[0]),format_func=lambda w:calendar.loc[w,'label'],key='scenario_start_week')
        last_week=b.selectbox('End week',week_options,index=week_options.index(defaults.index[-1]),format_func=lambda w:calendar.loc[w,'label'],key='scenario_end_week')
        first_q=weekly.quarter_of(calendar.loc[first_week,'period_start'])
        last_q=weekly.quarter_of(calendar.loc[last_week,'period_end']-pd.Timedelta(days=1))
        period_label=calendar.loc[first_week,'label']+' through '+calendar.loc[last_week,'label']+' 2027'
        st.caption('Weekly capacity screen: quarterly rates and equipment releases are prorated by calendar days. Opening and closing weeks are partial.')
    a,b=st.columns(2)
    mode=a.radio('Demand input',['Percentage change','Target WSPW'],horizontal=True,key='demand_mode')
    pct=b.number_input('Starts change (%)',min_value=-100.,max_value=100.,value=10.,step=5.,key='scenario_pct') if mode=='Percentage change' else 0.
    target=b.number_input('Target starts / week',min_value=0.,value=float(state['wspw'][request_site][program][C.PLAN_QUARTERS.index(first_q)]),step=25.,key='target_'+request_site+'_'+program+'_'+first_q) if mode=='Target WSPW' else None
    if target is not None:st.caption('The target weekly rate applies to this program throughout the selected request window.')
    milestone=st.text_input('Urgent experiment / milestone (optional)',key='hot_milestone') if request_priority=='HOT' else ''
    recovery=st.checkbox('Test equipment availability recovery',key='recovery_enabled')
    recovery_group,points=None,0.
    if recovery:
        a,b=st.columns(2)
        recovery_group=a.selectbox('Equipment to recover',[g for g in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g]>0],format_func=rm.GROUP_LABELS.get,key='recovery_group_'+request_site)
        points=b.number_input('Recovery (percentage points)',min_value=.5,max_value=15.,value=5.,step=.5,key='recovery_points')
    signature=(site,request_site,program,pct,target,timing,first_q,last_q,first_week,last_week,ceiling,recovery_group,points,request_priority,milestone)
    if st.button('Compare scenario',type='primary'):
        try:
            if timing=='Week range':
                review=weekly.run(data,request_site,program,pct,first_week,last_week,ceiling,recovery_group,points,target_wspw=target)
                ss['inspect_week']=first_week
            else:
                review=rm.run_review(data,request_site,program,pct,first_q,last_q,ceiling,recovery_group,points,target_wspw=target)
                review['timing_mode']='quarters';ss['inspect_quarter']=first_q
            review.update(request_priority=request_priority,milestone=milestone)
            ss['review']=review;ss['review_signature']=signature
        except ValueError as exc:ss['review']=None;st.error(str(exc))
    review=ss.get('review')
    if not review:return
    if ss.get('review_signature')!=signature:st.info('Inputs changed. Select Compare scenario to update the results.');return
    include_other=st.checkbox('Include the other fab in results',value=False,key='scenario_include_other') if site=='Both' else False
    result_scope='Both' if include_other else request_site
    st.subheader(rm.SITE_LABELS[request_site]+' · '+ec.PROGRAM_SHORT[program]+' · '+request_priority+' request')
    st.caption(period_label+' · '+('Combined fab comparison' if include_other else 'Results: '+rm.SITE_LABELS[request_site]+' only')+' · baseline retains all program demand')
    if timing=='Week range':
        inspect=st.selectbox('Inspect week',week_options,format_func=lambda w:calendar.loc[w,'label'],key='inspect_week')
        before=weekly.summary(review,result_scope,inspect);after=weekly.summary(review,result_scope,inspect,after=True)
        comparison=review['weekly_comparison'];current=rm.select_scope(comparison[comparison.week_start==inspect],result_scope)
        inspection_label=calendar.loc[inspect,'label']
        if not first_week<=inspect<=last_week:st.caption('This inspection week is outside the requested change window.')
    else:
        inspect=st.selectbox('Inspect quarter',C.PLAN_QUARTERS,key='inspect_quarter')
        brief=rm.decision_brief(review,inspect,result_scope);before,after=brief['before'],brief['after']
        comparison=review['comparison'];current=rm.select_scope(comparison,result_scope,inspect);inspection_label=inspect
    cols=st.columns(4)
    card(cols[0],'Starts / week',f"{before['starts']:,.0f} → {after['starts']:,.0f}",('Weekly-equivalent rates · ' if timing=='Week range' else '')+rm.SITE_LABELS[result_scope])
    card(cols[1],'At-risk equipment',f"{before['at_risk']} → {after['at_risk']}",f'{ceiling*100:.0f}% limit · named below','alert')
    card(cols[2],'Peak scenario load',number(after['worst_load'],1,'%'),rm.SITE_LABELS[after['worst_site']]+' · '+rm.GROUP_LABELS[after['worst_group']],'danger' if after['worst_load']>100 else 'alert')
    card(cols[3],'Over capacity',str(after['over_capacity']),'Fab / equipment groups >100%')
    risky=current[current.loading_pct_scenario.ge(100*ceiling)|current.loading_pct_scenario.isna()].sort_values('loading_pct_scenario',ascending=False)
    names=[rm.SITE_LABELS[r.site]+' · '+rm.GROUP_LABELS[r.tool_group]+' ('+number(r.loading_pct_base,1,'%')+' → '+number(r.loading_pct_scenario,1,'%')+')' for r in risky.itertuples()]
    st.markdown('**At-risk equipment:** '+('; '.join(names) if names else 'None above the planning limit.'))
    if after['over_capacity']:st.caption('The request exceeds available equipment capacity. Compare demand timing, qualified resources or availability recovery before committing the load.')
    elif after['at_risk']:st.caption('The named equipment exceeds the planning limit. Review the operating response for the requested window.')
    else:st.caption('The request fits this aggregate capacity screen. Confirm operating assumptions and experiment timing.')
    if request_priority=='HOT':
        st.info('HOT request · '+(milestone if milestone else 'Urgency / need-by milestone not supplied')+'. Priority changes dispatch order, not equipment hours. Decide which normal work may move; queue and delivery impacts need a scheduling or simulation check.')
    else:st.caption('NORMAL request · existing 2027 plan has no HOT/NORMAL split. This request label does not infer a priority mix for the baseline.')
    a,b=st.columns([1.3,1],gap='large')
    with panel(a,inspection_label+' · baseline vs scenario'):chart(weekly.equipment_chart(current,result_scope,ceiling),'scenario_loads')
    with panel(b,'Risk across the planning horizon'):
        chart(weekly.risk_chart(review,result_scope,ceiling) if timing=='Week range' else ec.scenario_risks(review,result_scope,ceiling),'scenario_risks')
    with st.expander('All equipment results & applied assumptions'):
        frame=table_view(current)
        columns=['Site','Equipment','loading_pct_base','loading_pct_scenario','change_pp','status']
        if timing=='Quarter range':columns.insert(-1,'tool_gap_scenario')
        st.dataframe(frame[columns].rename(columns={'loading_pct_base':'Baseline loading (%)','loading_pct_scenario':'Scenario loading (%)','change_pp':'Change (pp)','tool_gap_scenario':'Additional tools at limit','status':'Status'}).round(1),hide_index=True,width='stretch')
        st.write(review['applied'])
        st.caption('Availability recovery applies only during the requested window. Weekly loads use exact day overlap with each quarterly baseline and equipment release. This is aggregate loading; lot sequence, queue time, expediting benefit and due dates are not predicted. Request priority is a declared scenario attribute, not inferred from historical lots.')
    exported=rm.select_scope(comparison,result_scope).assign(request_site=request_site,request_program=program,request_priority=request_priority,timing_mode=review['timing_mode'],requested_window=period_label,milestone=milestone)
    st.download_button('Download scenario comparison',export_table(exported),f'{result_scope}_scenario_comparison.csv','text/csv')

def lot_page():
    title('Lot flow & cycle time',pd.Timestamp(first).strftime('%d %b')+'–'+(pd.Timestamp(last)+pd.Timedelta(days=6)).strftime('%d %b %Y')+' · lot review')
    if not programs:st.info('Select at least one program in the sidebar.');return
    p=em.execution(data,site,programs,priority,first,last,include_short)
    tabs=st.tabs(['Cycle-time drivers','Lot explorer'])
    with tabs[0]:
        cols=st.columns(4);parts=p['breakdown'];total=parts.sum(min_count=1)
        share=100*(parts.queue_days+parts.hold_days)/total if total>0 else float('nan')
        card(cols[0],'Completed cycle-time lots',str(len(p['cycles'])),'Short loops included' if include_short else 'Full-flow programs')
        card(cols[1],'90th percentile cycle time',number(p['p90_days'],1,' d'),'Completed-lot distribution')
        card(cols[2],'Queue + hold share',number(share,1,'%'),'Share of completed-lot cycle time')
        card(cols[3],'WIP at period end',str(len(p['wip_lots'])),'Lots active at selected cutoff')
        st.write('')
        a,b=st.columns(2,gap='large')
        with panel(a,'Cycle-time components by program'):chart(pc.cycle_breakdown(p,'program'),'lot_components')
        with panel(b,'HOT vs NORMAL cycle time'):chart(pc.priority_compare(p),'priority_compare')
        a,b=st.columns(2,gap='large')
        with panel(a,'Cycle-time spread'):chart(pc.cycle_distribution(p),'ct_distribution')
        with panel(b,'Age of WIP at period end'):chart(pc.wip_age(p),'wip_age')
        st.caption('Queue and hold shares describe elapsed time; engineering holds can be required development work. Priority comparisons show observed cohorts, not a causal improvement estimate.')
    with tabs[1]:
        population=st.radio('Lot population',['Completed in reporting period','WIP at period end','Started in reporting period'],horizontal=True,key='lot_population')
        if population=='Completed in reporting period':frame=p['completed'].copy()
        elif population=='WIP at period end':
            frame=p['wip_lots'].copy()
            frame[['end_time','wafers_out']]=None
        else:frame=p['lots'][p['lots'].started.ge(p['first'])&p['lots'].started.lt(p['cutoff'])].copy()
        frame['Program']=frame.program.map(ec.PROGRAM_SHORT);frame['Site']=frame.site.map(rm.SITE_LABELS)
        cols=['lot_id','Site','Program','priority','start_time','end_time','wafers_in','wafers_out']
        if population=='WIP at period end':cols+=['age_days','progress_pct']
        else:cols+=['cycle_time_days','process_days','queue_days','transport_days','hold_days']
        if population=='Started in reporting period':
            # Later completions must not appear as known at an earlier review cutoff.
            future=frame.ended.ge(p['cutoff'])|frame.ended.isna()
            frame.loc[future,['end_time','wafers_out','cycle_time_days',*em.COMPONENTS]]=None
        labels={'lot_id':'Lot ID','priority':'Priority','start_time':'Started','end_time':'Completed','wafers_in':'Input wafers','wafers_out':'Output wafers','cycle_time_days':'Cycle time (d)','process_days':'Processing (d)','queue_days':'Queue (d)','transport_days':'Transport (d)','hold_days':'Hold (d)','age_days':'Age at cutoff (d)','progress_pct':'Route progress (%)'}
        st.caption(f'{len(frame):,} lots · '+population.lower())
        if population=='WIP at period end':
            st.caption('Age is reconstructed at the selected cutoff. Route progress is available only at the final simulation snapshot.')
        st.dataframe(frame[cols].rename(columns=labels).round(2),hide_index=True,width='stretch')
        st.download_button('Download filtered lots',export_table(frame[cols],labels),'filtered_lots.csv','text/csv')


if page=='Fab performance':performance_page()
elif page=='Demand outlook':demand_page()
elif page=='Capacity & scenarios':
    title('Capacity & scenarios',quarter+' review · requested 2027 plan')
    tabs=st.tabs(['Capacity outlook','Scenario comparison'])
    with tabs[0]:capacity_outlook()
    with tabs[1]:scenario_page()
else:lot_page()
with st.expander('Data scope & model notes'):
    st.caption('Independent synthetic interview prototype. The source contains 5 program routes; there are no product/SKU, process-step event or individual-machine records in the fab simulation. Lot and priority are the finest execution detail. Quarter/week capacity is an aggregate screen, checked separately by fab and equipment group; it does not predict scenario cycle time, due dates, costs or ROI. Machine OEE uses a separate worked example or uploaded records.')
st.markdown('<div class="footer"><span>TD Fab · stakeholder review</span><span>Synthetic data · v5.0</span></div>',unsafe_allow_html=True)
