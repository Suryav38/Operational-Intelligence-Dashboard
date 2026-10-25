"""Stakeholder review: execution first, explicit cohorts, capacity and lot detail."""
from __future__ import annotations
import html
import os,json,hashlib
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import executive_charts as ec
import performance_charts as pc
import execution_model as em
import fab_demo as D
import launch_model as L
import launch_charts as LC
import dashboard_ai as DA
import planning_copilot as rag
import assistant as ai_errors
import scenario_requests as request_model
from dotenv import load_dotenv
import fab_capacity as fc
import fab_config as C
import fab_data as fd
import review_model as rm
import demand_outlook as demand
import weekly_scenarios as weekly
import dashboard_ui as ui
load_dotenv(fd.ROOT/'.env')

st.set_page_config(page_title='TD Fab | Performance & Planning',page_icon='◫',layout='wide')
st.markdown('<style>'+Path(__file__).with_name('dashboard.css').read_text()+'</style>',unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def load_data(fingerprint):return fd.load()
try:data=load_data(fd.fingerprint())
except Exception as exc:st.error(f'Project data could not be loaded: {exc}');st.stop()
ss=st.session_state

def card(col,label,value,sub='',tone=''):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value {tone}">{html.escape(str(value))}</div><div class="kpi-sub">{html.escape(sub)}</div></div>',unsafe_allow_html=True)
def number(value,decimals=1,suffix=''):
    return '—' if pd.isna(value) else f'{value:,.{decimals}f}'+suffix

def chart(fig,key):st.plotly_chart(fig,width='stretch',config={'displayModeBar':False,'scrollZoom':False},key=key)
def panel(col,name):
    box=col.container(border=True);box.subheader(name);return box

def title(name,period=''):
    st.markdown('<div class="topline"><div class="eyebrow">TD / STRATEGY & OPERATIONS</div><span class="badge">SYNTHETIC DEMO</span></div>',unsafe_allow_html=True)
    st.title(name)
    if period:st.caption(period)

@st.cache_data(show_spinner=False)
def cached_capacity(fingerprint,case,limit):
    return L.capacity(load_data(fingerprint),case,limit)

@st.cache_data(show_spinner=False)
def cached_execution(fingerprint,scope,selected,priority,first,last,include_short):
    return em.execution(load_data(fingerprint),scope,list(selected),priority,first,last,include_short)
def table_view(frame):
    out=frame.copy();out['Site']=out.site.map(rm.SITE_LABELS);out['Equipment']=out.tool_group.map(rm.GROUP_LABELS);return out

def export_table(frame,labels=None):
    return frame.rename(columns=labels or {}).to_csv(index=False)

def secret(name,default=''):
    try:return str(st.secrets.get(name) or os.getenv(name,default)).strip()
    except Exception:return str(os.getenv(name,default)).strip()

@st.cache_resource(show_spinner=False)
def note_index(fingerprint,api_key,semantic):
    index=rag.DocumentIndex(fd.DOCS)
    if semantic and api_key:
        try:
            from openai import OpenAI
            index.enable_semantic(OpenAI(api_key=api_key,timeout=25,max_retries=1))
        except Exception:index.notice='Semantic retrieval unavailable; keyword evidence is used.'
    return index


def ai_panel(label,key,context,examples):
    with st.expander(label):
        st.caption('Reads selected values, deployed project tables and dated notes. Model tools calculate results; notes provide context.')
        choice=st.selectbox('Example question',examples,key=key+'_example')
        question=st.text_area('Question',choice,height=85,max_chars=4000,key=key+'_question_'+str(examples.index(choice)))
        semantic=st.checkbox('Use semantic + keyword note retrieval',value=True,key=key+'_semantic')
        api_key=secret('OPENAI_API_KEY');model=secret('OPENAI_MODEL','gpt-4o-mini')
        context=rag.json_safe(context)
        fingerprint=rag.documents_fingerprint(fd.DOCS)
        stamp=hashlib.sha256(json.dumps([context,question,semantic,fingerprint],sort_keys=True,default=str).encode()).hexdigest()
        a,b=st.columns(2)
        if a.button('Ask AI',disabled=not api_key,key=key+'_ask',type='primary'):
            try:
                with st.spinner('Reading this view and retrieving evidence…'):
                    index=note_index(fingerprint,api_key,semantic)
                    answer=DA.ViewCopilot(data,index,context).answer(question,api_key,model)
                ss[key+'_answer']={'stamp':stamp,'result':answer}
            except Exception as exc:st.error(ai_errors.explain_error(exc))
        if b.button('Preview evidence',key=key+'_preview'):
            index=rag.DocumentIndex(fd.DOCS)
            ss[key+'_evidence']={'stamp':stamp,'context':context,'documents':index.search(question,4)}
        if not api_key:st.caption('Live AI needs OPENAI_API_KEY in app settings → Secrets. Evidence preview works without a key.')
        saved=ss.get(key+'_answer')
        if saved and saved['stamp']==stamp:
            out=saved['result'];st.markdown(out['answer']);st.caption('AI interpretation · '+out['retrieval_mode']+' · verify the cited evidence')
            if out.get('retrieval_notice'):st.caption(out['retrieval_notice'])
            with st.expander('Sources and model operations'):
                for source in out['sources']:
                    st.markdown('**['+source['id']+'] '+source.get('source',source.get('title','Evidence'))+'**')
                    if source['kind']=='document':st.text(source['text'])
                    else:
                        if source.get('sql'):st.code(source['sql'],language='sql')
                        st.json(json.dumps(source['data'],default=str),expanded=False)
                st.write(out['trace'])
        elif saved:st.caption('The filters or question changed. Ask again to refresh the answer.')
        preview=ss.get(key+'_evidence')
        if preview and preview['stamp']==stamp:
            with st.expander('Current values and matching notes',expanded=True):
                st.json(json.dumps(context,default=str),expanded=False)
                for doc in preview['documents']:st.markdown('**'+doc['source']+'**');st.text(doc['text'])
        with st.expander('Where the AI gets data'):
            st.caption('GitHub stores the versioned CSV/SQLite files and text notes. Streamlit loads that deployed copy. The assistant receives these exact filtered values, can make read-only table queries, retrieve note passages and call capacity calculations. It does not read screenshot pixels or fetch live GitHub data. Precomputed DES results are labelled separately from live capacity calculations.')


def route_page():
    catalog=data['program_catalog'];eligible=[p for p in catalog.program if p in programs]
    if not eligible:st.info('Select at least one qualified program above.');return
    if ss.get('route_program') not in eligible:ss['route_program']=eligible[0]
    program=st.selectbox('Inspect route',eligible,format_func=D.NAMES.get,key='route_program')
    definition=catalog[catalog.program==program].iloc[0];route=data['route_steps'].query('program==@program')
    a,b,c,d=st.columns(4)
    card(a,'Technology family',definition.technology_family,'HBM belongs to DRAM technology')
    card(b,'Mask layers',str(int(definition.mask_layers)),definition.route_version)
    card(c,'Ordered route operations',str(len(route)),'Repeated visits to shared equipment')
    card(d,'Qualified fabs',definition.qualified_fabs.replace('FAB_','Fab '),'Pilot release '+D.LAUNCH_DATE if program==D.PILOT else 'Established illustrative route')
    st.caption('Synthetic layer modules for workload and re-entrant flow. These are declared engineering assumptions, not a validated device traveler.')
    layers=st.slider('Layers to inspect',1,int(definition.mask_layers),(1,min(8,int(definition.mask_layers))),key='route_layers_'+program)
    selected=route[route.layer.between(*layers)]
    a,b=st.columns(2,gap='large')
    with panel(a,'Full-route equipment visits'):chart(LC.route_visits(route),'route_visits')
    with panel(b,'Ordered visits · selected layers'):chart(LC.ordered_route(selected),'ordered_route')
    with st.expander('Ordered traveler and processing assumptions'):
        st.dataframe(selected[['step_no','layer','tool_group','recipe_id','min_per_wafer','batch_size_lots','batch_run_hours','sample_rate','qualified_fabs']].rename(columns={'step_no':'Operation','layer':'Layer','tool_group':'Equipment group','recipe_id':'Compatible recipe','min_per_wafer':'Equivalent min/wafer','batch_size_lots':'Max lots/batch','batch_run_hours':'Batch run (h)','sample_rate':'Sample fraction','qualified_fabs':'Qualified fabs'}),hide_index=True,width='stretch')
        st.caption('Furnace capacity uses full-batch equivalent minutes/wafer. In the DES, compatible lots share one six-hour machine cycle, maximum three lots and a two-hour formation wait. Partial batches use a full run; observed batch fill therefore affects flow. Other groups are pools of identical tools. Metrology is sampled.')
    st.download_button('Download ordered program route',export_table(route),program+'_ordered_route.csv','text/csv')


def program_introduction():
    if site=='FAB_B':st.info('The EUV pilot is qualified only at Fab A. Select Fab A or Combined to inspect this study.');return
    case=pilot_case;summary=L.study_summary(data,case)
    st.subheader('Fab A · '+D.NAMES[D.PILOT]+' · '+D.CASES[case])
    st.caption('Precomputed DES · pilot release 17 May 2027 · week 7 of a 16-week study · '+str(summary['replications'])+' matched random-seed runs · existing release plans retained')
    a,b,c,d=st.columns(4)
    card(a,'Post-release average WIP',number(summary['baseline_post_launch_wip'],0)+' → '+number(summary['post_launch_wip'],0),'Input-wafer equivalents · baseline → case')
    card(b,'Existing-lot completion shift',number(summary['mean_delay_days'],2,' d'),'Mean additional time vs matched baseline')
    card(c,'90th percentile shift',number(summary['p90_delay_days'],2,' d'),'Matched initial-WIP lots')
    card(d,'Pilot lots completed',number(summary['pilot_completed_lots_mean'],1),'Mean per run by study end')
    st.caption(f"Matched existing lot instances across five runs: {summary['matched_lots']:,} / {summary['initial_lots']:,}; unfinished at study end: {summary['unfinished_lots']:,}. Completion shifts exclude unfinished comparisons and are not customer lateness.")
    _,base=cached_capacity(fd.fingerprint(),'baseline',ceiling)
    _,proposed=cached_capacity(fd.fingerprint(),case,ceiling)
    current=rm.select_scope(rm.compare_tables(base,proposed,ceiling),'FAB_A',quarter)
    study_tabs=st.tabs(['WIP & existing work','Demand & capacity'],key='pilot_study_view',on_change='rerun')
    if study_tabs[0].open:
        with study_tabs[0]:
            a,b=st.columns(2,gap='large')
            with panel(a,'WIP · matched simulation cases'):chart(LC.wip(data,case),'launch_wip')
            with panel(b,'Pilot impact detail'):
                detail=st.selectbox('Impact view',['Existing-lot completion shift','Equipment queues','Pilot starts & completions'],key='pilot_impact_view')
                if detail=='Existing-lot completion shift':chart(LC.existing_delays(data,case),'launch_delays')
                elif detail=='Equipment queues':chart(LC.queues(data,case),'launch_queues')
                else:chart(LC.pilot_flow(data,case),'pilot_flow')
            st.caption('Fixed Fab A study cohort · all existing programs. Completion shifts are relative to matched baseline lots, not customer lateness.')
    else:
        with study_tabs[1]:
            a,b=st.columns(2,gap='large')
            with panel(a,quarter+' · route-aware capacity change'):chart(weekly.equipment_chart(current,'FAB_A',ceiling),'pilot_capacity')
            with panel(b,'Weekly loading · exact release dates'):
                group=st.selectbox('Equipment group',C.TOOL_GROUP_ORDER,format_func=rm.GROUP_LABELS.get,key='pilot_weekly_equipment')
                chart(LC.weekly_loading(data,case,group,ceiling),'pilot_weekly_loading')
            risk=current[current.loading_pct_scenario.ge(ceiling*100)|current.loading_pct_scenario.isna()].copy()
            st.dataframe(table_view(risk)[['Site','Equipment','loading_pct_base','loading_pct_scenario','tool_gap_base','tool_gap_scenario']].rename(columns={'loading_pct_base':'Pre-pilot loading (%)','loading_pct_scenario':'Case loading (%)','tool_gap_base':'Pre-pilot tool gap','tool_gap_scenario':'Case tool gap'}).round(1),hide_index=True,width='stretch')
            st.caption('Quarter charts average the pilot ramp; weekly loading uses exact release dates. This capacity screen does not rerun the DES.')
    with st.expander('Starting WIP, paired lots and study assumptions'):
        initial=data['launch_initial_wip'];seed=st.selectbox('Inspect simulation seed',sorted(initial.seed.unique()),key='launch_seed')
        st.dataframe(initial[initial.seed==seed].head(150),hide_index=True,width='stretch')
        st.caption('Every case uses the same initial WIP, release calendar and per-lot random inputs for a seed. Initial state includes route position, queue/processing/hold state and remaining in-flight time. Lines show replicate means; shading is the selected case’s minimum–maximum range, not a real-fab prediction interval. HOT and NORMAL change non-preemptive dispatch order, not equipment hours. Equipment requirements exclude prices and ROI. EUV routes are not qualified at Fab B. See MODEL_GUIDE.md.')
        st.download_button('Download paired existing-lot comparisons',export_table(data['launch_lot_comparison'].query('case==@case')),'paired_existing_lots.csv','text/csv')
    ai_panel('Ask AI about this program introduction','ai_launch',dict(view='Pilot introduction DES',site='FAB_A',quarter=quarter,pilot_case=case,planning_limit=ceiling,summary=summary,
        displayed_mean_delay_days=number(summary['mean_delay_days'],2),displayed_p90_delay_days=number(summary['p90_delay_days'],2),displayed_average_wip=number(summary['post_launch_wip'],0),capacity_comparison=current.round(3).to_dict('records'),new_program=D.NAMES[D.PILOT],release_date=D.LAUNCH_DATE),[
        'What changes for existing lots when this pilot program is introduced, and which equipment needs attention?',
        'Does the slower pilot ramp remove the capacity constraint, or only reduce the WIP impact?',
        'Which route and qualification assumptions prevent moving the EUV pilot into Fab B?'])


st.sidebar.markdown('<div class="brand"><div class="brand-mark">TD</div><div><div class="brand-title">Fab performance & planning</div><div class="brand-sub">Stakeholder review</div></div></div>',unsafe_allow_html=True)
PAGES=list(ui.PAGE_VIEWS)
if ss.get('page') not in PAGES:ss['page']=PAGES[0]
page=st.sidebar.radio('Workspace',PAGES,key='page')
st.sidebar.caption('52 historical weeks · 2027 requested plan')
st.sidebar.caption('Independent synthetic planning demo · v8.0')
title('2027 demand outlook' if page=='Demand outlook' else page)
filter_slot=st.container()
tab_key='view_'+str(PAGES.index(page))
view_memory=ss.setdefault('_view_memory',{})
if tab_key not in ss and page in view_memory:ss[tab_key]=view_memory[page]
tabs=st.tabs(ui.PAGE_VIEWS[page],key=tab_key,on_change='rerun')
active_index=next((i for i,t in enumerate(tabs) if t.open),0)
active_view=ui.PAGE_VIEWS[page][active_index]
view_memory[page]=active_view
with filter_slot:
    filters=ui.render_filters(data,page,active_view)
site=filters['site'];programs=filters['programs'];family=filters['family']
quarter=filters['quarter'];pilot_case=filters['pilot_case'];ceiling=filters['ceiling']
priority=filters['priority'];first=filters['first'];last=filters['last'];include_short=filters['include_short']
state=capacity=None
if page=='Capacity & scenarios':state,capacity=cached_capacity(fd.fingerprint(),pilot_case,ceiling)


def overview(p):
    cols=st.columns(4)
    card(cols[0],'Completions / week',number(p['avg_outs'],0),'Selected programs & priority')
    card(cols[1],'Cycle time',number(p['cycle_days'],1,' d'),'All selected programs' if include_short else 'Full-flow completed lots')
    card(cols[2],'Average WIP',number(p['avg_wip'],0),'Input-wafer equivalents')
    card(cols[3],'Line yield',number(p['yield_pct'],2,'%'),'Completed-cohort wafer survival')
    st.write('')
    a,b=st.columns(2,gap='large')
    with panel(a,'Weekly wafer flow'):chart(pc.weekly_flow(p),'weekly_flow')
    with panel(b,'Operating trend'):
        trend=st.radio('Trend metric',['WIP','Cycle time'],horizontal=True,key='overview_trend')
        chart(pc.weekly_trend(p,'wip_wafers' if trend=='WIP' else 'avg_cycle_time_days'),'weekly_operating_trend')
    with st.expander('Program breakdown & scorecard'):
        frame=rm.select_scope(data['weekly_program_flow'],site)
        frame=frame[frame.program.isin(programs)&frame.week_start.between(first,last)]
        if priority!='All priorities':frame=frame[frame.priority==priority]
        fig=go.Figure()
        for idx,program in enumerate(C.PROGRAM_ORDER):
            if program not in programs:continue
            values=frame[frame.program==program].groupby('week_start').wafer_starts.sum()
            fig.add_bar(x=values.index,y=values,name=D.NAMES[program],marker_color=ec.PROGRAM_COLORS[idx])
        ec.style(fig,340);fig.update_layout(barmode='stack',legend=dict(y=-.3),margin=dict(b=110))
        fig.update_xaxes(type='date',tickformat='%b %Y');fig.update_yaxes(title='Starts / week')
        chart(fig,'cycle_program')
        summary=p['completed'].groupby(['site','program']).agg(completed_lots=('lot_id','size'),wafers_in=('wafers_in','sum'),wafers_out=('wafers_out','sum'),mean_cycle_days=('cycle_time_days','mean')).reset_index()
        summary['Yield (%)']=100*summary.wafers_out/summary.wafers_in
        summary['Site']=summary.site.map(rm.SITE_LABELS);summary['Program']=summary.program.map(ec.PROGRAM_SHORT)
        st.dataframe(summary[['Site','Program','completed_lots','wafers_out','mean_cycle_days','Yield (%)']].rename(columns={'completed_lots':'Completed lots','wafers_out':'Wafers out','mean_cycle_days':'Mean cycle time (d)'}).round(2),hide_index=True,width='stretch')
    with st.expander('Metric definitions'):
        st.caption('Completions, WIP and yield follow all selected programs. The short-loop option applies to cycle-time charts. Cycle time is completed-lot weighted; yield uses input and output from the same completed cohorts. WIP is reconstructed at each week end from lot timestamps and original input quantities. Empty cohorts display —.')
        st.caption('Scorecard cycle times include every listed program. Little’s Law is a diagnostic; finite-window completed cohorts and weekly WIP snapshots have different boundary effects.')
    st.download_button('Download filtered weekly performance',export_table(p['weekly']),'filtered_weekly_performance.csv','text/csv')
    context=dict(view='Historical fab performance',site=site,quarter=quarter,planning_limit=ceiling,period=[str(p['first']),str(p['cutoff'])],
        programs=programs,priority=priority,include_short_in_cycle_time=include_short,
        displayed_metrics=dict(completions_per_week=number(p['avg_outs'],0),average_wip=number(p['avg_wip'],0),cycle_days=number(p['cycle_days'],1),yield_pct=number(p['yield_pct'],2)),
        metrics=dict(completions_per_week=round(p['avg_outs'],1),cycle_days=round(p['cycle_days'],2),average_wip_input_wafers=round(p['avg_wip'],1),wafer_survival_yield_pct=round(p['yield_pct'],2)),
        weekly=p['weekly'].round(3).to_dict('records'),source='52-week synthetic lot-flow history; no future scenario applied')
    ai_panel('Ask AI about this weekly performance','ai_performance',context,[
        'What are the main changes in weekly wafer flow, WIP and cycle time, and which operating notes add context?',
        'Which DRAM programs drive the workload, and what evidence would support investigating a queue increase?',
        'How does the declared 75% DRAM technology mix relate to these historical starts?'])



def equipment_page():
    st.caption('Shared equipment activity · all programs and priorities · selected site and reporting weeks')
    available=[g for g in C.TOOL_GROUP_ORDER if any(C.INSTALLED_TOOLS[s][g] for s in ec.sites(site))]
    selected=st.multiselect('Equipment groups to investigate',available,default=[g for g in rm.FOCUS_GROUPS if g in available],format_func=rm.GROUP_LABELS.get,key='health_groups_'+site)
    if not selected:st.info('Select equipment to compare states and queues.');return
    frame=em.equipment_health(data,site,selected,first,last)
    fleet=rm.select_scope(data['installed_tools'],site)
    cols=st.columns(4);worst=frame.sort_values('loading_pct',ascending=False).iloc[0]
    card(cols[0],'Installed fleet',number(fleet.tools.sum(),0),'All groups · fleet as of 27 Sep 2026')
    card(cols[1],'Peak production loading',number(worst.loading_pct,1,'%'),rm.SITE_LABELS[worst.site]+' · '+rm.GROUP_LABELS[worst.tool_group])
    card(cols[2],'Longest equipment queue',number(frame.avg_queue_h.max(),2,' h'),'Selected equipment · mean per visit')
    furnace=em.equipment_health(data,site,['FURN'],first,last)
    card(cols[3],'Furnace batch fill',number(furnace.batch_lots.sum()/furnace.batch_count.sum() if furnace.batch_count.sum() else float('nan'),2),'All selected fabs · maximum 3 lots')
    a,b=st.columns(2,gap='large')
    with panel(a,'Equipment states'):chart(pc.equipment_states(frame),'equipment_states')
    with panel(b,'Weekly equipment queues'):chart(pc.queue_trend(data,site,selected,first,last),'equipment_queues')
    with st.expander('Installed tools & equipment detail'):
        chart(LC.inventory(data,site),'fleet_inventory')
        st.dataframe(table_view(fleet)[['Site','Equipment','tools']].rename(columns={'tools':'Installed tools'}),hide_index=True,width='stretch')
        view=table_view(frame)
        cols=['Site','Equipment','tools','productive_pct','standby_pct','engineering_pct','sched_down_pct','unsched_down_pct','loading_pct','avg_queue_h']
        st.dataframe(view[cols].rename(columns={'tools':'Tools','productive_pct':'Productive (%)','standby_pct':'Standby (%)','engineering_pct':'Engineering (%)','sched_down_pct':'Scheduled down (%)','unsched_down_pct':'Unscheduled down (%)','loading_pct':'Production loading (%)','avg_queue_h':'Queue / visit (h)'}).round(2),hide_index=True,width='stretch')
        st.caption('State percentages use calendar hours; production loading excludes engineering and downtime from available hours. Engineering work is not automatically waste. Identical qualified tools are pooled. Furnace recipes share one batch cycle. Zero EUV tools at Fab B means EUV routes are not qualified there.')
    st.download_button('Download equipment health',export_table(frame),'selected_equipment_health.csv','text/csv')
    ai_panel('Ask AI about equipment and operating notes','ai_equipment',dict(view='Historical equipment health',site=site,quarter=quarter,planning_limit=ceiling,
        period=[first,last],equipment_groups=selected,filter_scope='fab, equipment and reporting weeks; shared states retain all programs/priorities',
        inventory=fleet.to_dict('records'),equipment=frame.round(3).to_dict('records')),[
        'Which equipment has the longest queues, and what dated operating records should we investigate?',
        'What happened to Fab A EUV availability around the scanner recovery event?',
        'Which equipment quantities and route qualifications matter if we consolidate work between fabs?'])

def attainment_page():
    st.caption('Historical requested and actual starts · selected quarters, site and programs')
    frame=em.historical_attainment(data,site,programs)
    lo,hi=ss['attainment_period'];frame=frame[frame.quarter.between(lo,hi)]
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


def performance_page():
    # Only the active workspace tab builds charts and calculates its cohorts.
    if active_index==0:
        if not programs:st.info('Select at least one program above.')
        else:
            p=cached_execution(fd.fingerprint(),site,tuple(programs),priority,first,last,include_short)
            overview(p)
    elif active_index==1:equipment_page()
    elif active_index==2:attainment_page()
    else:route_page()


def capacity_outlook():
    m=rm.capacity_summary(capacity,state,site,quarter,ceiling);cols=st.columns(4)
    card(cols[0],'Wafer starts / week',number(m['starts'],0),'All programs in selected fab(s)')
    card(cols[1],'At-risk equipment',f"{m['at_risk']} / {m['groups']}",f'At or above {ceiling*100:.0f}%','alert' if m['at_risk'] else '')
    card(cols[2],'Peak equipment load',number(m['worst_load'],1,'%'),rm.SITE_LABELS[m['worst_site']]+' · '+rm.GROUP_LABELS[m['worst_group']],'danger' if m['worst_load']>100 else 'alert')
    card(cols[3],'Additional tools at limit',str(int(rm.select_scope(capacity,site,quarter).tool_gap.sum())),'Model requirement · before qualification / cost review')
    st.caption('Summary cards retain all equipment and all programs in the selected fab(s). Filters below focus the charts.')
    c1,c2=st.columns([2,1])
    default_groups=[g for g in rm.FOCUS_GROUPS if g in set(capacity.tool_group)]
    ss.setdefault('capacity_groups',default_groups)
    groups=c1.multiselect('Equipment groups',C.TOOL_GROUP_ORDER,format_func=rm.GROUP_LABELS.get,key='capacity_groups')
    status=c2.selectbox('Equipment status',['All equipment','Above planning limit','Over capacity','Within planning limit'],key='capacity_risk')
    display=rm.select_scope(capacity,site)
    selected_q=display[display.quarter.eq(quarter)&display.tool_group.isin(groups)]
    if status=='Above planning limit':selected_q=selected_q[selected_q.loading_pct.ge(ceiling*100)|selected_q.loading_pct.isna()]
    elif status=='Over capacity':selected_q=selected_q[selected_q.loading_pct.gt(100)|selected_q.loading_pct.isna()]
    elif status=='Within planning limit':selected_q=selected_q[selected_q.loading_pct.lt(ceiling*100)]
    pairs=set(zip(selected_q.site,selected_q.tool_group))
    display=display[[pair in pairs for pair in zip(display.site,display.tool_group)]]
    if display.empty:st.info('No equipment matches these chart filters. Summary cards still show the complete fab workload.')
    else:
        a,b=st.columns([1.2,1],gap='large')
        with panel(a,'Loading across quarters'):chart(ec.loading_heatmap(display,site,quarter,ceiling,True),'capacity_heatmap')
        with panel(b,quarter+' · equipment detail'):
            detail=st.selectbox('Detail view',['Equipment loading','Program contribution','Requested starts'],key='capacity_detail')
            if detail=='Equipment loading':chart(ec.equipment_compare(display,site,quarter,ceiling,True),'equipment_comparison')
            elif detail=='Program contribution':chart(LC.workload(data,pilot_case,site,quarter,selected_q,ceiling,programs,True),'load_drivers')
            else:chart(ec.starts_mix(state,site,quarter,programs),'starts_mix')
    st.caption(f'Green <{ceiling*100:.0f}% · amber {ceiling*100:.0f}–100% · red >100%. Program contribution retains other programs as a separate segment.')
    with st.expander('Equipment gaps, releases & calculation details'):
        current=table_view(rm.select_scope(capacity,site,quarter));current['Status']=current.loading_pct.map(lambda x:rm.risk_label(x,ceiling))
        st.dataframe(current[['Site','Equipment','tools','tools_needed_at_ceiling','tool_gap','loading_pct','required_h_wk','available_h_wk','Status']].rename(columns={'tools':'Installed / released','tools_needed_at_ceiling':'Needed at limit','tool_gap':'Additional tools','loading_pct':'Loading (%)','required_h_wk':'Required h/week','available_h_wk':'Available h/week'}).round(1),hide_index=True,width='stretch')
        arrivals=rm.select_scope(data['tool_arrivals'],site);st.dataframe(arrivals,hide_index=True,width='stretch')
        choices=list(zip(current.site,current.tool_group))
        selected=st.selectbox('Trace equipment',choices,format_func=lambda k:rm.SITE_LABELS[k[0]]+' · '+rm.GROUP_LABELS[k[1]],key='trace_equipment')
        st.dataframe(L.workload(data,pilot_case,*selected,quarter).round(2),hide_index=True,width='stretch')
        st.caption('Required h/week = Σ starts × passes × minutes/pass × sampling ÷ 60. Available h/week = resources × 168 × production availability. This is an aggregate capacity screen; capital costs and ROI are not included.')
    st.download_button('Download filtered equipment review',export_table(display.assign(planning_limit_pct=ceiling*100,demand_basis='requested')),f'{site}_2027_capacity_review.csv','text/csv')
    ai_panel('Ask AI about this capacity plan','ai_capacity',dict(view='2027 capacity outlook',site=site,quarter=quarter,pilot_case=pilot_case,planning_limit=ceiling,
        program_focus=programs,summary=m,displayed_starts_per_week=number(m['starts'],0),equipment=rm.select_scope(capacity,site,quarter).round(3).to_dict('records')),[
        'Which equipment constrains the requested DRAM-heavy plan, and what routes or notes explain the workload?',
        'If one EUV tool becomes unavailable at Fab A during this quarter, what happens to capacity?',
        'Could qualified DRAM Node A work move to Fab B, and would that create another constraint?'])



def demand_page():
    if not programs:st.info('Select at least one program above.');return
    frame=demand.plan(data,site,programs,pilot_case)
    comparison=demand.comparison_frame(data,site,programs,pilot_case)
    if frame.empty:st.info('No 2027 demand for the selected fab and programs.');return
    totals=demand.quarterly(frame);changes=demand.comparison_changes(comparison,quarter)
    lo,hi=filters['demand_period']
    visible=comparison[comparison.quarter.between(lo,hi)]
    plan_visible=frame[frame.quarter.between(lo,hi)]
    if active_index==0:
        current=totals.loc[quarter];cols=st.columns(4)
        card(cols[0],'Requested starts / week',number(current.requested_wspw,0),quarter+' · selected programs')
        card(cols[1],'Expected starts / week',number(current.expected_wspw,0),quarter+' · historical realization model')
        card(cols[2],'Year-on-year request',number(changes['yoy_pct'],1,'%'),changes['prior_quarter']+' → '+quarter)
        card(cols[3],'Quarter-on-quarter request',number(changes['qoq_pct'],1,'%'),changes['qoq_quarter']+' → '+quarter)
        a,b=st.columns([1.25,1],gap='large')
        with panel(a,'Demand comparison'):
            chart_view=st.selectbox('Chart view',['Historical → outlook','2026 vs 2027','Requested vs expected'],key='demand_chart_view')
            if chart_view=='2026 vs 2027':
                fig=demand.year_comparison_chart(visible,chart_filters=False)
                quarters=[f'Q{q}' for q in range(1,5) if any(f'{y}Q{q}' in set(visible.quarter) for y in (2026,2027))]
                fig.update_xaxes(categoryorder='array',categoryarray=quarters)
                if quarters:
                    indices=[int(q[1])-1 for q in quarters]
                    fig.update_xaxes(range=[min(indices)-.5,max(indices)+.5])
            else:
                if chart_view=='Requested vs expected':
                    rows=visible[visible.quarter.str.startswith('2027')]
                    if rows.empty:st.info('Choose a chart period that includes 2027 to compare requests and estimates.')
                    fig=demand.history_outlook_chart(rows,chart_filters=False)
                    fig.data[1].visible=False
                    positions=[demand.COMPARISON_QUARTERS.index(q) for q in C.PLAN_QUARTERS if lo<=q<=hi]
                    fig.update_xaxes(range=[min(positions)-.5,max(positions)+.5] if positions else [3.5,7.5])
                else:
                    fig=demand.history_outlook_chart(visible,chart_filters=False)
                    positions=[demand.COMPARISON_QUARTERS.index(q) for q in demand.COMPARISON_QUARTERS if lo<=q<=hi]
                    fig.update_xaxes(range=[min(positions)-.5,max(positions)+.5])
            fig.update_layout(height=390,legend=dict(y=-.28),margin=dict(t=20,b=110))
            chart(fig,'demand_main')
        with panel(b,'2027 demand mix'):
            breakdown=st.selectbox('Break down demand by',list(demand.BREAKDOWNS),key='demand_breakdown')
            if pilot_case!='baseline':ss['demand_measure']='Requested plan'
            measure=st.radio('Chart basis',list(demand.MEASURES),horizontal=True,key='demand_measure',disabled=pilot_case!='baseline')
            if plan_visible.empty:st.info('Include a 2027 quarter in Chart period to see the planned mix.')
            else:
                fig=demand.mix_chart(plan_visible,breakdown,measure)
                fig.update_xaxes(categoryorder='array',categoryarray=[q for q in C.PLAN_QUARTERS if lo<=q<=hi])
                fig.update_layout(height=335)
                chart(fig,'demand_mix')
        st.caption('KPI cards use '+quarter+'; both charts follow the selected chart period and programs.')
        if lo<='2026Q4' and hi>='2027Q1':
            st.caption('2026Q4 → 2027Q1 requested: '+number(changes['prior_q4_requested'],0)+' → '+number(changes['next_q1_requested'],0)+' WSPW ('+number(changes['transition_pct'],1,'%)')+'. Q4 is a planning reference, not an actual.')
        if pilot_case!='baseline':st.caption('Pilot expected starts remain unknown because the new program has no realization history.')
    elif active_index==1:
        st.caption('2027 detail · requested and expected start rates · selected site and programs')
        if pilot_case!='baseline':ss['demand_detail_measure']='Requested plan'
        measure=st.radio('Detail basis',list(demand.MEASURES),horizontal=True,key='demand_detail_measure',disabled=pilot_case!='baseline')
        if plan_visible.empty:st.info('Include a 2027 quarter in Chart period to see planning detail.')
        else:
            a,b=st.columns([1,1.3],gap='large')
            with panel(a,'Demand by fab'):chart(demand.fab_chart(plan_visible,measure),'demand_fabs')
            with panel(b,'Quarterly ramp · fab & program'):chart(demand.ramp_chart(plan_visible,measure,quarter),'demand_ramp')
            view=plan_visible.copy();view['Fab']=view.site.map(rm.SITE_LABELS);view['Program']=view.program.map(ec.PROGRAM_SHORT)
            st.dataframe(view[['quarter','Fab','family','Program','route_technology','requested_wspw','expected_wspw','realization_rate']].rename(columns={'quarter':'Quarter','family':'Family','route_technology':'Route technology','requested_wspw':'Requested WSPW','expected_wspw':'Expected WSPW','realization_rate':'Realization factor'}).round(2),hide_index=True,width='stretch')
    else:
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
    if active_index!=2:
        with st.expander('Definitions & assumptions'):
            st.caption('WSPW = wafer starts per week. Quarter values are average weekly rates, not total quarterly wafers. Requested is the supplied plan; expected = requested × a program realization factor fitted to history through 2026Q3, pooled across fabs. Capacity uses requested demand.')
            st.caption('Historical actuals end at 2026Q3. The hatched/diamond 2026Q4 reference is 2,950 WSPW for the full network; fab/program shares are assumed from baseline 2027Q1. It does not train the model. Future actuals and unsupported pilot estimates remain missing.')
            st.caption('Mix assumption: 75% DRAM technology including HBM, 15% NAND, 10% experiments. This is wafer-start mix, not revenue. Route technology means EUV inclusion, not wafer diameter, grade or SKU.')
        with st.popover('Export data'):
            st.download_button('Download 2027 demand outlook',export_table(plan_visible.assign(program_name=plan_visible.program.map(ec.PROGRAM_SHORT))),'2027_demand_outlook.csv','text/csv')
            st.download_button('Download 2026–2027 comparison',export_table(visible),'2026_2027_demand_comparison.csv','text/csv')


def scenario_page():
    operation=st.selectbox('Test a disruption or response',request_model.OPERATIONS,key='scenario_operation')
    if ss.get('_scenario_scope')!=(site,quarter):
        ss['_scenario_scope']=(site,quarter);ss['review']=None;ss['scenario_first']=quarter;ss['scenario_last']=quarter
    a,b,c=st.columns([.9,1.5,.9])
    request_site=a.selectbox('Fab receiving the change',list(C.SITES),format_func=rm.SITE_LABELS.get,key='scenario_site') if site=='Both' else site
    if site!='Both':a.markdown('**Fab receiving the change**');a.write(rm.SITE_LABELS[site])
    to_site='FAB_B' if request_site=='FAB_A' else 'FAB_A';outage_group='EUV';unavailable=1
    if operation=='Tool unavailability':
        available=[g for g in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g]>0]
        outage_group=b.selectbox('Unavailable equipment',available,format_func=rm.GROUP_LABELS.get,key='outage_group_'+request_site)
        program='DRAM_A'
    else:
        eligible=[p for p in programs if p in C.QUALIFIED[request_site] and (operation!='Qualified load transfer' or p in C.QUALIFIED[to_site])]
        if not eligible:st.info('The selected routes are not qualified for this action. EUV routes cannot transfer to Fab B. Pilot ramp changes use the Program introduction cases.');return
        if ss.get('scenario_program') not in eligible:ss['scenario_program']='DRAM_B' if 'DRAM_B' in eligible else eligible[0]
        program=b.selectbox('Program',eligible,format_func=ec.PROGRAM_SHORT.get,key='scenario_program')
    request_priority=c.selectbox('Request priority',['NORMAL','HOT'],key='scenario_priority',disabled=operation=='Tool unavailability')
    timing=st.radio('Scenario timing',['Quarter range','Week range'],horizontal=True,key='scenario_timing')
    first_week=last_week=None
    if timing=='Quarter range':
        a,b=st.columns(2)
        ss.setdefault('scenario_first',quarter);ss.setdefault('scenario_last',quarter)
        first_q=a.selectbox('From quarter',C.PLAN_QUARTERS,key='scenario_first');last_q=b.selectbox('Through quarter',C.PLAN_QUARTERS,key='scenario_last')
        period_label=first_q+'–'+last_q;first_period,last_period=first_q,last_q
    else:
        calendar=weekly.calendar().set_index('week_start');week_options=list(calendar.index)
        q_start=pd.Timestamp(int(quarter[:4]),3*(int(quarter[-1])-1)+1,1);q_end=q_start+pd.DateOffset(months=3)
        defaults=calendar[(calendar.period_start>=q_start)&(calendar.period_end<=q_end)];a,b=st.columns(2)
        first_week=a.selectbox('Start week',week_options,index=week_options.index(defaults.index[0]),format_func=lambda w:calendar.loc[w,'label'],key='scenario_start_week')
        last_week=b.selectbox('End week',week_options,index=week_options.index(defaults.index[-1]),format_func=lambda w:calendar.loc[w,'label'],key='scenario_end_week')
        first_q=weekly.quarter_of(calendar.loc[first_week,'period_start']);last_q=weekly.quarter_of(calendar.loc[last_week,'period_end']-pd.Timedelta(days=1))
        period_label=calendar.loc[first_week,'label']+' through '+calendar.loc[last_week,'label']+' 2027';first_period,last_period=first_week,last_week
        st.caption('Weekly capacity screen: exact calendar overlap with quarterly rates, the selected pilot ramp and equipment releases.')
    pct=0.;target=None
    if operation=='Tool unavailability':
        unavailable=int(st.number_input('Number of unavailable tools',min_value=1,max_value=100,value=1,step=1,key='unavailable_tools'))
    elif operation=='Qualified load transfer':
        st.caption('Qualified transfer: '+rm.SITE_LABELS[request_site]+' → '+rm.SITE_LABELS[to_site]+'. Both fabs are evaluated; total network wafer starts are conserved.')
        pct=st.number_input('Share of source program to transfer (%)',min_value=0.,max_value=100.,value=10.,step=5.,key='transfer_pct')
    else:
        a,b=st.columns(2);mode=a.radio('Demand input',['Percentage change','Target WSPW'],horizontal=True,key='demand_mode')
        pct=b.number_input('Starts change (%)',min_value=-100.,max_value=100.,value=10.,step=5.,key='scenario_pct') if mode=='Percentage change' else 0.
        target=b.number_input('Target starts / week',min_value=0.,value=float(state['wspw'][request_site][program][C.PLAN_QUARTERS.index(first_q)]),step=25.,key='target_'+request_site+'_'+program+'_'+first_q) if mode=='Target WSPW' else None
    milestone=st.text_input('Urgent experiment / milestone (optional)',key='hot_milestone') if request_priority=='HOT' and operation!='Tool unavailability' else ''
    recovery=st.checkbox('Test equipment availability recovery',key='recovery_enabled') if operation=='Demand growth' else False
    recovery_group,points=None,0.
    if recovery:
        a,b=st.columns(2);recovery_group=a.selectbox('Equipment to recover',[g for g in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g]>0],format_func=rm.GROUP_LABELS.get,key='recovery_group_'+request_site)
        points=b.number_input('Recovery (percentage points)',min_value=.5,max_value=15.,value=5.,step=.5,key='recovery_points')
    signature=(site,pilot_case,operation,request_site,program,pct,target,timing,first_period,last_period,ceiling,recovery_group,points,request_priority,milestone,outage_group,unavailable)
    if st.button('Compare scenario',type='primary'):
        try:
            review=request_model.run(data,operation,request_site,program,pct,outage_group,unavailable,to_site,first_period,last_period,
                'weeks' if timing=='Week range' else 'quarters',ceiling,recovery_group,points,target,pilot_case)
            review.update(request_priority=request_priority,milestone=milestone);ss['review']=review;ss['review_signature']=signature
            ss['inspect_week' if timing=='Week range' else 'inspect_quarter']=first_period
        except ValueError as exc:ss['review']=None;st.error(str(exc))
    review=ss.get('review')
    if not review:return
    if ss.get('review_signature')!=signature:st.info('Inputs changed. Select Compare scenario to update the results.');return
    include_other=st.checkbox('Include the other fab in results',value=False,key='scenario_include_other') if site=='Both' and operation!='Qualified load transfer' else operation=='Qualified load transfer'
    result_scope='Both' if include_other else request_site
    request_name=rm.GROUP_LABELS[outage_group]+' unavailability' if operation=='Tool unavailability' else ec.PROGRAM_SHORT[program]+' · '+request_priority+' request'
    st.subheader(rm.SITE_LABELS[request_site]+(' → '+rm.SITE_LABELS[to_site] if operation=='Qualified load transfer' else '')+' · '+request_name)
    st.caption(period_label+' · '+('Combined fab comparison' if include_other else 'Results: '+rm.SITE_LABELS[request_site]+' only')+' · baseline: '+D.CASES[pilot_case])
    if timing=='Week range':
        inspect=st.selectbox('Inspect week',week_options,format_func=lambda w:calendar.loc[w,'label'],key='inspect_week')
        before=weekly.summary(review,result_scope,inspect);after=weekly.summary(review,result_scope,inspect,True)
        comparison=review['weekly_comparison'];current=rm.select_scope(comparison[comparison.week_start==inspect],result_scope);inspection_label=calendar.loc[inspect,'label']
        if not first_week<=inspect<=last_week:st.caption('This week is outside the requested change window.')
    else:
        inspect=st.selectbox('Inspect quarter',C.PLAN_QUARTERS,key='inspect_quarter')
        brief=rm.decision_brief(review,inspect,result_scope);before,after=brief['before'],brief['after']
        comparison=review['comparison'];current=rm.select_scope(comparison,result_scope,inspect);inspection_label=inspect
    a,b,c,d=st.columns(4)
    card(a,'Starts / week',f"{before['starts']:,.0f} → {after['starts']:,.0f}",('Weekly-equivalent rates · ' if timing=='Week range' else '')+rm.SITE_LABELS[result_scope])
    card(b,'At-risk equipment',f"{before['at_risk']} → {after['at_risk']}",f'{ceiling*100:.0f}% limit · named below','alert')
    card(c,'Peak scenario load','No capacity' if pd.isna(after['worst_load']) else number(after['worst_load'],1,'%'),rm.SITE_LABELS[after['worst_site']]+' · '+rm.GROUP_LABELS[after['worst_group']],'danger' if pd.isna(after['worst_load']) or after['worst_load']>100 else 'alert')
    card(d,'Over capacity',str(after['over_capacity']),'Fab / equipment groups >100% or no capacity')
    risky=current[current.loading_pct_scenario.ge(100*ceiling)|current.loading_pct_scenario.isna()].sort_values('loading_pct_scenario',ascending=False)
    fmt=lambda x:'No capacity' if pd.isna(x) else number(x,1,'%')
    names=[rm.SITE_LABELS[r.site]+' · '+rm.GROUP_LABELS[r.tool_group]+' ('+fmt(r.loading_pct_base)+' → '+fmt(r.loading_pct_scenario)+')' for r in risky.itertuples()]
    st.markdown('**At-risk equipment:** '+('; '.join(names) if names else 'None above the planning limit.'))
    if operation!='Tool unavailability' and request_priority=='HOT':st.info('HOT request · '+(milestone if milestone else 'Urgency / need-by milestone not supplied')+'. Priority changes dispatch order, not equipment hours. Queue and delivery effects require a matched flow simulation.')
    a,b=st.columns([1.3,1],gap='large')
    with panel(a,inspection_label+' · baseline vs scenario'):chart(weekly.equipment_chart(current,result_scope,ceiling),'scenario_loads')
    with panel(b,'Risk across the planning horizon'):chart(weekly.risk_chart(review,result_scope,ceiling) if timing=='Week range' else ec.scenario_risks(review,result_scope,ceiling),'scenario_risks')
    with st.expander('All equipment results & applied assumptions'):
        frame=table_view(current);st.dataframe(frame[['Site','Equipment','loading_pct_base','loading_pct_scenario','change_pp','status']].rename(columns={'loading_pct_base':'Baseline loading (%)','loading_pct_scenario':'Scenario loading (%)','change_pp':'Change (pp)','status':'Status'}).round(1),hide_index=True,width='stretch')
        st.write(review['applied']);st.caption('This live screen includes the selected pilot loading but does not rerun the DES. Qualification is enforced; transfers conserve network starts and may move a bottleneck. Tool loss applies only within the selected window. No price, ROI, queue or due-date prediction is made here.')
    exported=rm.select_scope(comparison,result_scope).assign(request_site=request_site,request_program=program,request_priority=request_priority,timing_mode=review['timing_mode'],requested_window=period_label,pilot_case=pilot_case,operation=operation)
    st.download_button('Download scenario comparison',export_table(exported),f'{result_scope}_scenario_comparison.csv','text/csv')
    ai_panel('Ask AI about this scenario','ai_scenario',dict(view='Live capacity scenario',site=result_scope,quarter=quarter,pilot_case=pilot_case,planning_limit=ceiling,
        request=dict(operation=operation,site=request_site,program=program,priority=request_priority,period=period_label),before=before,after=after,equipment=current.round(3).to_dict('records')),[
        'Which named equipment constrains this request, and which documented response should we examine?',
        'Can we consolidate qualified work into the other fab without simply moving the constraint?',
        'What is the difference between this capacity calculation and the pilot WIP/completion simulation?'])

def lot_page():
    if not programs:st.info('Select at least one program above.');return
    p=cached_execution(fd.fingerprint(),site,tuple(programs),priority,first,last,include_short)
    if active_index==0:
        cols=st.columns(4);parts=p['breakdown'];total=parts.sum(min_count=1)
        share=100*(parts.queue_days+parts.hold_days)/total if total>0 else float('nan')
        card(cols[0],'Completed cycle-time lots',str(len(p['cycles'])),'Short loops included' if include_short else 'Full-flow programs')
        card(cols[1],'90th percentile cycle time',number(p['p90_days'],1,' d'),'Completed-lot distribution')
        card(cols[2],'Queue + hold share',number(share,1,'%'),'Share of completed-lot cycle time')
        card(cols[3],'WIP at period end',str(len(p['wip_lots'])),'Lots active at selected cutoff')
        st.write('')
        a,b=st.columns(2,gap='large')
        with panel(a,'Cycle-time components by program'):chart(pc.cycle_breakdown(p,'program'),'lot_components')
        with panel(b,'Lot analysis'):
            detail=st.selectbox('Analysis view',['WIP aging','Cycle-time spread','HOT vs NORMAL'],key='lot_analysis')
            if detail=='WIP aging':chart(pc.wip_age(p),'wip_age')
            elif detail=='Cycle-time spread':chart(pc.cycle_distribution(p),'ct_distribution')
            else:chart(pc.priority_compare(p),'priority_compare')
        st.caption('Queue and hold shares describe elapsed time; engineering holds can be required development work. Priority comparisons show observed cohorts, not a causal improvement estimate.')
    else:
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
        count_before=len(frame)
        c1,c2,c3=st.columns([2,1,1])
        search=c1.text_input('Search lot ID',key='lot_search',placeholder='Enter all or part of a lot ID')
        metric='age_days' if population=='WIP at period end' else 'cycle_time_days'
        minimum=c2.number_input('Minimum '+('age' if metric=='age_days' else 'cycle time')+' (days)',min_value=0.,value=0.,step=1.,key='lot_min_days')
        maximum=c3.number_input('Maximum days (0 = any)',min_value=0.,value=0.,step=1.,key='lot_max_days')
        if search:frame=frame[frame.lot_id.astype(str).str.contains(search.strip(),case=False,regex=False)]
        if maximum and maximum<minimum:st.info('Maximum days must be at least the minimum.');return
        if minimum>0:frame=frame[frame[metric].ge(minimum)]
        if maximum>0:frame=frame[frame[metric].le(maximum)]
        if population=='Started in reporting period' and (minimum>0 or maximum>0):
            st.caption('Cycle-time limits include only lots completed by the reporting cutoff; unfinished cycle times remain unknown.')
        labels={'lot_id':'Lot ID','priority':'Priority','start_time':'Started','end_time':'Completed','wafers_in':'Input wafers','wafers_out':'Output wafers','cycle_time_days':'Cycle time (d)','process_days':'Processing (d)','queue_days':'Queue (d)','transport_days':'Transport (d)','hold_days':'Hold (d)','age_days':'Age at cutoff (d)','progress_pct':'Route progress (%)'}
        st.caption(f'{len(frame):,} of {count_before:,} lots · '+population.lower())
        if population=='WIP at period end':
            st.caption('Age is reconstructed at the selected cutoff. Route progress is available only at the final simulation snapshot.')
        st.dataframe(frame[cols].rename(columns=labels).round(2),hide_index=True,width='stretch')
        st.download_button('Download filtered lots',export_table(frame[cols],labels),'filtered_lots.csv','text/csv')


with tabs[active_index]:
    if page=='Fab performance':performance_page()
    elif page=='Demand outlook':demand_page()
    elif page=='Capacity & scenarios':
        if active_index==0:capacity_outlook()
        elif active_index==1:scenario_page()
        else:program_introduction()
    else:lot_page()
ui.remember_filters()
with st.expander('Data scope & model notes'):
    st.caption('Independent synthetic interview prototype. History is simulated through 27 Sep 2026. Calendar 2027 is a requested plan. Capacity calculates requested-route hours; the separate precomputed DES estimates WIP and matched-lot completion shifts. All data and notes are synthetic.')
st.markdown('<div class="footer"><span>TD Fab · stakeholder review</span><span>Synthetic data · v8.0</span></div>',unsafe_allow_html=True)
