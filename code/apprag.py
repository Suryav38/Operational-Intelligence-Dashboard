"""Stakeholder workspace: comparative charts, scoped scenarios and grounded AI."""
from __future__ import annotations
import html
import json
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import assistant as ai
import executive_charts as ec
import fab_capacity as fc
import fab_config as C
import fab_data as fd
import planning_copilot as rag
import review_model as rm

st.set_page_config(page_title='TD Planning | Stakeholder Review',page_icon='◫',layout='wide')
st.markdown('<style>'+Path(__file__).with_name('dashboard.css').read_text()+'</style>',unsafe_allow_html=True)
load_dotenv(fd.ROOT/'.env')

@st.cache_data(show_spinner=False)
def load_data():return fd.load()
try:data=load_data()
except Exception as exc:
    st.error(f'The project data could not be loaded: {exc}');st.stop()

def secret(name,default=''):
    try:return st.secrets.get(name) or os.getenv(name,default)
    except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):return os.getenv(name,default)
API_KEY=str(secret('OPENAI_API_KEY')).strip().strip('"')
MODEL=str(secret('OPENAI_MODEL','gpt-4o-mini')).strip()
ss=st.session_state

def card(col,label,value,sub='',tone=''):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div><div class="kpi-value {tone}">{html.escape(str(value))}</div><div class="kpi-sub">{html.escape(sub)}</div></div>',unsafe_allow_html=True)

def chart(fig,key):st.plotly_chart(fig,width='stretch',config={'displayModeBar':False},key=key)

def title(name,period):
    st.markdown('<div class="topline"><div class="eyebrow">TD / STRATEGY & OPERATIONS</div><span class="badge">SYNTHETIC DEMO</span></div>',unsafe_allow_html=True)
    st.title(name)
    st.caption(rm.SITE_LABELS[site]+' · '+period)

def panel(col,name):
    box=col.container(border=True);box.subheader(name);return box

def table_view(t):
    t=t.copy();t['Site']=t.site.map(rm.SITE_LABELS);t['Equipment']=t.tool_group.map(rm.GROUP_LABELS)
    return t

st.sidebar.markdown('<div class="brand"><div class="brand-mark">TD</div><div><div class="brand-title">Planning workspace</div><div class="brand-sub">Stakeholder review</div></div></div>',unsafe_allow_html=True)
page=st.sidebar.radio('Workspace',['Capacity plan','Scenario review','Fab performance','AI copilot'],key='page')
st.sidebar.divider()
site=st.sidebar.selectbox('Site',['Both','FAB_A','FAB_B'],format_func=rm.SITE_LABELS.get,key='site')
if page!='Fab performance':
    ss.setdefault('quarter',ss.get('_planning_quarter','2027Q2'))
    quarter=st.sidebar.selectbox('Review quarter',C.PLAN_QUARTERS,key='quarter')
    ss['_planning_quarter']=quarter
else:quarter=ss.get('_planning_quarter','2027Q2')
with st.sidebar.expander('Review assumptions'):
    ceiling=st.slider('Planning limit (%)',75,95,85,5,key='ceiling_pct',disabled=page=='Fab performance')/100
    st.caption('Loading uses available production hours. Combined capacity risks remain separate by fab and equipment group. All programs stay in the model.')
    st.caption('85% is a demo planning policy. Historical operation: 29 Jun–27 Sep 2026. Planning basis: requested starts.')
st.sidebar.divider()
st.sidebar.markdown('**AI copilot**')
st.sidebar.caption('Key configured' if API_KEY else 'Setup needed: OPENAI_API_KEY')
st.sidebar.caption('11 structured tables · 15 planning documents')
state=fc.base_state(data);table=fc.loading_table(state,ceiling)


def capacity_page():
    title('Capacity outlook',quarter+' review · 2027 plan')
    m=rm.capacity_summary(table,state,site,quarter,ceiling)
    cols=st.columns(4)
    card(cols[0],'Wafer starts / week',f"{m['starts']:,.0f}",'Requested plan')
    card(cols[1],'At-risk equipment',f"{m['at_risk']} / {m['groups']}",f'At or above {ceiling*100:.0f}%','alert' if m['at_risk'] else '')
    card(cols[2],'Peak equipment load',f"{m['worst_load']:.1f}%",rm.SITE_LABELS[m['worst_site']]+' · '+rm.GROUP_LABELS[m['worst_group']],'danger' if m['worst_load']>100 else 'alert')
    card(cols[3],'Over capacity',str(m['over_capacity']),'Fab / equipment groups >100%','danger' if m['over_capacity'] else '')
    st.write('')
    cols=st.columns([1.25,1],gap='large')
    with panel(cols[0],'Loading across quarters'):
        all_groups=st.toggle('Show all equipment groups',value=False,key='all_groups')
        chart(ec.loading_heatmap(table,site,quarter,ceiling,all_groups),'capacity_heatmap')
    with panel(cols[1],quarter+' · equipment comparison'):
        chart(ec.equipment_compare(table,site,quarter,ceiling),'equipment_comparison')
    st.caption(f'Green: below {ceiling*100:.0f}% · amber: {ceiling*100:.0f}%–100% · red: over capacity. Dashed line = planning limit. Dotted line = available capacity.')
    cols=st.columns(2,gap='large')
    with panel(cols[0],quarter+' · wafer starts by program'):
        chart(ec.starts_mix(state,site,quarter),'starts_mix')
    with panel(cols[1],'What consumes the equipment hours?'):
        chart(ec.workload_drivers(state,site,rm.select_scope(table,site,quarter),ceiling),'load_drivers')
    with st.expander('Exceptions, equipment releases & calculation details'):
        current=table_view(rm.select_scope(table,site,quarter))
        current['Status']=current.loading_pct.map(lambda x:rm.risk_label(x,ceiling))
        exceptions=current[current.Status!='Within planning limit']
        st.dataframe(exceptions[['Site','Equipment','loading_pct','required_h_wk','available_h_wk','Status']].round(1),hide_index=True,width='stretch')
        arrivals=data['tool_arrivals']
        if site!='Both':arrivals=arrivals[arrivals.site==site]
        st.dataframe(arrivals,hide_index=True,width='stretch')
        selected=current[['site','tool_group']].apply(tuple,axis=1).tolist()
        choice=st.selectbox('Trace equipment',selected,format_func=lambda k:rm.SITE_LABELS[k[0]]+' · '+rm.GROUP_LABELS[k[1]],key='trace_equipment')
        detail=rm.program_workload(state,*choice,quarter)
        st.dataframe(detail.round(2),hide_index=True,width='stretch')
        st.caption('Required h/week = Σ starts × passes × minutes/pass × sampling ÷ 60. Available h/week = resources × 168 × production availability. Resources released in a quarter contribute for that full quarter.')
    st.download_button('Download capacity review',rm.select_scope(table,site).assign(planning_limit_pct=ceiling*100,demand_basis='requested').to_csv(index=False),f'{site}_2027_capacity_review.csv','text/csv')


def copy_ai_draft():
    d=ai.validate_request(ai.StartRequest(**ss['ai_draft']))
    scope='Both' if ss.get('site')=='Both' else d.site
    ss['site'],ss['quarter']=scope,d.first_quarter
    ss['_planning_quarter']=d.first_quarter
    ss['_scenario_scope']=(scope,d.first_quarter)
    ss['scenario_site']=d.site
    ss['scenario_program'],ss['scenario_pct']=d.program,float(d.change_pct)
    ss['scenario_first'],ss['scenario_last']=d.first_quarter,d.last_quarter
    ss['recovery_enabled']=False;ss['review']=None


def scenario_page():
    title('Scenario comparison',quarter+' review · baseline vs proposed change')
    previous=ss.get('review')
    if ss.get('_scenario_scope')!=(site,quarter):
        ss['_scenario_scope']=(site,quarter)
        previous=None;ss['review']=None
        for k in ['scenario_site','scenario_program','scenario_pct','scenario_first','scenario_last','recovery_enabled']:
            ss.pop(k,None)
    ss.setdefault('scenario_site',previous['site'] if previous else ('FAB_A' if site=='Both' else site))
    request_site=ss['scenario_site'] if site=='Both' else site
    ss.setdefault('scenario_program',previous['program'] if previous else ('DRAM_B' if request_site=='FAB_A' else 'NAND_C'))
    ss.setdefault('scenario_pct',float(previous['pct']) if previous else 10.0)
    ss.setdefault('scenario_first',previous['first_quarter'] if previous else quarter)
    ss.setdefault('scenario_last',previous['last_quarter'] if previous else quarter)
    with st.container(border=True):
        st.subheader('AI request reader')
        note=st.text_area('Planning note',f'Increase DRAM Node B wafer starts by 10% at Fab A during {quarter} only.',height=75,key='planning_note')
        if st.button('Extract request',disabled=not API_KEY):
            try:
                with st.spinner('Reading the request…'):ss['ai_draft']=ai.extract_request(note,API_KEY,MODEL).model_dump()
                ss['ai_note_source']=note
            except Exception as exc:st.error(ai.explain_error(exc));ss.pop('ai_draft',None)
        if not API_KEY:st.caption('Enable live AI in Streamlit app settings → Secrets. Manual controls remain available.')
        if ss.get('ai_draft') and ss.get('ai_note_source')==note:
            d=ai.StartRequest(**ss['ai_draft'])
            st.dataframe(pd.DataFrame([{'Program':d.program,'Site':d.site,'Change %':d.change_pct,'From':d.first_quarter,'Through':d.last_quarter}]),hide_index=True,width='stretch')
            try:ai.validate_request(d);valid=True
            except ValueError as exc:valid=False;st.warning(str(exc))
            st.button('Copy reviewed request to controls',on_click=copy_ai_draft,disabled=not valid)
            with st.expander('Source quote & open questions'):st.write(d.source_quote);st.write(d.open_questions)
    if site=='Both':request_site=st.selectbox('Change at',list(C.SITES),format_func=rm.SITE_LABELS.get,key='scenario_site')
    if ss.get('scenario_program') not in C.QUALIFIED[request_site]:ss['scenario_program']='DRAM_A'
    cols=st.columns([1.4,1,1,1])
    program=cols[0].selectbox('Program',C.QUALIFIED[request_site],format_func=lambda p:ec.PROGRAM_SHORT[p],key='scenario_program')
    pct=cols[1].number_input('Starts change (%)',min_value=-100.,max_value=100.,step=5.,key='scenario_pct')
    first=cols[2].selectbox('From',C.PLAN_QUARTERS,key='scenario_first')
    last=cols[3].selectbox('Through',C.PLAN_QUARTERS,key='scenario_last')
    recovery=st.checkbox('Test availability recovery',key='recovery_enabled')
    recovery_group,points=None,0.
    if recovery:
        a,b=st.columns(2)
        recovery_group=a.selectbox('Equipment to recover',[g for g in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g]>0],format_func=rm.GROUP_LABELS.get,key='recovery_group')
        points=b.number_input('Recovery (percentage points)',min_value=.5,max_value=15.,value=5.,step=.5)
    signature=(site,request_site,program,pct,first,last,ceiling,recovery_group,points)
    if st.button('Compare scenario',type='primary'):
        try:
            ss['review']=rm.run_review(data,request_site,program,pct,first,last,ceiling,recovery_group,points)
            ss['review_signature']=signature;ss['inspect_quarter']=first;ss.pop('ai_explanation',None)
        except ValueError as exc:ss['review']=None;st.error(str(exc))
    review=ss.get('review')
    if not review:return
    if ss.get('review_signature')!=signature:st.info('Inputs changed. Select Compare scenario to update the results.');return
    inspect=st.selectbox('Inspect quarter',C.PLAN_QUARTERS,key='inspect_quarter')
    brief=rm.decision_brief(review,inspect,site);before,after=brief['before'],brief['after']
    cols=st.columns(4)
    card(cols[0],'Starts / week',f"{before['starts']:,.0f} → {after['starts']:,.0f}")
    card(cols[1],'At-risk equipment',f"{before['at_risk']} → {after['at_risk']}",f'{ceiling*100:.0f}% limit','alert')
    card(cols[2],'Peak load',f"{after['worst_load']:.1f}%",rm.SITE_LABELS[after['worst_site']]+' · '+after['worst_group'],'danger' if after['worst_load']>100 else 'alert')
    card(cols[3],'Over capacity',str(after['over_capacity']),'Fab / equipment groups','danger' if after['over_capacity'] else '')
    st.write('')
    a,b=st.columns([1.3,1],gap='large')
    with panel(a,inspect+' · baseline vs scenario'):chart(ec.scenario_loads(review,site,inspect,ceiling),'scenario_loads')
    with panel(b,'Risk across the planning horizon'):chart(ec.scenario_risks(review,site,ceiling),'scenario_risks')
    st.caption(brief['recommendation'])
    context=rm.ai_review_context(review,inspect,site);context_key=json.dumps(context,sort_keys=True)
    if st.button('Explain results with AI',disabled=not API_KEY,type='primary'):
        try:
            with st.spinner('Explaining the model results…'):out=ai.explain_scenario(context,API_KEY,MODEL)
            ss['ai_explanation']={'context_key':context_key,'result':out}
        except Exception as exc:ss.pop('ai_explanation',None);st.error(ai.explain_error(exc))
    saved=ss.get('ai_explanation')
    if saved and saved['context_key']==context_key:
        st.caption('AI-generated interpretation · verify against evidence')
        out=saved['result'];st.write(out['summary']);st.write(out['operating_implication'])
        for item in out['suggested_checks']:st.write('• '+item)
    elif saved:st.caption('Generate a new AI explanation for this quarter and scenario.')
    with st.expander('All equipment results & applied assumptions'):
        t=table_view(rm.select_scope(review['comparison'],site,inspect))
        st.dataframe(t[['Site','Equipment','loading_pct_base','loading_pct_scenario','change_pp','status']].round(1),hide_index=True,width='stretch')
        st.write(review['applied']);st.caption(brief['limit'])
    st.download_button('Download scenario comparison',rm.select_scope(review['comparison'],site).assign(planning_limit_pct=ceiling*100).to_csv(index=False),f'{site}_scenario_comparison.csv','text/csv')


def production_page():
    p=rm.production_metrics(data,site)
    title('Fab performance','29 Jun–27 Sep 2026 · 13 simulated weeks')
    cols=st.columns(4)
    card(cols[0],'Completions / week',f"{p['avg_outs']:,.0f}",'Average wafers out')
    card(cols[1],'Full-flow cycle time',f"{p['cycle_days']:.1f} d",'Completed-lot weighted')
    card(cols[2],'Average WIP',f"{p['wip']:,.0f}",'Input-wafer equivalents')
    card(cols[3],'Line yield',f"{p['line_yield']:.2f}%",'Completed-cohort wafer survival')
    st.write('')
    a,b=st.columns(2,gap='large');w=p['weekly'];dates=pd.to_datetime(w.week_start)
    with panel(a,'Weekly wafer flow'):
        fig=go.Figure()
        for col,name,color in [('wafer_starts','Starts',ec.BLUE),('wafer_outs','Completions',ec.TEAL)]:
            fig.add_bar(x=dates,y=w[col],name=name,marker_color=color,hovertemplate='%{x|%d %b}<br>%{y:,} wafers<extra>'+name+'</extra>')
        ec.style(fig,290);fig.update_layout(barmode='group');fig.update_yaxes(title='Wafers / week')
        fig.update_xaxes(tickmode='array',tickvals=dates.iloc[::2],tickformat='%d %b');chart(fig,'weekly_flow')
    with panel(b,'WIP trend by fab'):
        fig=go.Figure()
        for s in ec.sites(site):
            sw=data['sim_fab_weekly'].query('site==@s').sort_values('week_start')
            fig.add_scatter(x=pd.to_datetime(sw.week_start),y=sw.wip_wafers,name=rm.SITE_LABELS[s],mode='lines+markers',line=dict(color=ec.SITE_COLORS[s],width=2),marker=dict(size=4))
        if site=='Both':fig.add_scatter(x=dates,y=w.wip_wafers,name='Combined',mode='lines',line=dict(color='#24374f',dash='dot',width=2))
        ec.style(fig,290);fig.update_yaxes(title='Input-wafer equivalents',rangemode='tozero')
        fig.update_xaxes(tickmode='array',tickvals=dates.iloc[::3],tickformat='%d %b');chart(fig,'weekly_wip')
    a,b=st.columns(2,gap='large')
    with panel(a,'Cycle time by program and fab'):
        fig=go.Figure();programs=[p for p in C.PROGRAM_ORDER if p!='SHORT']
        for s in ec.sites(site):
            frame=p['by_program'].query('site==@s').set_index('program').reindex(programs)
            fig.add_bar(y=[ec.PROGRAM_SHORT[x] for x in programs],x=frame.cycle_days,orientation='h',name=rm.SITE_LABELS[s],marker_color=ec.SITE_COLORS[s],hovertemplate='%{y}<br>%{x:.1f} days<extra>'+rm.SITE_LABELS[s]+'</extra>')
        ec.style(fig,315);fig.update_layout(barmode='group');fig.update_xaxes(title='Average days / completed full-flow lot');fig.update_yaxes(autorange='reversed');chart(fig,'cycle_by_program')
    with panel(b,'Where cycle time is spent'):
        frame=p['full'].groupby('site')[['process_days','queue_days','transport_days','hold_days']].mean()
        fig=go.Figure()
        for col,name,color in [('process_days','Processing',ec.BLUE),('queue_days','Queues',ec.AMBER),('transport_days','Dispatch / transport','#a6b5cc'),('hold_days','Engineering holds',ec.TEAL)]:
            fig.add_bar(y=frame.index.map(rm.SITE_LABELS),x=frame[col],orientation='h',name=name,marker_color=color,hovertemplate='%{y}<br>%{x:.1f} days<extra>'+name+'</extra>')
        ec.style(fig,315);fig.update_layout(barmode='stack',margin=dict(l=0,r=12,t=12,b=90),legend=dict(y=-.32));fig.update_xaxes(title='Average days / completed full-flow lot');fig.update_yaxes(autorange='reversed');chart(fig,'cycle_breakdown')
    with st.expander('Equipment queues, definitions & source rows'):
        q=table_view(rm.select_scope(data['sim_tool_group_q3_2026'],site)).sort_values('avg_queue_h',ascending=False)
        st.dataframe(q[['Site','Equipment','loading_pct','avg_queue_h']].round(2),hide_index=True,width='stretch')
        st.caption('Combined cycle time is weighted by completed full-flow lots. Yield uses output and input wafers from the same completed cohorts. WIP sums aligned weekly snapshots and uses each active lot’s original input quantity. Short loops are excluded from cycle-time KPIs. These historical results do not change when a future scenario is evaluated.')
    st.download_button('Download weekly performance',w.to_csv(index=False),f'{site}_Q3_2026_performance.csv','text/csv')


@st.cache_resource(show_spinner=False)
def document_index(fingerprint,api_key,semantic):
    index=rag.DocumentIndex(fd.DOCS)
    if semantic:
        try:
            from openai import OpenAI
            index.enable_semantic(OpenAI(api_key=api_key,timeout=25,max_retries=1))
        except Exception:index.notice='Semantic indexing unavailable; keyword retrieval is being used.'
    return index


def copilot_page():
    title('Ask the planning copilot',quarter+' dashboard context · tables + planning documents')
    available=ss.get('review')
    if available and (available['ceiling']!=ceiling or (site!='Both' and available['site']!=site)):available=None
    if available is None:ss['copilot_use_scenario']=False
    use=st.checkbox('Use latest reviewed scenario',disabled=available is None,key='copilot_use_scenario')
    active=available if use and available else None
    snapshot=rag.json_safe(rag.dashboard_snapshot(data,site,quarter,ceiling,active))
    index=rag.DocumentIndex(fd.DOCS)
    st.caption(f"{len(data)} structured tables · {len({c['source'] for c in index.chunks})} planning documents · "+('reviewed scenario' if active else 'requested-start baseline'))
    examples=['Why is the most constrained equipment at risk, and what do the planning notes add?',
              'Compare the two fabs: which constraints and documented actions matter most?',
              'Which programs drive etch workload, and what mitigations are documented?',
              'What do operating metrics and engineering notes suggest we investigate first?',
              'Can NAND workload move to another qualified fab, and what happens to capacity?',
              'How do historical requested starts compare with actual starts?',
              'Which process-change notices affect etch or deposition, and what needs reconciliation?']
    chosen=st.selectbox('Example question',examples,key='copilot_example')
    question=st.text_area('Question',chosen,height=90,max_chars=4000,key='copilot_question_'+str(examples.index(chosen)))
    with st.expander('AI connection & retrieval settings'):
        semantic=st.checkbox('Use semantic document retrieval',value=True,key='semantic_retrieval')
        st.caption('Add OPENAI_API_KEY and optionally OPENAI_MODEL to Streamlit app settings → Secrets. Semantic retrieval combines embeddings and keyword matches; fallback is labeled. Structured tables are queried separately.')
    context_key=json.dumps({'snapshot':snapshot,'question':question,'semantic':semantic,'docs':rag.documents_fingerprint(fd.DOCS)},sort_keys=True)
    a,b=st.columns(2)
    if a.button('Ask AI',type='primary',disabled=not API_KEY,width='stretch'):
        try:
            with st.spinner('Reading the dashboard and retrieving evidence…'):
                idx=document_index(rag.documents_fingerprint(fd.DOCS),API_KEY,semantic)
                result=rag.Copilot(data,idx,site,quarter,ceiling,active).answer(question,API_KEY,MODEL)
            ss['copilot_answer']={'context_key':context_key,'result':result}
        except Exception as exc:ss.pop('copilot_answer',None);st.error(ai.explain_error(exc))
    if b.button('Preview evidence',width='stretch'):
        ss['copilot_preview']={'context_key':context_key,'docs':index.search(question,4)}
    if not API_KEY:st.info('Add your API key to enable live answers. Evidence preview works without a key.')
    saved=ss.get('copilot_answer')
    if saved and saved['context_key']==context_key:
        out=saved['result'];st.subheader('AI insight');st.markdown(out['answer'])
        st.caption('AI-generated · '+out['retrieval_mode']+' · verify against sources')
        if out.get('retrieval_notice'):st.caption(out['retrieval_notice'])
        with st.expander(f"Evidence · {len(out['sources'])} sources"):
            selected=st.selectbox('Source',range(len(out['sources'])),format_func=lambda i:'['+out['sources'][i]['id']+'] '+out['sources'][i].get('source',out['sources'][i].get('title','')),key='copilot_source')
            source=out['sources'][selected]
            if source['kind']=='document':st.text(source['text'])
            else:
                if source.get('sql'):st.code(source['sql'],language='sql')
                st.json(source['data'],expanded=False)
        with st.expander('What the AI read and calculated'):
            for step in out['trace']:st.write(step['step']+' · '+', '.join(step.get('sources',[])))
        st.download_button('Download answer and evidence',json.dumps(out,indent=2,default=str),f'{site}_{quarter}_AI_evidence.json','application/json')
    elif saved:st.caption('Context changed. Ask AI again to refresh its evidence.')
    preview=ss.get('copilot_preview')
    with st.expander('Dashboard & document preview',expanded=bool(preview and preview['context_key']==context_key)):
        t=table_view(pd.DataFrame(snapshot['capacity']))
        st.dataframe(t[['Site','Equipment','loading_pct','required_h_wk','available_h_wk']].round(1),hide_index=True,width='stretch')
        if preview and preview['context_key']==context_key:
            for doc in preview['docs']:st.markdown('**'+doc['source']+'**');st.text(doc['text'])


if page=='Capacity plan':capacity_page()
elif page=='Scenario review':scenario_page()
elif page=='Fab performance':production_page()
else:copilot_page()
with st.expander('Model notes'):
    st.caption('Synthetic, aggregate quarterly model. Capacity is checked per site and equipment group. No lot-level due-date prediction, real-time status, savings or ROI is claimed. Data and document retrieval support the AI; source notes do not automatically change the baseline.')
st.markdown('<div class="footer"><span>TD Planning · stakeholder review</span><span>Synthetic data · v3.0</span></div>',unsafe_allow_html=True)
