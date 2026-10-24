"""Focused capacity, scenario and historical fab-performance workspace."""
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
import fab_capacity as fc
import fab_config as C
import fab_data as fd
import review_model as rm

st.set_page_config(page_title='TD Planning | Capacity & Operations', page_icon='◫', layout='wide')
st.markdown('<style>' + Path(__file__).with_name('dashboard.css').read_text() + '</style>', unsafe_allow_html=True)
load_dotenv(fd.ROOT / '.env')
BLUE, TEAL, AMBER, RED, INK, MUTED = '#185adb', '#27877f', '#b97520', '#bd4343', '#14243b', '#64748b'
PALETTE = [BLUE, '#4797bd', TEAL, '#899ac0', '#afb8c6']

@st.cache_data(show_spinner=False)
def load_data():
    return fd.load()

try:
    data = load_data()
except (FileNotFoundError, pd.errors.ParserError) as exc:
    st.error(f'The project data could not be loaded: {exc}')
    st.stop()


def secret(name, default=''):
    try:
        return st.secrets.get(name) or os.getenv(name, default)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv(name, default)

API_KEY = str(secret('OPENAI_API_KEY')).strip().strip('"')
MODEL = str(secret('OPENAI_MODEL', 'gpt-4o-mini')).strip()
ss = st.session_state


def card(col, label, value, sub, tone=''):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
                 f'<div class="kpi-value {tone}">{html.escape(str(value))}</div>'
                 f'<div class="kpi-sub">{html.escape(sub)}</div></div>', unsafe_allow_html=True)


def decision(label, text):
    st.markdown(f'<div class="decision"><div class="decision-label">{html.escape(label)}</div>'
                f'<div class="decision-text">{html.escape(text)}</div></div>', unsafe_allow_html=True)


def style(fig, height=270):
    fig.update_layout(template='plotly_white', height=height, font=dict(family='Arial, sans-serif', size=11, color=INK),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=12, t=18, b=8),
                      legend=dict(orientation='h', y=-.20, x=0, font=dict(size=10)),
                      hoverlabel=dict(bgcolor='white', font_size=12))
    fig.update_xaxes(gridcolor='#e7edf4', zeroline=False)
    fig.update_yaxes(gridcolor='#e7edf4', zeroline=False)
    return fig


def chart(fig, key):
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'responsive': True}, key=key)


def title(name, subtitle, historical=False):
    snapshot = pd.to_datetime(data['sim_fab_weekly']['week_start'].max()).date() + pd.Timedelta(days=6)
    st.markdown('<div class="topline"><div><div class="eyebrow">TECHNOLOGY DEVELOPMENT / IE STRATEGY</div>'
                f'<div class="period">Baseline snapshot · {snapshot:%d %b %Y}</div></div>'
                f'<span class="badge">SYNTHETIC DEMO · {"HISTORICAL" if historical else "PLANNING"}</span></div>',
                unsafe_allow_html=True)
    st.title(name)
    st.markdown(f'<div class="scope-line">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def legend(ceiling):
    st.markdown(f'<div class="legend"><span><i class="dot" style="background:#dcece7"></i>Below {ceiling*100:.0f}%</span>'
                f'<span><i class="dot" style="background:#f5e3c3"></i>{ceiling*100:.0f}%–100%</span>'
                '<span><i class="dot" style="background:#f4d7d7"></i>Above 100%</span></div>', unsafe_allow_html=True)


def load_heatmap(table, site, quarter, groups, ceiling):
    t = rm.select_scope(table, site)
    p = t.pivot(index='tool_group', columns='quarter', values='loading_pct')
    p = p.reindex([g for g in groups if g in p.index]).reindex(columns=C.PLAN_QUARTERS)
    z = p.map(lambda v: 2 if pd.isna(v) or v > 100 else (1 if v >= ceiling*100 else 0))
    text = p.map(lambda v: 'No tool' if pd.isna(v) else f'{v:.1f}%')
    fig = go.Figure(go.Heatmap(z=z.values, x=p.columns, y=[rm.GROUP_LABELS[g] for g in p.index],
                              text=text.values, texttemplate='%{text}', textfont=dict(size=13, color=INK),
                              zmin=0, zmax=2, showscale=False, xgap=4, ygap=4,
                              colorscale=[[0,'#dcece7'],[.2499,'#dcece7'],[.25,'#f5e3c3'],
                                          [.7499,'#f5e3c3'],[.75,'#f4d7d7'],[1,'#f4d7d7']],
                              hovertemplate='%{y} · %{x}<br>Loading %{text}<extra></extra>'))
    qi = C.PLAN_QUARTERS.index(quarter)
    fig.add_shape(type='rect', xref='x', yref='paper', x0=qi-.49, x1=qi+.49, y0=0, y1=1,
                  line=dict(color=BLUE, width=2), fillcolor='rgba(0,0,0,0)')
    style(fig, max(205, len(p)*37+70))
    fig.update_xaxes(side='top', showgrid=False)
    fig.update_yaxes(autorange='reversed', showgrid=False)
    return fig


def readable_capacity(table, site, quarter, ceiling):
    t = rm.select_scope(table, site, quarter).copy()
    t['Status'] = t['loading_pct'].map(lambda v: rm.risk_label(v, ceiling))
    t['Equipment'] = t['tool_group'].map(rm.GROUP_LABELS)
    t['Budget h / week'] = t['available_h_wk']*ceiling
    t['Excess h / week'] = (t['required_h_wk']-t['Budget h / week']).clip(lower=0)
    return t.sort_values('loading_pct', ascending=False)


st.sidebar.markdown('<div class="brand"><div class="brand-mark">TD</div><div>'
                    '<div class="brand-title">Planning workspace</div><div class="brand-sub">Capacity & operations</div>'
                    '</div></div>', unsafe_allow_html=True)
page = st.sidebar.radio('Workspace', ['Capacity plan', 'Scenario review', 'Fab performance'], key='page')
st.sidebar.divider()
site = st.sidebar.selectbox('Site', list(C.SITES), format_func=rm.SITE_LABELS.get, key='site')
if page != 'Fab performance':
    quarter = st.sidebar.selectbox('Review quarter', C.PLAN_QUARTERS, index=1, key='quarter')
    st.sidebar.caption('Demand basis: requested wafer starts. All programs remain in the calculation.')
else:
    quarter = ss.get('quarter', '2027Q2')
    st.sidebar.caption('Operating window: 29 Jun–27 Sep 2026. 13 simulated weeks; historical snapshot.')
with st.sidebar.expander('Assumptions & definitions'):
    ceiling = st.slider('Planning limit (%)', 75, 95, 85, step=5, key='ceiling_pct', disabled=page=='Fab performance')/100
    st.caption('85% is a demo planning policy. Loading uses production hours after maintenance, breakdowns and engineering time.')
    st.caption('WSPW = wafer starts per week. A lot starts with 25 wafers. A pass is one visit to an equipment group.')
    st.caption('Resources are pooled within each group. Quarter-average loading does not schedule individual lots.')
st.sidebar.divider()
st.sidebar.caption('Independent interview prototype. Fictional sites and programs; no company data or branding.')

state = fc.base_state(data, 'requested')
table = fc.loading_table(state, ceiling)
focus = [g for g in rm.FOCUS_GROUPS if C.INSTALLED_TOOLS[site][g] > 0]


def capacity_page():
    summary = rm.capacity_summary(table, state, site, quarter, ceiling)
    title('Wafer loading & capacity', f'{rm.SITE_LABELS[site]} · {quarter} review · 2027 quarterly outlook · Requested starts')
    cols = st.columns(4)
    card(cols[0], 'PLANNED WAFER STARTS', f"{summary['starts']:,.0f}", 'wafers / week · all programs')
    card(cols[1], 'ABOVE PLANNING LIMIT', f"{summary['at_risk']} / {summary['groups']}", f'equipment groups · limit {ceiling*100:.0f}%', 'alert' if summary['at_risk'] else '')
    card(cols[2], 'HIGHEST EQUIPMENT LOAD', f"{summary['worst_load']:.1f}%", rm.GROUP_LABELS[summary['worst_group']], 'danger' if summary['worst_load']>100 else 'alert')
    card(cols[3], 'OVER AVAILABLE CAPACITY', str(summary['over_capacity']), 'equipment groups above 100%', 'danger' if summary['over_capacity'] else '')
    worst = rm.select_scope(table, site, quarter).sort_values('loading_pct', ascending=False).iloc[0]
    drivers = fc.drivers(state, site, worst.tool_group, quarter)
    text = (f"{rm.GROUP_LABELS[worst.tool_group]} is the tightest resource at {worst.loading_pct:.1f}% loading. "
            f"{drivers[0]['program']} accounts for {drivers[0]['share_pct']:.0f}% of its workload. "
            f"Review {summary['at_risk']} equipment group(s) against the {ceiling*100:.0f}% planning limit.")
    decision('PLANNING EXCEPTION', text)
    left, right = st.columns([1.65, 1], gap='large')
    with left, st.container(border=True):
        st.subheader('2027 capacity outlook')
        show_all = st.toggle('Show all equipment groups', value=False, key='all_groups')
        groups = C.TOOL_GROUP_ORDER if show_all else focus
        legend(ceiling)
        chart(load_heatmap(table, site, quarter, groups, ceiling), 'capacity_heatmap')
        st.caption('Outlined column = review quarter. Planned equipment releases are included from their release quarter.')
        hidden = rm.select_scope(table, site, quarter)
        hidden = hidden[~hidden.tool_group.isin(groups)]
        hidden_risks = len(fc.constraints(hidden, ceiling))
        st.caption(f'Focused view: {len(groups)} groups. {hidden_risks} additional group(s) above the limit outside this view. Headline counts include every group.')
    with right, st.container(border=True):
        st.subheader('Planned capacity releases')
        arrivals = [a for a in C.TOOL_ARRIVALS if a[0] == site]
        for _, group, release, count in arrivals:
            st.markdown(f'**+{count} {rm.GROUP_LABELS[group]}**')
            st.caption(f'Production release · {release}')
        st.divider()
        st.caption('Capacity begins at the start of the release quarter in this model. Commissioning and qualification dates need confirmation before commitment.')
        st.caption('Fab A supports all programs. Fab B supports DRAM Node A, NAND Gen C and short-loop experiments.')
    left, right = st.columns([1.1, 1.65], gap='large')
    installed = [g for g in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[site][g] > 0]
    with left, st.container(border=True):
        st.subheader('Explain the workload')
        group = st.selectbox('Equipment group', installed, format_func=rm.GROUP_LABELS.get, key='drill_group')
        row = rm.select_scope(table, site, quarter).query('tool_group == @group').iloc[0]
        detail = rm.program_workload(state, site, group, quarter)
        fig = go.Figure(go.Bar(x=[row.required_h_wk], y=['Workload'], orientation='h', marker_color=BLUE,
                              text=[f'{row.required_h_wk:,.1f} h'], textposition='inside',
                              hovertemplate='%{x:.1f} required hours/week<extra></extra>', showlegend=False))
        budget = row.available_h_wk*ceiling
        fig.update_layout(barmode='stack')
        fig.add_vline(x=budget, line_color=AMBER, line_dash='dash')
        fig.add_vline(x=row.available_h_wk, line_color=MUTED, line_dash='dot')
        style(fig, 195)
        fig.update_xaxes(title='Processing hours / week', range=[0,max(row.available_h_wk,row.required_h_wk)*1.12])
        fig.update_yaxes(showticklabels=False)
        chart(fig, 'program_workload')
        st.caption(f'Required {row.required_h_wk:,.1f} h/week · planning budget {budget:,.1f} h/week (dashed) · available {row.available_h_wk:,.1f} h/week (dotted).')
    with right, st.container(border=True):
        st.subheader(f'{quarter} · program contribution')
        st.dataframe(detail[['Program','Starts / week','Passes / wafer','Required h / week','Share %']]
                     .style.format({'Starts / week':'{:,.0f}', 'Passes / wafer':'{:,.0f}', 'Required h / week':'{:,.1f}', 'Share %':'{:.1f}%'}),
                     hide_index=True, width='stretch')
        with st.expander('Trace the calculation'):
            st.write('Required hours/week = Σ (starts/week × passes/wafer × minutes/pass × sampling fraction) ÷ 60.')
            st.write('Available hours/week = installed resources × 168 × production availability. Loading = required ÷ available.')
            st.dataframe(detail.style.format(precision=2), hide_index=True, width='stretch')
            st.caption('Batch furnaces use an equivalent per-wafer rate. Inspection applies the configured sampling fraction.')
    with st.container(border=True):
        st.subheader(f'{quarter} · exception register')
        current = readable_capacity(table, site, quarter, ceiling)
        exceptions = current[current['Status'] != 'Within planning limit']
        if exceptions.empty:
            st.success('All equipment groups are within the modeled planning limit.')
        else:
            display = exceptions[['Equipment','Status','loading_pct','required_h_wk','available_h_wk','Excess h / week']].rename(
                columns={'loading_pct':'Load %','required_h_wk':'Required h / week','available_h_wk':'Available h / week'})
            st.dataframe(display.style.format({'Load %':'{:.1f}%', 'Required h / week':'{:,.1f}',
                          'Available h / week':'{:,.1f}','Excess h / week':'{:,.1f}'}), hide_index=True, width='stretch')
        st.caption('Excess hours are measured against the planning budget, not against the full physical capacity. All equipment groups are checked.')
        export = rm.select_scope(table, site).assign(demand_basis='requested', planning_limit_pct=ceiling*100)
        st.download_button('Download capacity review', export.to_csv(index=False), f'{site}_2027_capacity_review.csv', 'text/csv')


def copy_ai_draft():
    draft = ai.validate_request(ai.StartRequest(**ss['ai_draft']))
    ss['site'], ss['quarter'] = draft.site, draft.first_quarter
    ss['_scenario_scope'] = (draft.site, draft.first_quarter)
    ss['scenario_program'], ss['scenario_pct'] = draft.program, float(draft.change_pct)
    ss['scenario_first'], ss['scenario_last'] = draft.first_quarter, draft.last_quarter
    ss['recovery_enabled'] = False
    ss['review'] = None
    ss['copied_ai_note'] = True


def scenario_page():
    title('Review a loading change', f'{rm.SITE_LABELS[site]} · Requested-start baseline · Evaluate all equipment groups')
    decision('REVIEW FLOW', 'Define the request → check the inputs → compare loading → identify the operating decision.')
    if ss.get('_scenario_scope') != (site, quarter):
        ss['_scenario_scope'] = (site, quarter)
        ss['scenario_program'] = 'DRAM_B' if site == 'FAB_A' else 'NAND_C'
        ss['scenario_pct'] = 10.0
        ss['scenario_first'] = quarter
        ss['scenario_last'] = quarter
        ss['recovery_enabled'] = False
        ss['review'] = None
    with st.expander('Read a planning note with AI', expanded=False):
        note = st.text_area('Planning note', f'Increase DRAM Node B wafer starts by 10% at Fab A during {quarter} only.', height=90, key='planning_note')
        a,b = st.columns([1,2])
        if a.button('Extract request', disabled=not API_KEY, width='stretch'):
            try:
                with st.spinner('Reading the request…'):
                    ss['ai_draft'] = ai.extract_request(note, API_KEY, MODEL).model_dump()
                ss['ai_note_source'] = note
            except Exception as exc:
                ss.pop('ai_draft', None)
                st.error(ai.explain_error(exc))
        if not API_KEY:
            b.caption('AI note reading is optional. Configure OPENAI_API_KEY to enable it; the guided comparison below works without a key.')
        else:
            b.caption('AI extracts fields for review. The capacity model calculates every result.')
        draft_dict = ss.get('ai_draft')
        if draft_dict and ss.get('ai_note_source') == note:
            draft = ai.StartRequest(**draft_dict)
            st.write(draft.model_dump(exclude={'source_quote'}))
            st.caption('Source: ' + draft.source_quote)
            try:
                ai.validate_request(draft)
                valid = True
            except ValueError as exc:
                valid = False
                st.warning(str(exc))
            st.button('Copy reviewed request to controls', on_click=copy_ai_draft, disabled=not valid)
    if ss.get('copied_ai_note'):
        st.caption('The extracted request has been copied below. Review the fields, then compare the scenario.')
    with st.container(border=True):
        st.subheader('1 · Define the request')
        c = st.columns([1.5,1,1,1])
        program = c[0].selectbox('Development program', C.QUALIFIED[site], format_func=lambda p:C.PROGRAMS[p]['name'], key='scenario_program')
        pct = c[1].number_input('Wafer-start change (%)', min_value=-100.0, max_value=100.0, step=5.0, key='scenario_pct')
        first = c[2].selectbox('Start quarter', C.PLAN_QUARTERS, key='scenario_first')
        last = c[3].selectbox('End quarter', C.PLAN_QUARTERS, key='scenario_last')
        recovery = st.checkbox('Test an availability recovery assumption', key='recovery_enabled')
        recovery_group, points = None, 0.0
        if recovery:
            c1,c2 = st.columns(2)
            recovery_group = c1.selectbox('Equipment to recover', [g for g in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[site][g]>0],
                                           format_func=rm.GROUP_LABELS.get, key='recovery_group')
            points = c2.number_input('Availability increase (percentage points)', min_value=.5, max_value=15.0, value=5.0, step=.5)
            st.caption('This is a what-if assumption. Operations must confirm that the additional productive time can be recovered.')
        signature = (site,program,pct,first,last,ceiling,recovery_group,points)
        if st.button('Compare scenario', type='primary'):
            try:
                ss['review'] = rm.run_review(data,site,program,pct,first,last,ceiling,recovery_group,points)
                ss['review_signature'] = signature
                ss['inspect_quarter'] = first
            except ValueError as exc:
                ss['review'] = None
                st.error(str(exc))
    review = ss.get('review')
    if not review:
        st.info('Ready to compare. The baseline is preserved; your change is evaluated as a separate scenario.')
        return
    if ss.get('review_signature') != signature:
        st.info('Inputs changed. Select Compare scenario to update the results.')
        return
    with st.container(border=True):
        st.subheader('2 · Compare the impact')
        inspect = st.selectbox('Inspect quarter', C.PLAN_QUARTERS, index=C.PLAN_QUARTERS.index(first), key='inspect_quarter')
        st.caption(f'{rm.SITE_LABELS[site]} · {inspect} · All equipment groups · Requested-start baseline')
        brief = rm.decision_brief(review, inspect)
        before,after = brief['before'],brief['after']
        cols = st.columns(3)
        card(cols[0], 'WAFER STARTS / WEEK', f"{before['starts']:,.0f} → {after['starts']:,.0f}", f"{after['starts']-before['starts']:+,.0f} wafers / week")
        card(cols[1], 'ABOVE PLANNING LIMIT', f"{before['at_risk']} → {after['at_risk']}", f'all equipment groups · {ceiling*100:.0f}% limit', 'alert' if after['at_risk'] else '')
        card(cols[2], 'HIGHEST LOAD AFTER CHANGE', f"{after['worst_load']:.1f}%", rm.GROUP_LABELS[after['worst_group']], 'danger' if after['worst_load']>100 else 'alert')
        comparison = rm.select_scope(review['comparison'],site,inspect)
        exceptions = set(comparison.loc[(comparison.loading_pct_scenario>=ceiling*100)|comparison.loading_pct_scenario.isna(),'tool_group'])
        groups = [g for g in C.TOOL_GROUP_ORDER if g in set(focus)|exceptions]
        plot = comparison[comparison.tool_group.isin(groups)]
        fig = go.Figure()
        for col,label,color in [('loading_pct_base','Baseline','#bbc9df'),('loading_pct_scenario','Scenario',BLUE)]:
            fig.add_bar(x=plot.tool_group.map(rm.GROUP_LABELS),y=plot[col],name=label,marker_color=color,
                        text=plot[col].map(lambda v:f'{v:.1f}%'),textposition='outside',
                        hovertemplate='%{x}<br>%{y:.1f}% loading<extra>'+label+'</extra>')
        fig.add_hline(y=ceiling*100,line_dash='dash',line_color=AMBER)
        fig.add_hline(y=100,line_dash='dot',line_color=RED)
        style(fig,325)
        fig.update_layout(barmode='group',margin=dict(l=0,r=12,t=25,b=20))
        fig.update_yaxes(title='Equipment loading (%)',range=[0,max(110,plot[['loading_pct_base','loading_pct_scenario']].max().max()*1.18)])
        chart(fig,'scenario_comparison')
        st.caption(f'Dashed: {ceiling*100:.0f}% planning limit. Dotted: 100% available capacity. Focus groups and all scenario exceptions are shown; the table checks every group.')
        view = comparison[['tool_group','loading_pct_base','loading_pct_scenario','change_pp','status']].rename(
            columns={'tool_group':'Equipment','loading_pct_base':'Baseline %','loading_pct_scenario':'Scenario %','change_pp':'Change (pp)','status':'Scenario status'})
        view['Equipment'] = view['Equipment'].map(rm.GROUP_LABELS)
        st.dataframe(view.style.format({'Baseline %':'{:.1f}%','Scenario %':'{:.1f}%','Change (pp)':'{:+.1f}'}),hide_index=True,width='stretch')
    with st.container(border=True):
        st.subheader('3 · Decision brief')
        st.write(brief['recommendation'])
        biggest = brief['biggest']
        if abs(biggest['change_pp'])>=.1:
            st.write(f"Largest loading change: {rm.GROUP_LABELS[biggest['tool_group']]} "
                     f"{biggest['loading_pct_base']:.1f}% → {biggest['loading_pct_scenario']:.1f}% "
                     f"({biggest['change_pp']:+.1f} percentage points).")
        else:
            st.write('The selected scenario does not change loading in this quarter.')
        if after['at_risk'] > before['at_risk']:
            st.write('The scenario introduces additional equipment exceptions. Review experiment timing and the constrained resources before committing the request.')
        elif after['at_risk']:
            st.write('Existing exceptions remain visible. A similar exception count does not imply spare capacity.')
        st.caption(brief['owners'])
        st.caption(brief['limit'])
        with st.expander('Applied assumptions & full-year comparison'):
            for applied in review['applied']: st.write('• '+applied)
            st.dataframe(rm.select_scope(review['comparison'],site)[['quarter','tool_group','loading_pct_base','loading_pct_scenario','change_pp']],hide_index=True,width='stretch')
        export = rm.select_scope(review['comparison'],site).assign(planning_limit_pct=ceiling*100,demand_basis='requested')
        st.download_button('Download scenario comparison',export.to_csv(index=False),f'{site}_scenario_comparison.csv','text/csv')
        payload = {'site':site,'review_quarter':inspect,'changes':review['changes'],'planning_limit_pct':ceiling*100,
                   'baseline':before,'scenario':after,'recommendation':brief['recommendation'],'limitation':brief['limit']}
        st.download_button('Download decision brief',json.dumps(payload,indent=2,default=str),f'{site}_{inspect}_decision_brief.json','application/json')


def production_page():
    p = rm.production_metrics(data,site)
    title('Fab performance', f"{rm.SITE_LABELS[site]} · {p['window_start']:%d %b}–{p['window_end']:%d %b %Y} · 13 simulated weeks",historical=True)
    cols=st.columns(4)
    card(cols[0],'WAFER COMPLETIONS / WEEK',f"{p['avg_outs']:,.0f}",'average finished wafers / week')
    card(cols[1],'FULL-FLOW CYCLE TIME',f"{p['cycle_days']:.1f} d",'completed full-flow lots · short loops excluded')
    card(cols[2],'AVERAGE ACTIVE WIP',f"{p['wip']:,.0f}",'input-wafer equivalents · weekly snapshots')
    card(cols[3],'LINE YIELD',f"{p['line_yield']:.2f}%",'wafer survival · completed-lot cohorts')
    b=p['breakdown']
    biggest=b.idxmax()
    names={'process_days':'Processing','queue_days':'Equipment queues','transport_days':'Transport & dispatch','hold_days':'Engineering holds'}
    decision('OPERATING OBSERVATION',f"{names[biggest]} account for {b[biggest]:.1f} days of average full-flow cycle time. "
             'Review the time breakdown alongside equipment queues before choosing an improvement action.')
    left,right=st.columns([1.65,1],gap='large')
    w=p['weekly'].copy()
    w['date']=pd.to_datetime(w.week_start)
    with left,st.container(border=True):
        st.subheader('Weekly wafer flow')
        fig=go.Figure()
        for col,label,color in [('wafer_starts','Starts',BLUE),('wafer_outs','Completions',TEAL)]:
            fig.add_bar(x=w.date,y=w[col],name=label,marker_color=color,hovertemplate='%{x|%d %b}<br>%{y:,} wafers<extra>'+label+'</extra>')
        style(fig,285)
        fig.update_layout(barmode='group',bargap=.22)
        fig.update_yaxes(title='Wafers / week')
        fig.update_xaxes(tickformat='%d %b',tickmode='array',tickvals=w.date.iloc[::2])
        chart(fig,'weekly_flow')
        st.caption('Completions include lots released in earlier weeks. Starts and completions in the same week are different cohorts.')
    with right,st.container(border=True):
        st.subheader('Work in process')
        fig=go.Figure(go.Scatter(x=w.date,y=w.wip_wafers,mode='lines+markers',line=dict(color=BLUE,width=2),
                                marker=dict(size=5),fill='tozeroy',fillcolor='rgba(24,90,219,.07)',
                                hovertemplate='%{x|%d %b}<br>%{y:,} input-wafer equivalents<extra></extra>'))
        style(fig,285)
        fig.update_yaxes(title='Input-wafer equivalents',rangemode='tozero')
        fig.update_xaxes(tickformat='%d %b',tickmode='array',tickvals=w.date.iloc[::4])
        chart(fig,'weekly_wip')
        st.caption('End-of-week snapshots. Active lots retain their original 25-wafer quantity in this metric.')
    left,right=st.columns([1.35,1],gap='large')
    with left,st.container(border=True):
        st.subheader('Where cycle time is spent')
        full=p['by_program'][p['by_program'].program!='SHORT'].copy()
        fig=go.Figure()
        for col,label,color in [('process_days','Processing',BLUE),('queue_days','Equipment queues',AMBER),
                                ('transport_days','Transport & dispatch','#a6b5cc'),('hold_days','Engineering holds',TEAL)]:
            fig.add_bar(y=full.program.map(lambda x:C.PROGRAMS[x]['name']),x=full[col],name=label,orientation='h',
                        marker_color=color,hovertemplate='%{y}<br>%{x:.1f} days<extra>'+label+'</extra>')
        style(fig,330)
        fig.update_layout(barmode='stack',margin=dict(l=0,r=12,t=15,b=100),
                          legend=dict(y=-.30,yanchor='top'))
        fig.update_xaxes(title='Average days / completed lot')
        fig.update_yaxes(autorange='reversed')
        chart(fig,'cycle_breakdown')
        st.caption('Full-flow programs only. Transport includes modeled dispatch delays; holds represent engineering review pauses.')
    with right,st.container(border=True):
        st.subheader('Equipment queue watch')
        queues=rm.select_scope(data['sim_tool_group_q3_2026'],site).sort_values('avg_queue_h',ascending=False).head(5)
        q=queues[['tool_group','avg_queue_h','loading_pct']].rename(columns={'tool_group':'Equipment','avg_queue_h':'Queue h / visit','loading_pct':'Loading %'})
        q['Equipment']=q['Equipment'].map(rm.GROUP_LABELS)
        st.dataframe(q.style.format({'Queue h / visit':'{:.2f}','Loading %':'{:.1f}%'}),hide_index=True,width='stretch')
        st.caption('Full-window averages. Loading is productive time divided by production-available time; it differs from percent of calendar time productive.')
    with st.expander('Metric definitions, source data & production readiness'):
        st.write('Line yield = total wafers surviving completed lots ÷ the input wafers of those same lots. It measures wafer survival, not electrical die yield.')
        st.write('Full-flow cycle time excludes short-loop experiments and incomplete WIP. WIP is an average of weekly snapshots in input-wafer equivalents.')
        st.write('This dataset supports historical wafer flow, cycle time, queues, WIP and line yield. A live shift dashboard needs timestamped operation events, equipment states, shift calendars, approved targets and refresh monitoring. OEE and die yield also need additional data.')
        st.dataframe(p['by_program'].assign(program=lambda t:t.program.map(lambda x:C.PROGRAMS[x]['name'])).round(2),hide_index=True,width='stretch')
    st.download_button('Download weekly performance',p['weekly'].to_csv(index=False),f'{site}_Q3_2026_performance.csv','text/csv')


if page == 'Capacity plan': capacity_page()
elif page == 'Scenario review': scenario_page()
else: production_page()

with st.expander('Data & model reference',expanded=False):
    st.caption('Source: supplied synthetic project tables. Model assumptions are defined in fab_config.py. All programs and equipment groups remain in the calculation.')
    st.write('Planning: average weekly wafer starts within each quarter. Equipment additions apply for the entire release quarter. Demand adjustments do not rerun the historical simulation.')
    st.write('The historical request-realization regression remains in the source package as an advanced reference. The presentation uses requested starts consistently.')
    st.write('This is an independent prototype informed by the role’s planning responsibilities. It does not reproduce an internal company dashboard.')
st.markdown('<div class="footer"><span>TD Planning · Independent interview prototype</span>'
            '<span>All data synthetic · Draft planning analysis · v2.0</span></div>',unsafe_allow_html=True)
