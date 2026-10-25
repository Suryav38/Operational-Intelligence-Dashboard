"""TD Capacity Plan 2027: what tools the plan needs, by when, and what avoids buying them."""
from __future__ import annotations

import hashlib
import math
import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import assistant as ai_errors
import capital_model as K
import dashboard_ai as DA
import dashboard_ui as ui
import demand_outlook as demand
import executive_charts as ec
import execution_model as em
import fab_config as C
import fab_data as fd
import fab_demo as D
import forecast_model as fm
import launch_charts as LC
import launch_model as L
import plan_charts as PC
import planning_copilot as rag
import review_model as rm
import scenario_requests as request_model
import weekly_scenarios as weekly

load_dotenv(fd.ROOT / '.env')
st.set_page_config(page_title='TD Capacity Plan 2027', page_icon='▦', layout='wide', initial_sidebar_state='collapsed')
st.markdown('<style>' + Path(__file__).with_name('dashboard.css').read_text() + '</style>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(fingerprint):
    return fd.load()


try:
    data = load_data(fd.fingerprint())
except Exception as exc:
    st.error(f'Project data could not be loaded: {exc}')
    st.stop()
ss = st.session_state
FP = fd.fingerprint()


# ------------------------------------------------------------------ cached models
@st.cache_data(show_spinner=False)
def cached_buy(fingerprint, case, basis, ceiling):
    return K.buy_list(load_data(fingerprint), case, basis, ceiling)


@st.cache_data(show_spinner=False)
def cached_levers(fingerprint, case, ceiling, raise_pts, max_share):
    return K.lever_steps(load_data(fingerprint), case, ceiling, raise_pts, max_share)


@st.cache_data(show_spinner=False)
def cached_capacity(fingerprint, case, limit):
    return L.capacity(load_data(fingerprint), case, limit)


# ------------------------------------------------------------------ small building blocks
def number(value, decimals=0, suffix=''):
    return '—' if value is None or pd.isna(value) else f'{value:,.{decimals}f}' + suffix


def esc(text):
    return html.escape(str(text))


def page_head(kicker, title, headline=None):
    st.markdown(f'<div class="kicker">{esc(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    if headline:
        st.markdown(f'<p class="headline">{headline}</p>', unsafe_allow_html=True)


def stats(items):
    """items: (value, label, tone) — tone '', 'warn' or 'bad'."""
    cells = ''.join(f'<div class="stat"><div class="stat-value {tone}">{esc(v)}</div><div class="stat-label">{esc(l)}</div></div>'
                    for v, l, tone in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def section(title, note='', where=st):
    where.markdown(f'<div class="section"><span class="section-title">{esc(title)}</span>'
                   f'<span class="section-note">{esc(note)}</span></div>', unsafe_allow_html=True)


def chart(fig, key, where=st):
    where.plotly_chart(fig, width='stretch', config={'displayModeBar': False, 'scrollZoom': False}, key=key)


def csv(frame, labels=None):
    return frame.rename(columns=labels or {}).to_csv(index=False)


def site_label(s):
    return rm.SITE_LABELS[s]


def group_label(s, g, both):
    return (site_label(s) + ' · ' if both else '') + rm.GROUP_LABELS[g]


# ------------------------------------------------------------------ AI (one quiet panel per decision page)
def secret(name, default=''):
    try:
        return str(st.secrets.get(name) or os.getenv(name, default)).strip()
    except Exception:
        return str(os.getenv(name, default)).strip()


@st.cache_resource(show_spinner=False)
def note_index(fingerprint, api_key, semantic):
    index = rag.DocumentIndex(fd.DOCS)
    if semantic and api_key:
        try:
            from openai import OpenAI
            index.enable_semantic(OpenAI(api_key=api_key, timeout=25, max_retries=1))
        except Exception:
            index.notice = 'Semantic retrieval unavailable; keyword evidence is used.'
    return index


def ai_panel(key, context, examples):
    with st.expander('Ask the planning assistant'):
        st.caption('Answers from the numbers on this page, the bundled tables and dated operating notes. '
                   'The capacity model does the arithmetic; the assistant explains and cites it.')
        choice = st.selectbox('Example question', examples, key=key + '_example')
        question = st.text_area('Question', choice, height=80, max_chars=4000, key=key + '_question_' + str(examples.index(choice)))
        semantic = st.checkbox('Use semantic + keyword note retrieval', value=True, key=key + '_semantic')
        api_key = secret('OPENAI_API_KEY')
        model = secret('OPENAI_MODEL', 'gpt-4o-mini')
        context = rag.json_safe(context)
        fingerprint = rag.documents_fingerprint(fd.DOCS)
        stamp = hashlib.sha256(json.dumps([context, question, semantic, fingerprint], sort_keys=True, default=str).encode()).hexdigest()
        a, b, _ = st.columns([1, 1, 4])
        if a.button('Ask AI', disabled=not api_key, key=key + '_ask', type='primary'):
            try:
                with st.spinner('Reading this view and retrieving evidence…'):
                    index = note_index(fingerprint, api_key, semantic)
                    answer = DA.ViewCopilot(data, index, context).answer(question, api_key, model)
                ss[key + '_answer'] = {'stamp': stamp, 'result': answer}
            except Exception as exc:
                st.error(ai_errors.explain_error(exc))
        if b.button('Preview evidence', key=key + '_preview'):
            ss[key + '_evidence'] = {'stamp': stamp, 'context': context, 'documents': rag.DocumentIndex(fd.DOCS).search(question, 4)}
        if not api_key:
            st.caption('Live answers need OPENAI_API_KEY in the app secrets. Evidence preview works without a key.')
        saved = ss.get(key + '_answer')
        if saved and saved['stamp'] == stamp:
            out = saved['result']
            st.markdown(out['answer'])
            st.caption('AI interpretation · ' + out['retrieval_mode'] + ' · check the cited evidence')
            if out.get('retrieval_notice'):
                st.caption(out['retrieval_notice'])
            with st.expander('Sources and model operations'):
                for source in out['sources']:
                    st.markdown('**[' + source['id'] + '] ' + source.get('source', source.get('title', 'Evidence')) + '**')
                    if source['kind'] == 'document':
                        st.text(source['text'])
                    else:
                        if source.get('sql'):
                            st.code(source['sql'], language='sql')
                        st.json(json.dumps(source['data'], default=str), expanded=False)
                st.write(out['trace'])
        elif saved:
            st.caption('The assumptions or question changed. Ask again to refresh the answer.')
        preview = ss.get(key + '_evidence')
        if preview and preview['stamp'] == stamp:
            st.json(json.dumps(context, default=str), expanded=False)
            for doc in preview['documents']:
                st.markdown('**' + doc['source'] + '**')
                st.text(doc['text'])


# ------------------------------------------------------------------ header + assumptions
g = ui.render_header()
page, site, pilot_case, ceiling = g['page'], g['site'], g['pilot_case'], g['ceiling']
BOTH = site == 'Both'
CASE_NOTE = ui.CASES[pilot_case]


# ================================================================== 1 · CAPITAL PLAN
def load_cell(value, gap, ceiling):
    if pd.isna(value):
        return '<td class="q bad">no tool</td>'
    tone = 'bad' if value > 100 else 'warn' if value >= ceiling * 100 else 'ok'
    return f'<td class="q {tone}">{value:.0f}%</td>'


def capital_table(view, show_covered):
    rows = view if show_covered else view[view.action.ne('Covered')]
    rows = rows.assign(_c=rows.action.eq('Covered')).sort_values(['_c', 'net_new', 'pull_in', 'peak_loading_pct'],
                                                                 ascending=[True, False, False, False])
    head = ('<tr>' + ('<th class="l">Fab</th>' if BOTH else '') + '<th class="l">Tool group</th><th>Installed</th>'
            '<th>On order</th><th>Peak need</th><th>Buy</th><th>Pull in</th><th>Need by</th>'
            + ''.join(f'<th>{q[-2:]} load</th>' for q in K.Q) + '</tr>')
    body = []
    for r in rows.itertuples():
        cls = ' class="covered"' if r.action == 'Covered' else ''
        on_order = f'{r.on_order} <span class="sub">({esc(r.on_order_label.split(" in ")[-1])})</span>' if r.on_order else '<span class="sub">—</span>'
        body.append(f'<tr{cls}>' + (f'<td class="l">{site_label(r.site)}</td>' if BOTH else '')
                    + f'<td class="l">{esc(rm.GROUP_LABELS[r.tool_group])}</td><td class="num">{r.installed}</td>'
                    f'<td class="num">{on_order}</td><td class="num">{r.needed_peak}</td>'
                    f'<td class="buy {"zero" if not r.net_new else ""}">{r.net_new or "–"}</td>'
                    f'<td class="buy {"zero" if not r.pull_in else ""}">{r.pull_in or "–"}</td>'
                    f'<td class="num">{r.need_by or "—"}</td>'
                    + ''.join(load_cell(getattr(r, f'load_{q}'), getattr(r, f'gap_{q}'), ceiling) for q in K.Q) + '</tr>')
    st.markdown(f'<div class="cap-wrap"><table class="cap"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="legend-row"><span><i style="background:var(--warn-bg)"></i>{ceiling*100:.0f}–100% loaded</span>'
                '<span><i style="background:var(--bad-bg)"></i>over 100% · cannot run the plan</span>'
                '<span>Load = required hours ÷ hours from tools released that quarter</span></div>', unsafe_allow_html=True)


def program_hours(state, s, grp, qi):
    rows = []
    for p in K.PROGRAMS:
        starts = state['wspw'][s].get(p, [0.] * 4)[qi]
        if not starts:
            continue
        spec = D.PROGRAMS[p]
        tool = C.TOOL_GROUPS[grp]
        rows.append({'Program': D.NAMES[p], 'Starts / wk': starts, 'Visits / wafer': spec['passes'][grp],
                     'Min / visit': tool['min_per_wafer'] * spec['time_mult'].get(grp, 1.), 'Sampled': tool.get('sample_rate', 1.),
                     'Hours / wk': starts * D.hours_per_wafer(p, grp)})
    frame = pd.DataFrame(rows).sort_values('Hours / wk', ascending=False)
    if (frame['Sampled'] == 1).all():
        frame = frame.drop(columns='Sampled')
    frame['Share'] = (100 * frame['Hours / wk'] / frame['Hours / wk'].sum()).round(0).astype(int).astype(str) + '%'
    return frame


def capital_page():
    top, side = st.columns([3, 1.1], vertical_alignment='bottom')
    with side:
        basis = st.radio('Size the plan to', ['requested', 'expected'], horizontal=True, key='capital_basis',
                         format_func={'requested': 'Requested starts', 'expected': 'Expected starts'}.get,
                         help='Expected = request × the share each program has historically run (Demand page).')
    frame = cached_buy(FP, pilot_case, basis, ceiling)
    view = K.scope(frame, site)
    with top:
        page_head(f'{ui.SITES[site].upper()} · {CASE_NOTE.upper()} · {ceiling*100:.0f}% LIMIT', 'Capital plan',
                  K.headline(view, ceiling, basis))
    t = K.totals(view)
    loads = view.melt(id_vars=['site', 'tool_group'], value_vars=[f'load_{q}' for q in K.Q]).dropna()
    worst = loads.sort_values('value', ascending=False).iloc[0] if len(loads) else None
    stats([
        (t['net_new'], 'Net-new tools to buy', 'bad' if t['net_new'] else ''),
        (t['pull_in'], 'On-order tools to pull in', 'warn' if t['pull_in'] else ''),
        (f"{t['groups']} / {t['groups_total']}", 'Tool groups short of the limit', 'warn' if t['groups'] else ''),
        (t['first_need'] or '—', 'First quarter short', ''),
        (number(worst.value, 0, '%') if worst is not None else '—',
         f"Peak load · {group_label(worst.site, worst.tool_group, BOTH)} {worst.variable[-6:]}" if worst is not None else 'Peak load', 'bad'),
    ])

    head, toggle = st.columns([4, 1], vertical_alignment='bottom')
    with head:
        section('Tool requirement by group', 'Buy = peak need − (installed + on order) · Pull in = on-order tools needed before they arrive')
    show_covered = toggle.checkbox('Show covered groups', key='show_covered')
    capital_table(view, show_covered)

    # Drill-down: why does this group need tools?
    short = view[view.action.ne('Covered')].sort_values(['net_new', 'peak_loading_pct'], ascending=False)
    pool = pd.concat([short, view[~view.index.isin(short.index)].sort_values('peak_loading_pct', ascending=False)])
    options = list(zip(pool.site, pool.tool_group))
    if ss.get('driver_group') not in options:
        ss['driver_group'] = options[0]
    left, right = st.columns([1.2, 1.1], gap='large')
    with left:
        section('What drives the requirement', 'Required tool hours by program vs. hours the fleet supplies')
        s, grp = st.selectbox('Tool group', options, key='driver_group', format_func=lambda k: group_label(*k, True),
                              label_visibility='collapsed')
        state = K.plan_state(data, pilot_case, basis)
        chart(PC.drivers(state, s, grp, ceiling), 'capital_drivers')
    row = view[(view.site == s) & (view.tool_group == grp)].iloc[0]
    qi = int(pd.Series([row[f'load_{q}'] for q in K.Q]).fillna(1e9).values.argmax())
    q = K.Q[qi]
    hours = program_hours(state, s, grp, qi)
    per = 168 * state['avail'][s][grp][qi]
    tools = K.timed_tools()[K.SITES.index(s), K.GROUPS.index(grp), qi]
    required = hours['Hours / wk'].sum()
    with right:
        section(f'{q} · the arithmetic', group_label(s, grp, True))
        st.markdown(
            f'<div class="calc">Required = Σ starts × visits × min/visit × sampled ÷ 60 = <b>{required:,.0f} h/wk</b><br>'
            f'Available = {tools:.0f} tools × 168 h × {state["avail"][s][grp][qi]*100:.0f}% availability = <b>{tools*per:,.0f} h/wk</b><br>'
            f'Load = {required:,.0f} ÷ {tools*per:,.0f} = <b>{100*required/(tools*per) if tools else float("nan"):.1f}%</b><br>'
            f'Tools needed = ⌈{required:,.0f} ÷ ({per:,.0f} × {ceiling:.2f})⌉ = <b>{int(math.ceil(required/(per*ceiling)-1e-9))}</b></div>',
            unsafe_allow_html=True)
        st.dataframe(hours, hide_index=True, width='stretch',
                     column_config={'Starts / wk': st.column_config.NumberColumn(format='%.0f'),
                                    'Min / visit': st.column_config.NumberColumn(format='%.2f'),
                                    'Hours / wk': st.column_config.NumberColumn(format='%.0f')})

    labels = {'site': 'Fab', 'tool_group': 'Tool group', 'installed': 'Installed', 'on_order': 'On order',
              'on_order_label': 'Arrivals', 'needed_peak': 'Peak need', 'peak_quarter': 'Peak quarter', 'net_new': 'Buy',
              'pull_in': 'Pull in', 'need_by': 'Need by', 'action': 'Action'}
    export = view[list(labels) + [f'load_{q}' for q in K.Q]].assign(planning_limit_pct=ceiling * 100, basis=basis, plan_case=pilot_case)
    st.download_button('Download tool requirement (CSV)', csv(export, labels), 'td_2027_tool_requirement.csv', 'text/csv')
    ai_panel('ai_capital', dict(view='2027 capital plan', site=site, pilot_case=pilot_case, planning_limit=ceiling, demand_basis=basis,
                                totals=t, headline=K.headline(view, ceiling, basis).replace('<b>', '').replace('</b>', ''),
                                tool_requirement=export.round(1).to_dict('records')), [
        'Which tool groups drive the 2027 buy, and which programs create that load?',
        'If one EUV tool becomes unavailable at Fab A during 2027Q3, what happens to capacity?',
        'Could qualified DRAM Node A work move to Fab B, and would that create another constraint?'])


# ================================================================== 2 · DEMAND
def demand_page():
    frame = demand.plan(data, site, None, pilot_case)
    frame['expected_plan'] = frame.expected_wspw.fillna(frame.requested_wspw)  # pilot has no history: stays at request
    plan = frame.groupby('quarter')[['requested_wspw', 'expected_plan']].sum().reindex(K.Q)
    plan = plan.rename(columns={'expected_plan': 'expected_wspw'})
    history = em.historical_attainment(data, site).set_index('quarter')
    rate = history.actual_wspw.sum() / history.requested_wspw.sum()
    q1, q4 = plan.requested_wspw.iloc[0], plan.requested_wspw.iloc[-1]
    held, metrics = demand.validation(data, site)
    dram = frame[frame.technology_family.eq('DRAM')].requested_wspw.sum() / frame.requested_wspw.sum()
    headline = (f'Requested starts rise from <b>{q1:,.0f}</b> to <b>{q4:,.0f}</b> wafers a week through 2027 '
                f'(<b>{100*(q4/q1-1):+.0f}%</b>). Programs have run <b>{100*rate:.0f}%</b> of what they requested since 2024, '
                f'so expected starts end the year near <b>{plan.expected_wspw.iloc[-1]:,.0f}</b>.')
    page_head(f'{ui.SITES[site].upper()} · {CASE_NOTE.upper()}', 'Demand', headline)
    stats([
        (number(q4), '2027Q4 requested starts / week', ''),
        (number(plan.expected_wspw.iloc[-1]), '2027Q4 expected starts / week', ''),
        (number(100 * (q4 / q1 - 1), 0, '%'), 'Request growth, Q1 → Q4', ''),
        (number(100 * dram, 0, '%'), 'DRAM share of 2027 starts (incl. HBM)', ''),
        (f"{number(metrics['raw_mape'],0,'%')} → {number(metrics['expected_mape'],0,'%')}", 'Backtest error: raw request → adjusted', ''),
    ])
    left, right = st.columns([1.35, 1], gap='large')
    with left:
        section('History and plan', 'Quarterly average wafer starts per week')
        chart(PC.demand_trend(plan, history[['requested_wspw', 'actual_wspw']]), 'demand_trend')
    with right:
        section('2027 mix')
        a, b = st.columns(2)
        split = a.radio('Split by', ['Program', 'Fab'], horizontal=True, key='demand_split', label_visibility='collapsed')
        measure = b.radio('Measure', ['Requested', 'Expected'], horizontal=True, key='demand_measure', label_visibility='collapsed')
        chart(PC.demand_mix(frame, split, 'requested_wspw' if measure == 'Requested' else 'expected_plan'), 'demand_mix')
    left, right = st.columns([1, 1.35], gap='large')
    with left:
        section('How much of each request actually runs', 'Fit on 2024Q1–2026Q3, pooled across fabs')
        chart(PC.realization(fm.fit(data['wafer_start_history'])), 'demand_realization')
    with right:
        section('Why plan on expected starts at all')
        st.markdown(
            f'<p class="note">Each program’s factor is a least-squares fit of actual starts on requested starts, through the origin. '
            f'Tested on quarters the fit never saw (2026Q1–Q3), the raw request missed actual starts by <b>{number(metrics["raw_mape"],1,"%")}</b> on average; '
            f'request × factor missed by <b>{number(metrics["expected_mape"],1,"%")}</b> across {metrics["rows"]} fab-program-quarters. '
            'DRAM Node B and the HBM test vehicle over-request most, and they carry the EUV load, so this is the factor that moves the EUV answer.</p>'
            '<p class="note">This is request realization, not a market forecast. The pilot has no history, so it is planned at its request.</p>',
            unsafe_allow_html=True)
    with st.expander('Plan rows and download'):
        out = frame.assign(Fab=frame.site.map(rm.SITE_LABELS), Program=frame.program.map(D.NAMES))
        cols = {'quarter': 'Quarter', 'Fab': 'Fab', 'Program': 'Program', 'requested_wspw': 'Requested WSPW',
                'expected_plan': 'Expected WSPW', 'realization_rate': 'Realization factor'}
        st.dataframe(out[list(cols)].rename(columns=cols).round(2), hide_index=True, width='stretch')
        st.download_button('Download 2027 demand (CSV)', csv(out[list(cols)], cols), 'td_2027_demand.csv', 'text/csv')


# ================================================================== 3 · LEVERS
def levers_page():
    head = st.container()
    a, b, _ = st.columns([1, 1, 2])
    raise_pts = a.selectbox('Raise the planning limit by', [0, 5, 10], index=1, key='lever_raise', format_func=lambda v: f'{v} pts')
    max_share = b.selectbox('Most of a program that may move fabs', [25, 50, 100], index=1, key='lever_share', format_func=lambda v: f'{v}%')
    steps, transfer, higher = cached_levers(FP, pilot_case, ceiling, raise_pts, max_share)
    cols = [c for c in steps.columns if '|' in c and (BOTH or c.startswith(site + '|'))]
    steps = steps.assign(net_new=steps[cols].sum(axis=1))
    v = steps.net_new.tolist()
    headline = (f'Three levers take the buy from <b>{v[0]}</b> to <b>{v[-1]} tools</b> without changing what the programs get to run. '
                'None is free: each one moves risk from the capital budget into the fab.')
    with head:
        page_head(f'{ui.SITES[site].upper()} · {CASE_NOTE.upper()} · FROM {ceiling*100:.0f}% LIMIT', 'Levers', headline)
    left, right = st.columns([1.3, 1], gap='large')
    with left:
        section('Net-new tools, lever by lever', 'Each step keeps the ones before it')
        chart(PC.waterfall(steps), 'lever_waterfall')
    _, short_full, over_full = K.exposure(data, pilot_case, ceiling)
    mult = K.queue_multiplier(ceiling, higher) if higher > ceiling else 1
    moves = '; '.join(f"{m['share_pct']}% of {D.NAMES[m['program']]} {site_label(m['from_site'])} → {site_label(m['to_site'])} (~{m['avg_wspw']} WSPW)"
                      for m in transfer['moves']) or 'no move lowers the count'
    with right:
        section('What each lever costs you')
        items = [
            ('Plan to expected starts', v[1] - v[0],
             f'Buy for what programs historically run, not what they request. Risk: if every program runs its full request, '
             f'{short_full} tool groups sit above the limit and {over_full} above 100%.'),
            (f'Run tools to {higher*100:.0f}%', v[2] - v[1],
             f'Queue time grows with u/(1−u). Going from {ceiling*100:.0f}% to {higher*100:.0f}% loading multiplies queue time at a '
             f'loaded tool group by about <b>{mult:.1f}×</b>, which shows up as cycle time.' if higher > ceiling else 'Limit unchanged.'),
            ('Rebalance qualified work', v[3] - v[2],
             f'Searched {transfer["searched"]:,} share combinations of the programs qualified at both fabs. Best: {esc(moves)}. '
             'Network starts are unchanged; EUV programs cannot move (Fab B has no EUV).'),
        ]
        st.markdown(''.join(f'<div class="lever"><div class="lever-head"><span class="lever-name">{n}</span>'
                            f'<span class="lever-delta">{d:+d} tools</span></div><div class="lever-body">{t}</div></div>'
                            for n, d, t in items), unsafe_allow_html=True)
    left, right = st.columns([1, 1.3], gap='large')
    with left:
        section('Why the limit is not 100%', 'Queue time per visit vs. loading, calibrated to 2026')
        options = [(s, g_) for s in (C.SITES if BOTH else [site]) for g_ in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[s][g_]]
        s, grp = st.selectbox('Curve for', options, key='curve_group', format_func=lambda k: group_label(*k, True),
                              index=options.index(('FAB_A', 'EUV')) if ('FAB_A', 'EUV') in options else 0)
        chart(PC.operating_curve(fd.operating_curve(data, s, grp), ceiling, higher), 'operating_curve')
    with right:
        section('Where the savings land', 'Net-new tools by group after each step')
        table = steps.set_index('step')[cols].T
        table.columns = ['Requested', 'Expected', f'{higher*100:.0f}% limit', 'Rebalanced']
        table = table[table.sum(axis=1) > 0]
        table.index = [group_label(*k.split('|'), BOTH) for k in table.index]
        st.dataframe(table, width='stretch', height=min(460, 38 + 35 * len(table)))


# ================================================================== 4 · SCENARIOS
def scenario_test():
    programs = em.program_options(site, 'All families', include_pilot=False)
    quarter = '2027Q3'
    operation = st.selectbox('Test', request_model.OPERATIONS, key='scenario_operation')
    if ss.get('_scenario_scope') != (site,):
        ss['_scenario_scope'] = (site,)
        ss['review'] = None
    a, b, c = st.columns([.9, 1.5, .9])
    request_site = a.selectbox('Fab receiving the change', list(C.SITES), format_func=rm.SITE_LABELS.get, key='scenario_site') if BOTH else site
    if not BOTH:
        a.markdown('**Fab**')
        a.write(site_label(site))
    to_site = 'FAB_B' if request_site == 'FAB_A' else 'FAB_A'
    outage_group, unavailable = 'EUV', 1
    if operation == 'Tool unavailability':
        available = [g_ for g_ in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g_] > 0]
        outage_group = b.selectbox('Unavailable equipment', available, format_func=rm.GROUP_LABELS.get, key='outage_group_' + request_site)
        program = 'DRAM_A'
    else:
        eligible = [p for p in programs if p in C.QUALIFIED[request_site] and (operation != 'Qualified load transfer' or p in C.QUALIFIED[to_site])]
        if not eligible:
            st.info('No program here is qualified for this action. EUV routes cannot move to Fab B.')
            return
        if ss.get('scenario_program') not in eligible:
            ss['scenario_program'] = 'DRAM_B' if 'DRAM_B' in eligible else eligible[0]
        program = b.selectbox('Program', eligible, format_func=ec.PROGRAM_SHORT.get, key='scenario_program')
    request_priority = c.selectbox('Request priority', ['NORMAL', 'HOT'], key='scenario_priority', disabled=operation == 'Tool unavailability')
    timing = st.radio('Timing', ['Quarter range', 'Week range'], horizontal=True, key='scenario_timing')
    first_week = last_week = None
    if timing == 'Quarter range':
        a, b = st.columns(2)
        ss.setdefault('scenario_first', quarter)
        ss.setdefault('scenario_last', quarter)
        first_q = a.selectbox('From quarter', C.PLAN_QUARTERS, key='scenario_first')
        last_q = b.selectbox('Through quarter', C.PLAN_QUARTERS, key='scenario_last')
        period_label = first_q + '–' + last_q
        first_period, last_period = first_q, last_q
    else:
        calendar = weekly.calendar().set_index('week_start')
        week_options = list(calendar.index)
        q_start = pd.Timestamp(int(quarter[:4]), 3 * (int(quarter[-1]) - 1) + 1, 1)
        q_end = q_start + pd.DateOffset(months=3)
        defaults = calendar[(calendar.period_start >= q_start) & (calendar.period_end <= q_end)]
        a, b = st.columns(2)
        first_week = a.selectbox('Start week', week_options, index=week_options.index(defaults.index[0]), format_func=lambda w: calendar.loc[w, 'label'], key='scenario_start_week')
        last_week = b.selectbox('End week', week_options, index=week_options.index(defaults.index[-1]), format_func=lambda w: calendar.loc[w, 'label'], key='scenario_end_week')
        first_q = weekly.quarter_of(calendar.loc[first_week, 'period_start'])
        last_q = weekly.quarter_of(calendar.loc[last_week, 'period_end'] - pd.Timedelta(days=1))
        period_label = calendar.loc[first_week, 'label'] + ' through ' + calendar.loc[last_week, 'label'] + ' 2027'
        first_period, last_period = first_week, last_week
    pct, target = 0., None
    if operation == 'Tool unavailability':
        unavailable = int(st.number_input('Tools unavailable', min_value=1, max_value=100, value=1, step=1, key='unavailable_tools'))
    elif operation == 'Qualified load transfer':
        st.caption(f'{site_label(request_site)} → {site_label(to_site)}. Both fabs are evaluated; network starts are conserved.')
        pct = st.number_input('Share of the program to move (%)', min_value=0., max_value=100., value=10., step=5., key='transfer_pct')
    else:
        a, b = st.columns(2)
        mode = a.radio('Demand input', ['Percentage change', 'Target WSPW'], horizontal=True, key='demand_mode')
        state = L.planning_state(data, pilot_case)
        pct = b.number_input('Starts change (%)', min_value=-100., max_value=100., value=10., step=5., key='scenario_pct') if mode == 'Percentage change' else 0.
        target = b.number_input('Target starts / week', min_value=0., value=float(state['wspw'][request_site][program][C.PLAN_QUARTERS.index(first_q)]), step=25.,
                                key='target_' + request_site + '_' + program + '_' + first_q) if mode == 'Target WSPW' else None
    milestone = st.text_input('Urgent experiment / milestone (optional)', key='hot_milestone') if request_priority == 'HOT' and operation != 'Tool unavailability' else ''
    recovery = st.checkbox('Also test an availability recovery', key='recovery_enabled') if operation == 'Demand growth' else False
    recovery_group, points = None, 0.
    if recovery:
        a, b = st.columns(2)
        recovery_group = a.selectbox('Equipment to recover', [g_ for g_ in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g_] > 0], format_func=rm.GROUP_LABELS.get, key='recovery_group_' + request_site)
        points = b.number_input('Recovery (percentage points)', min_value=.5, max_value=15., value=5., step=.5, key='recovery_points')
    signature = (site, pilot_case, operation, request_site, program, pct, target, timing, first_period, last_period, ceiling, recovery_group, points, request_priority, milestone, outage_group, unavailable)
    if st.button('Run scenario', type='primary'):
        try:
            review = request_model.run(data, operation, request_site, program, pct, outage_group, unavailable, to_site, first_period, last_period,
                                       'weeks' if timing == 'Week range' else 'quarters', ceiling, recovery_group, points, target, pilot_case)
            review.update(request_priority=request_priority, milestone=milestone)
            ss['review'] = review
            ss['review_signature'] = signature
            ss['inspect_week' if timing == 'Week range' else 'inspect_quarter'] = first_q if timing != 'Week range' else first_period
        except ValueError as exc:
            ss['review'] = None
            st.error(str(exc))
    review = ss.get('review')
    if not review:
        return
    if ss.get('review_signature') != signature:
        st.info('Inputs changed. Run the scenario again to update the results.')
        return
    include_other = st.checkbox('Include the other fab in results', value=False, key='scenario_include_other') if BOTH and operation != 'Qualified load transfer' else operation == 'Qualified load transfer'
    result_scope = 'Both' if include_other else request_site
    request_name = rm.GROUP_LABELS[outage_group] + ' unavailable' if operation == 'Tool unavailability' else ec.PROGRAM_SHORT[program] + ' · ' + request_priority
    if timing == 'Week range':
        inspect = st.selectbox('Inspect week', week_options, format_func=lambda w: calendar.loc[w, 'label'], key='inspect_week')
        before = weekly.summary(review, result_scope, inspect)
        after = weekly.summary(review, result_scope, inspect, True)
        comparison = review['weekly_comparison']
        current = rm.select_scope(comparison[comparison.week_start == inspect], result_scope)
        inspection_label = calendar.loc[inspect, 'label']
    else:
        inspect = st.selectbox('Inspect quarter', C.PLAN_QUARTERS, key='inspect_quarter')
        brief = rm.decision_brief(review, inspect, result_scope)
        before, after = brief['before'], brief['after']
        comparison = review['comparison']
        current = rm.select_scope(comparison, result_scope, inspect)
        inspection_label = inspect
    section(site_label(request_site) + (' → ' + site_label(to_site) if operation == 'Qualified load transfer' else '') + ' · ' + request_name,
            period_label + ' · ' + ('both fabs' if include_other else site_label(request_site) + ' only'))
    stats([
        (f"{before['starts']:,.0f} → {after['starts']:,.0f}", 'Starts / week', ''),
        (f"{before['at_risk']} → {after['at_risk']}", f'Groups at or above {ceiling*100:.0f}%', 'warn' if after['at_risk'] > before['at_risk'] else ''),
        ('no capacity' if pd.isna(after['worst_load']) else number(after['worst_load'], 1, '%'),
         'Peak load · ' + group_label(after['worst_site'], after['worst_group'], True), 'bad' if pd.isna(after['worst_load']) or after['worst_load'] > 100 else 'warn'),
        (str(after['over_capacity']), 'Groups over 100%', 'bad' if after['over_capacity'] else ''),
    ])
    if operation != 'Tool unavailability' and request_priority == 'HOT':
        st.caption('HOT · ' + (milestone or 'no milestone given') + '. Priority changes dispatch order, not tool hours, so capacity results are the same as NORMAL.')
    a, b = st.columns([1.3, 1], gap='large')
    with a:
        section(inspection_label + ' · baseline vs scenario')
        chart(weekly.equipment_chart(current, result_scope, ceiling), 'scenario_loads')
    with b:
        section('Groups above the limit, across 2027')
        chart(weekly.risk_chart(review, result_scope, ceiling) if timing == 'Week range' else ec.scenario_risks(review, result_scope, ceiling), 'scenario_risks')
    with st.expander('All tool groups and applied assumptions'):
        frame = current.assign(Fab=current.site.map(rm.SITE_LABELS), Equipment=current.tool_group.map(rm.GROUP_LABELS))
        st.dataframe(frame[['Fab', 'Equipment', 'loading_pct_base', 'loading_pct_scenario', 'change_pp', 'status']].rename(columns={
            'loading_pct_base': 'Baseline load (%)', 'loading_pct_scenario': 'Scenario load (%)', 'change_pp': 'Change (pts)', 'status': 'Status'}).round(1),
            hide_index=True, width='stretch')
        st.write(review['applied'])
    exported = rm.select_scope(comparison, result_scope).assign(request_site=request_site, request_program=program, request_priority=request_priority,
                                                                timing_mode=review['timing_mode'], requested_window=period_label, plan_case=pilot_case, operation=operation)
    st.download_button('Download scenario (CSV)', csv(exported), f'{result_scope}_scenario.csv', 'text/csv')
    ai_panel('ai_scenario', dict(view='Live capacity scenario', site=result_scope, pilot_case=pilot_case, planning_limit=ceiling,
                                 request=dict(operation=operation, site=request_site, program=program, priority=request_priority, period=period_label),
                                 before=before, after=after, equipment=current.round(3).to_dict('records')), [
        'Which named equipment constrains this request, and which documented response should we examine?',
        'Can we consolidate qualified work into the other fab without simply moving the constraint?',
        'What is the difference between this capacity calculation and the pilot WIP/completion simulation?'])


def pilot_study():
    if site == 'FAB_B':
        st.info('The EUV pilot is qualified only at Fab A. Switch the fab to Fab A or Both fabs.')
        return
    case = pilot_case if pilot_case != 'baseline' else 'introduction'
    if pilot_case == 'baseline':
        st.caption('Plan case is “Current plan”, so this shows the pilot introduction case. Pick a pilot case above to switch.')
    summary = L.study_summary(data, case)
    section(f'Fab A · {D.NAMES[D.PILOT]} · {ui.CASES[case]}',
            f'Lot simulation · release {D.LAUNCH_DATE} · {summary["replications"]} matched runs')
    stats([
        (f"{number(summary['baseline_post_launch_wip'])} → {number(summary['post_launch_wip'])}", 'Average WIP after release (wafers)', 'warn'),
        (number(summary['mean_delay_days'], 2, ' d'), 'Mean delay to existing lots', ''),
        (number(summary['p90_delay_days'], 2, ' d'), '90th percentile delay', ''),
        (number(summary['pilot_completed_lots_mean'], 1), 'Pilot lots finished by study end', ''),
    ])
    a, b = st.columns(2, gap='large')
    with a:
        section('WIP with and without the pilot', 'Line = mean of runs, band = min–max')
        chart(LC.wip(data, case), 'launch_wip')
    with b:
        section('Where existing lots lose time')
        detail = st.radio('Impact view', ['Delay by program', 'Queue by tool group'], horizontal=True, key='pilot_impact_view', label_visibility='collapsed')
        chart(LC.existing_delays(data, case) if detail == 'Delay by program' else LC.queues(data, case), 'launch_detail')
    _, base = cached_capacity(FP, 'baseline', ceiling)
    _, proposed = cached_capacity(FP, case, ceiling)
    quarter = st.selectbox('Capacity quarter', C.PLAN_QUARTERS, index=2, key='pilot_quarter')
    current = rm.select_scope(rm.compare_tables(base, proposed, ceiling), 'FAB_A', quarter)
    a, b = st.columns(2, gap='large')
    with a:
        section(quarter + ' · loading with the pilot')
        chart(weekly.equipment_chart(current, 'FAB_A', ceiling), 'pilot_capacity')
    with b:
        section('Weekly loading at exact release dates')
        grp = st.selectbox('Tool group', C.TOOL_GROUP_ORDER, format_func=rm.GROUP_LABELS.get, key='pilot_weekly_equipment')
        chart(LC.weekly_loading(data, case, grp, ceiling), 'pilot_weekly_loading')
    with st.expander('How the study is set up'):
        st.markdown(f'<p class="note">Discrete-event simulation of Fab A, 16 weeks from 5 Apr 2027. The pilot releases on {D.LAUNCH_DATE} (week 7). '
                    'Each of five random seeds runs three cases from the identical starting WIP, and every existing lot keeps its own random stream, '
                    'so the delay is the same lot’s finish time with the pilot minus without it. '
                    f'{summary["matched_lots"]:,} of {summary["initial_lots"]:,} existing-lot instances are matched; {summary["unfinished_lots"]:,} unfinished.</p>',
                    unsafe_allow_html=True)
        st.download_button('Download paired lot comparison (CSV)', csv(data['launch_lot_comparison'].query('case==@case')), 'paired_existing_lots.csv', 'text/csv')


def scenarios_page():
    page_head(f'{ui.SITES[site].upper()} · {CASE_NOTE.upper()} · {ceiling*100:.0f}% LIMIT', 'Scenarios',
              'Stress the plan before committing capital: a demand change, a tool going down, or work moving between fabs. '
              'The pilot study shows what the capacity screen cannot: WIP and delay.')
    view = st.radio('Scenario type', ['Demand, tool or transfer', 'EUV pilot study'], horizontal=True, key='what_if', label_visibility='collapsed')
    if view == 'EUV pilot study':
        pilot_study()
    else:
        scenario_test()


# ================================================================== 5 · 2026 BASELINE
def baseline_page():
    check = K.model_check(data)
    check = check if BOTH else check[check.site == site]
    close = check.difference_pts.abs().le(3)
    outliers = check[~close].sort_values('difference_pts', key=abs, ascending=False)
    note = ''
    if len(outliers):
        r = outliers.iloc[0]
        note = (f' The exception is <b>{group_label(r.site, r.tool_group, True)}</b> ({r.difference_pts:+.0f} pts): '
                'furnace batches rarely fill, and a part-full batch still takes the whole 6-hour run.' if r.tool_group == 'FURN'
                else f' The largest gap is <b>{group_label(r.site, r.tool_group, True)}</b> ({r.difference_pts:+.0f} pts).')
    headline = (f'Before trusting the 2027 numbers: on the last 13 weeks of simulated 2026 production, the capacity model lands within '
                f'±3 pts of the lot simulation on <b>{int(close.sum())} of {len(check)}</b> tool groups.' + note)
    page_head(f'{ui.SITES[site].upper()} · 29 SEP 2025 – 27 SEP 2026', '2026 baseline', headline)
    history = em.historical_attainment(data, site)
    recent = history.tail(4)
    weekly_frame = data['sim_fab_weekly']
    sites = list(C.SITES) if BOTH else [site]
    last13 = weekly_frame[weekly_frame.site.isin(sites)]
    last13 = last13[last13.week_start >= sorted(last13.week_start.unique())[-13]]
    lots = data['sim_lots']
    done = lots[lots.site.isin(sites) & lots.status.eq('COMPLETE') & lots.program.ne('SHORT')]
    stats([
        (f'{int(close.sum())} / {len(check)}', 'Tool groups where model ≈ simulation (±3 pts)', ''),
        (number(check.difference_pts.abs().max(), 1, ' pts'), 'Largest model gap', 'warn' if check.difference_pts.abs().max() > 3 else ''),
        (number(100 * recent.actual_wspw.sum() / recent.requested_wspw.sum(), 0, '%'), 'Starts run vs requested, last 4 quarters', ''),
        (number(done.cycle_time_days.mean(), 1, ' d'), 'Full-flow cycle time (completed lots)', ''),
        (number(last13.groupby('week_start').wip_wafers.sum().mean()), 'Average WIP, last 13 weeks (wafers)', ''),
    ])
    left, right = st.columns([1.15, 1], gap='large')
    with left:
        section('Capacity model vs. lot simulation', 'Same starts, same fleet, last 13 weeks')
        chart(PC.model_check(check), 'model_check')
    with right:
        section('Requested vs. actual starts', 'Source of the realization factor')
        chart(PC.attainment(history), 'attainment')
        section('Cycle time', 'What the planning limit protects')
        chart(PC.cycle_trend(weekly_frame, site), 'cycle_trend')
    with st.expander('Availability: assumed vs. observed'):
        out = check.assign(Fab=check.site.map(rm.SITE_LABELS), Equipment=check.tool_group.map(rm.GROUP_LABELS))
        st.dataframe(out[['Fab', 'Equipment', 'assumed_nonproductive_pct', 'observed_nonproductive_pct', 'model_pct', 'simulated_pct', 'queue_h']].rename(columns={
            'assumed_nonproductive_pct': 'Assumed down + engineering (%)', 'observed_nonproductive_pct': 'Observed down + engineering (%)',
            'model_pct': 'Model load (%)', 'simulated_pct': 'Simulated load (%)', 'queue_h': 'Queue / visit (h)'}).round(1), hide_index=True, width='stretch')


PAGES = {'Capital plan': capital_page, 'Demand': demand_page, 'Levers': levers_page, 'Scenarios': scenarios_page, '2026 baseline': baseline_page}
PAGES[page]()
st.markdown('<div class="footer"><span>All fab data, routes, tool counts and plans are synthetic. No company data is used.</span>'
            '<span>Method notes: MODEL_GUIDE.md · VIEW_GUIDE.md</span></div>', unsafe_allow_html=True)
