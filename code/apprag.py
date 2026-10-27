"""Fab Capacity Planner 2027: what tools the plan needs, by when, and what avoids buying them."""
from __future__ import annotations

import hashlib
import math
import html
import json
import os
from pathlib import Path

import numpy as np
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
import fab_month as M
import forecast_model as fm
import launch_model as L
import plan_charts as PC
import planning_copilot as rag
import planning_context as PCX
import review_model as rm
import scenario_requests as request_model
import weekly_scenarios as weekly

load_dotenv(fd.ROOT / '.env')
st.set_page_config(page_title='Fab Capacity Planner 2027', page_icon='▦', layout='wide', initial_sidebar_state='collapsed')
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
def cached_buy(fingerprint, case, basis, ceiling, availability='plan'):
    return K.buy_list(load_data(fingerprint), case, basis, ceiling, availability=availability)


@st.cache_data(show_spinner=False)


@st.cache_data(show_spinner=False)
def cached_capacity(fingerprint, case, limit):
    return L.capacity(load_data(fingerprint), case, limit)


# ------------------------------------------------------------------ small building blocks
def number(value, decimals=0, suffix=''):
    return '—' if value is None or pd.isna(value) else f'{value:,.{decimals}f}' + suffix


def esc(text):
    return html.escape(str(text))


def page_head(kicker, title, headline=None):
    if kicker:
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


AI_NOTE = ('Answers from the numbers on this page, the bundled tables and dated operating notes. '
           'The capacity model does the arithmetic; the assistant explains and cites it.')


def ai_panel(key, context, examples, title='Ask the planning assistant', expanded=False, note=AI_NOTE):
    with st.expander(title, expanded=expanded):
        st.caption(note)
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
page, site, pilot_case, ceiling, avail = g['page'], g['site'], g['pilot_case'], g['ceiling'], g['availability']
BOTH = site == 'Both'
CASE_NOTE = ui.CASES[pilot_case]


def quarter_cell(add, value, ceiling, after=None):
    """+tools to add, then load before → after adding them. Shading follows the load after the response."""
    shown = value if after is None else after
    tone = 'bad' if pd.isna(shown) or shown > 100 else 'warn' if shown >= ceiling * 100 + .05 else 'ok'
    fmt = lambda v: 'no tool' if pd.isna(v) else f'{v:.0f}%'
    load = fmt(value) if after is None or (not add and abs((after or 0) - (value or 0)) < .5) else f'{fmt(value)} → {fmt(after)}'
    tools = f'+{add}' if add else '–'
    return f'<td class="q {tone}"><span class="qa{" zero" if not add else ""}">{tools}</span><span class="ql">{load}</span></td>'


def capital_table(view, show_covered):
    rows = view if show_covered else view[view.action.ne('Covered')]
    rows = rows.assign(_c=rows.action.eq('Covered')).sort_values(['_c', 'net_new', 'pull_in', 'peak_loading_pct'],
                                                                 ascending=[True, False, False, False])
    head = ('<tr>' + ('<th class="l">Fab</th>' if BOTH else '') + '<th class="l">Tool group</th><th>Installed</th>'
            '<th>On order</th><th>Pull in</th><th>Buy · 2027</th>'
            + ''.join(f'<th>{q}<div class="sub">add · load before → after</div></th>' for q in K.Q) + '<th>Fleet by Q4</th></tr>')
    body = []
    for r in rows.itertuples():
        cls = ' class="covered"' if r.action == 'Covered' else ''
        on_order = f'{r.on_order} <span class="sub">({esc(r.on_order_label.split(" in ")[-1])})</span>' if r.on_order else '<span class="sub">—</span>'
        pull = f'{r.pull_in} <span class="sub">to {r.pull_in_by}</span>' if r.pull_in else '–'
        body.append(f'<tr{cls}>' + (f'<td class="l">{site_label(r.site)}</td>' if BOTH else '')
                    + f'<td class="l">{esc(rm.GROUP_LABELS[r.tool_group])}</td><td class="num">{r.installed}</td>'
                    f'<td class="num">{on_order}</td>'
                    f'<td class="buy {"zero" if not r.pull_in else ""}">{pull}</td>'
                    f'<td class="buy {"zero" if not r.net_new else ""}">{r.net_new or "–"}</td>'
                    + ''.join(quarter_cell(getattr(r, f'add_{q}'), getattr(r, f'load_{q}'), ceiling, getattr(r, f'after_{q}')) for q in K.Q)
                    + f'<td class="num">{getattr(r, "fleet_2027Q4")}</td></tr>')
    st.markdown(f'<div class="cap-wrap"><table class="cap"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="legend-row"><span>+n = new tools to bring into service that quarter</span>'
                f'<span>load before = on installed + on-order tools released by then · after = with the adds and pull-ins in place</span>'
                f'<span><i style="background:var(--warn-bg)"></i>{ceiling*100:.0f}–100%</span>'
                '<span><i style="background:var(--bad-bg)"></i>over 100%</span></div>', unsafe_allow_html=True)


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


def fleet_view():
    f = M.fleet_status(data, site)
    s = M.summary(data, site)
    cal = f.installed.sum()
    util = (f.productive_pct * f.installed).sum() / cal if cal else float('nan')
    hot = f[f.loading_pct >= ceiling * 100]
    q = f.sort_values('queue_h', ascending=False).iloc[0]
    section(f'Current fleet · 2026Q4 to date ({s["start"]:%d %b} – {s["end"]:%d %b})', 'Installed + on order · how each tool group spent its time (simulated)')
    stats([
        (f'{int(cal):,}', f'Tools installed · {int(f.on_order.sum())} on order', ''),
        (number(util, 0, '%'), 'Utilization · productive share of calendar time', ''),
        (f'{f.unsched_down_h.sum():,.0f} h', f'Unscheduled downtime · {int(f.events.sum())} events', ''),
        (f'{len(hot)} / {len(f)}', f'Tool groups at or above the {ceiling*100:.0f}% limit', 'warn' if len(hot) else ''),
        (f'{q.queue_h:.1f} h', f'Longest queue per visit · {group_label(q.site, q.tool_group, True)}', 'bad' if q.queue_h > 4 else ''),
    ])
    left, right = st.columns(2, gap='large')
    with left:
        section('Where tool time went', '% of calendar time by state · E10-style split')
        chart(PC.time_split(f, BOTH), 'fleet_time')
    with right:
        section(f'Production loading vs. {ceiling*100:.0f}% planning limit', 'Productive ÷ available hours · label = queue per visit')
        chart(PC.loading_queue(f, ceiling, BOTH), 'fleet_loading')
    section('Fleet register', 'Largest unscheduled downtime first · 2026Q4 to date')
    rows = ''
    for r in f.sort_values('unsched_down_h', ascending=False).itertuples():
        arrival = esc(r.on_order_label.replace(' in ', ' · ')) if r.on_order else '–'
        rows += (f'<tr>' + (f'<td class="l">{site_label(r.site)}</td>' if BOTH else '') +
                 f'<td class="l">{esc(rm.GROUP_LABELS[r.tool_group])}</td><td class="num">{r.installed}</td>'
                 f'<td class="num">{arrival}</td>'
                 f'<td class="num">{r.productive_pct:.0f}%</td><td class="num">{r.loading_pct:.0f}%</td>'
                 f'<td class="num">{r.avail_pct:.0f}% <span class="sub">({r.avail_gap_pts:+.1f})</span></td>'
                 f'<td class="num">{r.unsched_down_h:,.0f} h <span class="sub">· {r.events}</span></td>'
                 f'<td class="l cause">{esc(r.top_cause)}</td><td class="num">{r.queue_h:.1f}</td></tr>')
    st.markdown('<div class="cap-wrap"><table class="cap register"><thead><tr>' + ('<th class="l">Fab</th>' if BOTH else '') +
                '<th class="l">Tool group</th><th>Installed</th><th>On order</th><th>Utilization</th><th>Loading</th>'
                '<th>Availability<div class="sub">vs. plan</div></th><th>Unsched. down<div class="sub">· events</div></th>'
                '<th class="l">Main downtime cause</th><th>Queue<div class="sub">h / visit</div></th></tr></thead><tbody>' + rows + '</tbody></table></div>',
                unsafe_allow_html=True)
    st.caption('Utilization = productive ÷ calendar time. Loading = productive ÷ hours the tools were up and released for production '
               '(calendar − PM − unscheduled down − engineering). Events and causes come from the Q4 tool logbook.')
    section('Availability: measured vs. planning assumption', 'What the 2027 plan assumes vs. what the tools delivered · 2026Q4 to date')
    chart(PC.availability(f.rename(columns={'avail_pct': 'observed_avail_pct'}), BOTH), 'fleet_availability')
    st.download_button('Download fleet register (CSV)', csv(f.round(2)), 'td_fleet_2026q4_to_date.csv', 'text/csv')


def outlook_view():
    top, side = st.columns([3, 1.1], vertical_alignment='bottom')
    with side:
        basis = st.radio('Size the plan to', ['requested', 'expected'], horizontal=True, key='capital_basis',
                         format_func={'requested': 'Requested starts', 'expected': 'Realization-adjusted starts'}.get,
                         help='Realization-adjusted = request × the share each program has historically run (Demand page). '
                              'It is not checked against capacity: it is the demand the fleet has to carry.')
    frame = cached_buy(FP, pilot_case, basis, ceiling, avail)
    view = K.scope(frame, site)
    with top:
        section('2027 baseline · tools the demand profile needs vs. current fleet', '2027 plan by quarter · installed + on order · by fab and tool group')
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
        section('New tools by quarter vs. installed + on-order fleet', 'Buy = 2027 total · each quarter: tools to add, load before → after adding them')
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
        section('Required vs. available tool-hours by program', 'Weekly · 2027 by quarter')
        s, grp = st.selectbox('Tool group', options, key='driver_group', format_func=lambda k: group_label(*k, True),
                              label_visibility='collapsed')
        state = K.plan_state(data, pilot_case, basis, avail)
        chart(PC.drivers(state, s, grp, ceiling), 'capital_drivers')
    row = view[(view.site == s) & (view.tool_group == grp)].iloc[0]
    qi = int(pd.Series([row[f'load_{q}'] for q in K.Q]).fillna(1e9).values.argmax())
    q = K.Q[qi]
    hours = program_hours(state, s, grp, qi)
    per = 168 * state['avail'][s][grp][qi]
    tools = K.timed_tools()[K.SITES.index(s), K.GROUPS.index(grp), qi]
    required = hours['Hours / wk'].sum()
    with right:
        section(f'Calculation · {q}', group_label(s, grp, True))
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
              'on_order_label': 'Arrivals', 'needed_peak': 'Peak need', 'net_new': 'Buy 2027', 'pull_in': 'Pull in',
              'pull_in_by': 'Pull in to', **{f'add_{q}': f'Add {q}' for q in K.Q}, **{f'load_{q}': f'Load {q} (%)' for q in K.Q}}
    export = view[list(labels)].assign(planning_limit_pct=ceiling * 100, basis=basis, plan_case=pilot_case, availability=avail)
    st.download_button('Download tool requirement (CSV)', csv(export, labels), 'td_2027_tool_requirement.csv', 'text/csv')
    ai_panel('ai_capital', dict(view='2027 capacity baseline', site=site, pilot_case=pilot_case, planning_limit=ceiling, demand_basis=basis,
                                fleet='current', availability=avail,
                                totals=t, headline=K.headline(view, ceiling, basis).replace('<b>', '').replace('</b>', ''),
                                tool_requirement=export.round(1).to_dict('records')), [
        'Which tool groups drive the 2027 buy, and which programs create that load?',
        'If one EUV tool becomes unavailable at Fab A during 2027Q3, what happens to capacity?',
        'Could qualified DRAM Node A work move to Fab B, and would that create another constraint?'])


# ================================================================== DEMAND OUTLOOK (the opener)
def demand_page():
    vf = PCX.versions_frame(data)
    vf = vf if BOTH else vf[vf.site == site]
    history = em.historical_attainment(data, site).set_index('quarter')
    hp = data['wafer_start_history'] if BOTH else data['wafer_start_history'].query('site==@site')
    bridge = data['demand_bridge_2026q4'] if BOTH else data['demand_bridge_2026q4'].query('site==@site')
    q4 = dict(requested_wspw=float(bridge.requested_wspw.sum()), forecast_wspw=float(bridge.forecast_wspw.sum()))
    held, metrics = demand.validation(data, site)
    short = PCX.VERSION_SHORT
    cur, today = PCX.CURRENT_VERSION, PCX.TODAY_VERSION
    tot = vf.groupby(['version', 'quarter'])[['requested_wspw', 'expected_wspw']].sum()
    # 2026 average (2026Q1–Q3 requests + 2026Q4 request) vs. 2027 average of the 2027 plan, requested starts / week
    h26 = hp[hp.quarter.str.startswith('2026')].groupby(['program', 'quarter']).requested_wspw.sum().unstack()
    h26['2026Q4'] = bridge.groupby('program').requested_wspw.sum()
    avg26 = h26.fillna(0.).mean(axis=1)
    avg27 = vf[vf.version == cur].groupby('program').requested_wspw.sum() / len(C.PLAN_QUARTERS)
    prog = pd.DataFrame({'y26': avg26, 'y27': avg27}).fillna(0.)
    prog = prog[prog.y26 > 0]
    prog['chg'] = prog.y27 - prog.y26
    prog['pct'] = 100 * prog.chg / prog.y26
    t26, t27 = float(prog.y26.sum()), float(avg27.sum())
    up, down = prog.pct.idxmax(), prog.pct.idxmin()

    def signed(v):
        v = round(float(v))
        return f'{v:+d}%' if v else '0%'
    page_head('', 'Demand outlook for 2027')
    stats([
        (number(t26), '2026 average requested starts / week', ''),
        (number(t27), '2027 average requested starts / week', ''),
        (signed(100 * (t27 / t26 - 1)), f'Request growth · 2026 avg → 2027 avg ({t27 - t26:+,.0f} / week)', ''),
        (signed(prog.pct[up]), f'Largest increase · {D.NAMES[up]} ({prog.chg[up]:+,.0f} / week)', ''),
        (signed(prog.pct[down]), f'{"Largest drop" if prog.pct[down] < 0 else "Smallest increase"} · {D.NAMES[down]} ({prog.chg[down]:+,.0f} / week)',
         'bad' if prog.pct[down] < 0 else ''),
    ])
    with st.container(key='demand_controls'):
        a, b, _ = st.columns([1, 1.25, 2.4], vertical_alignment='bottom')
        measure = a.radio('Measure', ['Requested', 'Realization-adjusted'], horizontal=True, key='demand_measure',
                          help='Request × each program\'s historical share actually started. Before capacity limits.')
        group = b.radio('Group by', ['Program', 'Family', 'Fab'], horizontal=True, key='demand_split')
    col = 'requested_wspw' if measure == 'Requested' else 'expected_wspw'

    section('Requested vs. actual wafer starts · 2024Q1–2027Q4', 'Quarterly average per week · starts: actual to 2026Q3, forecast from 2026Q4')
    chart(PC.version_trend(history[['requested_wspw', 'actual_wspw']], q4, vf, col, [cur]), 'dm_trend')

    section('Wafer starts by program and quarter · 2027 plan vs. 2026Q4', f'{measure} starts / week · value and change vs. 2026Q4')
    st.markdown(demand_matrix(vf, today, cur, col, group), unsafe_allow_html=True)

    section('Requested vs. actual starts by program · 2024–2027', 'Black = starts (actual → forecast) · grey = requested history · blue = 2027 request')
    chart(PC.program_multiples(hp, bridge, vf, col, [cur]), 'dm_multiples')

    left, right = st.columns([1.2, 1], gap='large')
    with left:
        section(f'2027 wafer-start mix by {group.lower()}', f'{measure} starts / week')
        chart(PC.version_mix(vf[vf.version == cur], col, group), 'dm_mix')
    with right:
        section('Actual vs. requested starts by program', 'Realization factor · fit 2024Q1–2026Q3')
        chart(PC.realization(fm.fit(data['wafer_start_history'])), 'demand_realization')

    section('Program priorities and milestones', f'{measure} starts / week')
    meta = data['program_planning'].set_index('program')
    by_prog = vf.groupby(['version', 'program'])[col].sum().unstack(0) / 4
    rows = []
    for p_ in C.PROGRAM_ORDER:
        if p_ not in by_prog.index:
            continue
        m = meta.loc[p_]
        date = m.milestone_date if isinstance(m.milestone_date, str) and m.milestone_date else '—'
        now, plan_ = by_prog.loc[p_, today], by_prog.loc[p_, cur]
        rows.append(f'<tr><td class="l">{esc(D.NAMES[p_])}</td><td class="num">{int(m.priority)}</td><td class="l">{esc(m.priority_label)}</td>'
                    f'<td class="l">{esc(m.milestone)}</td><td class="num">{esc(date)}</td>'
                    f'<td class="l">{"<span class=pill-prot>Protected</span>" if bool(m.protected) else ""}</td>'
                    f'<td class="num">{now:,.0f}</td><td class="num">{plan_:,.0f}</td><td class="num">{100*(plan_/now-1):+.0f}%</td></tr>')
    st.markdown('<div class="cap-wrap"><table class="cap"><thead><tr><th class="l">Program</th><th>Priority</th><th class="l">Role</th>'
                '<th class="l">Milestone</th><th>Date</th><th class="l"></th><th>2026Q4</th><th>2027 avg</th><th>Change</th>'
                '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>', unsafe_allow_html=True)

    with st.expander('How the demand numbers are built'):
        st.markdown(
            f'<p class="note"><b>Realization factor</b> = least-squares slope of actual on requested starts, through the origin, per program. '
            f'On quarters the fit never saw (2026Q1–Q3) the raw request missed actual starts by {number(metrics["raw_mape"],1,"%")}; '
            f'request × factor missed by {number(metrics["expected_mape"],1,"%")} across {metrics["rows"]} fab-program-quarters. Realization-adjusted = request × factor (before capacity limits).</p>'
            '<p class="note"><b>2026Q4</b> is in progress, so it has no actuals: each fab/program continues its own last-four-quarter trend, '
            'for requests and for starts. It is the comparison base for 2027 and never enters the fit.</p>'
            '<p class="note"><b>2027 plan.</b> Requests are on the same basis as historical requests (planned run rate ÷ realization factor), '
            'so the series runs on from 2026 without an artificial step.</p>',
            unsafe_allow_html=True)
        out = vf[vf.version == cur].assign(Fab=lambda f: f.site.map(rm.SITE_LABELS), Program=lambda f: f.program.map(D.NAMES))
        cols = {'quarter': 'Quarter', 'Fab': 'Fab', 'Program': 'Program', 'requested_wspw': 'Requested WSPW',
                'expected_wspw': 'Realization-adjusted WSPW', 'realization_rate': 'Realization factor'}
        st.dataframe(out[list(cols)].rename(columns=cols).round(1), hide_index=True, width='stretch')
        st.download_button('Download 2027 plan (CSV)', csv(out[list(cols)], cols), 'td_2027_plan.csv', 'text/csv')


def demand_matrix(vf, base, target, col, group):
    key = {'Program': 'program', 'Family': 'technology_family', 'Fab': 'site'}[group]
    label = {'program': D.NAMES.get, 'technology_family': str, 'site': rm.SITE_LABELS.get}[key]
    t = vf[vf.version == target].groupby([key, 'quarter'])[col].sum().unstack()
    b = vf[vf.version == base].groupby([key, 'quarter'])[col].sum().unstack().reindex_like(t).fillna(0)
    order = [p_ for p_ in C.PROGRAM_ORDER if p_ in t.index] if key == 'program' else list(t.index)
    t, b = t.reindex(order), b.reindex(order)
    t.loc['Total'], b.loc['Total'] = t.sum(), b.sum()

    def cell(v, d, total=False):
        pct = 100 * d / (v - d) if v - d else 0
        tone = ('up3' if pct >= 25 else 'up2' if pct >= 10 else 'up1' if pct >= 3 else
                'dn3' if pct <= -25 else 'dn2' if pct <= -10 else 'dn1' if pct <= -3 else '')
        return (f'<td class="mx {tone}{" tot" if total else ""}"><span class="v">{v:,.0f}</span>'
                f'<span class="d">{d:+,.0f} · {pct:+.0f}%</span></td>')
    head = '<tr><th class="l">' + ('Program' if key == 'program' else 'Group') + '</th>' + ''.join(f'<th>{q}</th>' for q in C.PLAN_QUARTERS) + '<th>2027 avg</th></tr>'
    body = []
    for k in t.index:
        total = k == 'Total'
        name = 'Total' if total else label(k)
        cells = ''.join(cell(t.loc[k, q], t.loc[k, q] - b.loc[k, q], total) for q in C.PLAN_QUARTERS)
        cells += cell(t.loc[k].mean(), t.loc[k].mean() - b.loc[k].mean(), total)
        body.append(f'<tr{" class=total" if total else ""}><td class="l">{esc(name)}</td>{cells}</tr>')
    return (f'<div class="cap-wrap"><table class="cap matrix"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'
            '<div class="legend-row">' + ''.join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in
            [('dn3', '−25% or more'), ('dn2', '−10 to −25%'), ('dn1', '−3 to −10%'), ('', '±3%'),
             ('up1', '+3 to +10%'), ('up2', '+10 to +25%'), ('up3', '+25% or more')]) + '</div>')


# ================================================================== RESPONSE PLAN
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def cached_levers(fingerprint, basis, ceiling_, availability, downtime_cut, max_share):
    return K.response_levers(load_data(fingerprint), 'baseline', basis, ceiling_, availability, downtime_cut, max_share)


def response_view():
    a, b, c, _ = st.columns([1.1, 1.1, 1.1, 1.4])
    basis = a.selectbox('Size to', ['requested', 'expected'], key='resp_basis',
                        format_func={'requested': 'Requested starts', 'expected': 'Realization-adjusted starts'}.get)
    cut = b.selectbox('Unscheduled downtime cut', [0, 10, 25, 50], index=2, key='resp_cut', format_func=lambda v: f'{v}%',
                      help='Reliability lever: the share of unscheduled downtime engineering recovers in 2027. The hours return as production time.')
    max_share = c.selectbox('Most of a program that may move fabs', [25, 50, 100], index=1, key='resp_share', format_func=lambda v: f'{v}%')
    r = cached_levers(FP, basis, ceiling, avail, cut, max_share)
    g = r['groups'] if BOTH else r['groups'][r['groups'].site == site]
    base, down, reb, buy = int(g.baseline.sum()), int(g.downtime_saves.sum()), int(g.rebalance_saves.sum()), int(g.buy.sum())
    cost = lambda col: float((g[col] * g.cost_each).sum())
    bl = r['buy'] if BOTH else r['buy'][r['buy'].site == site]
    section('Closing the 2027 gap · cheapest levers first', 'Start from the 2027 baseline buy, then take each lever before buying')
    stats([
        (f'{base}', f'2027 baseline buy · cost index {cost("baseline"):,.0f}', ''),
        (f'{-down:+d}', f'Cut unscheduled downtime {cut}%', ''),
        (f'{-reb:+d}', 'Rebalance work across fabs', ''),
        (f'{buy}', f'Tools to buy · cost index {cost("buy"):,.0f}', 'warn'),
        (f'{int(bl.pull_in.sum())}', 'On-order tools to pull in', ''),
    ])
    left, right = st.columns([1, 1.25], gap='large')
    with left:
        section('Net-new tools · baseline → levers → buy', f'At the {ceiling*100:.0f}% planning limit')
        wf = pd.DataFrame(dict(step=['2027 baseline', f'Downtime −{cut}%', 'Rebalance fabs'], net_new=[base, base - down, base - down - reb]))
        chart(PC.waterfall(wf, final='Tools to buy'), 'response_waterfall')
    with right:
        section('Where each lever acts', 'Tools by fab and tool group · unscheduled downtime from 2026Q4 performance')
        rows = ''
        meas = K.unscheduled_share(data, 'measured')
        show = g[(g.baseline > 0) | (g.buy > 0)].sort_values(['buy', 'baseline'], ascending=False)
        for x in show.itertuples():
            fmt = lambda v: f'{-v:+d}' if v else '–'
            rows += (f'<tr>' + (f'<td class="l">{site_label(x.site)}</td>' if BOTH else '') +
                     f'<td class="l">{esc(rm.GROUP_LABELS[x.tool_group])}</td>'
                     f'<td class="num">{100*meas.get((x.site, x.tool_group), 0):.1f}%</td>'
                     f'<td class="num">{x.baseline}</td><td class="num">{fmt(x.downtime_saves)}</td><td class="num">{fmt(x.rebalance_saves)}</td>'
                     f'<td class="num"><b>{x.buy}</b></td></tr>')
        st.markdown('<div class="cap-wrap"><table class="cap"><thead><tr>' + ('<th class="l">Fab</th>' if BOTH else '') +
                    '<th class="l">Tool group</th><th>Unsched. down<div class="sub">Q4 to date</div></th><th>Baseline</th>'
                    '<th>Downtime cut</th><th>Rebalance</th><th>Buy</th></tr></thead><tbody>' + rows + '</tbody></table></div>',
                    unsafe_allow_html=True)
    moves = r['transfer']['moves']
    section('Rebalancing · work moved between fabs', f'{r["transfer"]["searched"]:,} combinations searched · up to {max_share}% of a dual-qualified program · lowest cost index')
    mrows = ''.join(f'<tr><td class="l">{esc(D.NAMES[m["program"]])}</td><td class="l">{site_label(m["from_site"])} → {site_label(m["to_site"])}</td>'
                    f'<td class="num">{m["share_pct"]}%</td><td class="num">{m["avg_wspw"]:,}</td></tr>' for m in moves) or '<tr><td class="l" colspan="4">No move lowers the cost index</td></tr>'
    st.markdown('<div class="cap-wrap narrow"><table class="cap"><thead><tr><th class="l">Program</th><th class="l">Move</th><th>Share</th>'
                f'<th>Starts / week</th></tr></thead><tbody>{mrows}</tbody></table></div>', unsafe_allow_html=True)
    st.caption('A move frees tools at one fab and can add some at the other (negative numbers under Rebalance); it is kept only if the '
               'total cost index falls. EUV work cannot move: Fab B has no EUV.')
    section('Buy list · tools to add by quarter after the levers', 'Load before → after the adds and pull-ins')
    capital_table(bl, False)
    st.caption('Cost index: illustrative relative cost per tool (wet clean = 1, EUV = 18), used only to rank responses; not a price. '
               'Downtime cut and rebalancing are assumptions to be agreed with equipment engineering and the program owners.')
    export = bl[['site', 'tool_group', 'installed', 'on_order', 'needed_peak', 'net_new', 'pull_in', 'pull_in_by', *[f'add_{q}' for q in K.Q], 'cost_index']]
    st.download_button('Download response buy list (CSV)', csv(export), 'td_2027_response_buy_list.csv', 'text/csv')


# ================================================================== 4 · SCENARIOS
# ================================================================== FAB PERFORMANCE (2026Q4 to date)
def fab_performance_page():
    s = M.summary(data, site)
    tl = M.tools(data, site)
    above = tl[tl.loading_pct >= ceiling * 100]
    page_head(f'{ui.SITES[site].upper()} · 2026Q4 TO DATE · {s["start"]:%d %b} – {s["end"]:%d %b %Y} · SIMULATED', 'Fab performance')
    stats([
        (f'{s["weeks_done"]} / 13', 'Weeks of 2026Q4 completed', ''),
        (number(s['started_total']), f'Wafers started so far · {100*s["started_total"]/s["quarter_total_plan"]:.0f}% of Q4 total ({s["quarter_total_plan"]:,.0f})', ''),
        (number(s['starts_wspw']), f'Avg starts / week · {s["attainment_pct"]:.0f}% of requested weekly rate', 'warn' if s['attainment_pct'] < 95 else ''),
        (number(s['wip']), f'WIP · wafers · {number(s["outs_wspw"])} completed / week', ''),
        (f'{s["cycle_days"]:.1f} d', f'Cycle time · p90 {s["cycle_p90"]:.1f} d · yield {s["yield_pct"]:.1f}%', ''),
    ])
    left, right = st.columns([1.2, 1], gap='large')
    with left:
        section('Wafer starts vs. completions vs. 2026Q4 request', 'Weekly · wafers')
        chart(PC.month_flow(s['weekly'], s['plan_wspw']), 'month_flow')
    with right:
        section('WIP and cycle time', 'Weekly')
        chart(PC.month_wip_ct(s['weekly']), 'month_wip_ct')
    st.caption('2026Q4 to date is a simulated continuation of the same fab model (seed 42): rerunning it reproduces the stored 52-week history exactly, '
               'then adds 28 Sep – 25 Oct 2026. Request = 2026Q4 requested starts. Tool loading, downtime and availability are on '
               'Capacity planning → Current fleet.')
    fab_assistant(s, tl)


def fab_assistant(s, tl):
    ev = data['tool_events_q4']
    ev = ev if BOTH else ev[ev.site == site]
    by_group = (ev.groupby(['site', 'tool_group']).agg(events=('event_id', 'size'), down_hours=('down_hours', 'sum'))
                .reset_index().sort_values('down_hours', ascending=False))
    by_cause = (ev.groupby(['site', 'tool_group', 'cause']).down_hours.sum().round(1).reset_index()
                .sort_values('down_hours', ascending=False).head(12))
    context = dict(view='Fab performance · 2026Q4 to date (simulated)', site=site, planning_limit=ceiling, fleet='current', availability='measured',
                   period=f'{s["start"]:%d %b} – {s["end"]:%d %b %Y}', weeks_done=s['weeks_done'],
                   line=dict(starts_per_week=round(s['starts_wspw']), completions_per_week=round(s['outs_wspw']), wip=round(s['wip']),
                             cycle_time_days=round(s['cycle_days'], 1), cycle_time_p90_days=round(s['cycle_p90'], 1), line_yield_pct=round(s['yield_pct'], 1)),
                   tool_groups=tl.round(1).to_dict('records'),
                   unscheduled_down_by_group=by_group.round(1).to_dict('records'),
                   top_causes_by_hours=by_cause.to_dict('records'),
                   data_available='Table tool_events_q4 has one row per unscheduled-down event (tool_id, start_time, down_hours, cause, action). '
                                  'Weekly equipment reviews for each fab are in the notes (tool_logs/2026Q4).')
    section('Ask about the fab', 'Searches the Q4 tool logbook and event table · answers cite their sources')
    ai_panel('ai_fab', context, [
        'What problems did the deposition tools at Fab A have in Q4 so far, and how many hours were lost?',
        'Which tool groups lost the most time to unscheduled downtime, and what were the main causes?',
        'Which single tools went down most often, and for what reasons?',
        'Why is measured availability below the planning assumption for metal deposition at Fab A?',
        'Summarize the equipment review for Fab B for the week of 12 Oct.'],
        title='Fab assistant · tool logbook and events', expanded=False,
        note='Retrieves the weekly equipment reviews and queries the event table (read-only SQL), then answers with citations. '
             'The data is a simulated continuation of the fab model; every number must come from a cited source.')


# ================================================================== CAPACITY SCENARIO PLANNING · SCENARIOS
def scenario_test():
    programs = em.program_options(site, 'All families', include_pilot=False)
    quarter = '2027Q3'
    a, b = st.columns([1, 1.4])
    operation = a.selectbox('Test', request_model.OPERATIONS, key='scenario_operation')
    fleet = b.radio('Tools assumed', ['current', 'planned'], format_func=request_model.FLEETS.get, horizontal=True, key='scenario_fleet',
                    help='Current fleet = installed + on order. With planned 2027 purchases = the current fleet plus the buys from the 2027 baseline tab, each added in the quarter it is needed.')
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
        pkey = 'mix_from' if operation == 'Product mix shift' else 'scenario_program'
        default = 'NAND_C' if operation == 'Product mix shift' else 'DRAM_B'
        if ss.get(pkey) not in eligible:
            ss[pkey] = default if default in eligible else eligible[0]
        program = b.selectbox('Shift from' if operation == 'Product mix shift' else 'Program', eligible, format_func=ec.PROGRAM_SHORT.get, key=pkey)
    to_program, wafers = None, 0.
    if operation == 'Product mix shift':
        targets = [p for p in C.QUALIFIED[request_site] if p != program]
        if ss.get('scenario_to_program') not in targets:
            ss['scenario_to_program'] = 'HBM_TV' if 'HBM_TV' in targets else targets[0]
        to_program = c.selectbox('Shift to', targets, format_func=ec.PROGRAM_SHORT.get, key='scenario_to_program')
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
    elif operation == 'Product mix shift':
        wafers = st.number_input('Wafers / week to shift (total starts unchanged)', min_value=5., max_value=1000., value=50., step=5., key='mix_wafers')
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
    recovery = st.checkbox('Also test an availability recovery', key='recovery_enabled') if operation == 'Demand growth' else False
    recovery_group, points = None, 0.
    if recovery:
        a, b = st.columns(2)
        recovery_group = a.selectbox('Equipment to recover', [g_ for g_ in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g_] > 0], format_func=rm.GROUP_LABELS.get, key='recovery_group_' + request_site)
        points = b.number_input('Recovery (percentage points)', min_value=.5, max_value=15., value=5., step=.5, key='recovery_points')
    signature = (site, pilot_case, operation, request_site, program, pct, target, timing, first_period, last_period, ceiling, recovery_group, points,
                 outage_group, unavailable, fleet, to_program, wafers, avail)
    if st.button('Run scenario', type='primary'):
        try:
            review = request_model.run(data, operation, request_site, program, pct, outage_group, unavailable, to_site, first_period, last_period,
                                       'weeks' if timing == 'Week range' else 'quarters', ceiling, recovery_group, points, target, pilot_case,
                                       fleet=fleet, to_program=to_program, wafers=wafers, availability=avail)
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
    request_name = (rm.GROUP_LABELS[outage_group] + ' unavailable' if operation == 'Tool unavailability'
                    else f'{wafers:,.0f} WSPW {ec.PROGRAM_SHORT[program]} → {ec.PROGRAM_SHORT[to_program]}' if operation == 'Product mix shift'
                    else ec.PROGRAM_SHORT[program])
    win = request_model.window_quarters(review, result_scope)
    qs = list(dict.fromkeys(win.quarter))
    states = (review['base_state'], review['scenario_state'])
    sites = list(C.SITES) if result_scope == 'Both' else [result_scope]
    starts = [sum(st_['wspw'][s_][p_][C.PLAN_QUARTERS.index(q)] for s_ in sites for p_ in st_['wspw'][s_] for q in qs) / len(qs)
              for st_ in states]
    if timing == 'Week range':
        sb, sa = review['starts_before'], review['starts_after']
        inside = sb.week_start.between(first_week, last_week) & sb.site.isin(sites)
        starts = [sb[inside].groupby('week_start').starts_weekly_equiv.sum().mean(),
                  sa[sa.week_start.between(first_week, last_week) & sa.site.isin(sites)].groupby('week_start').starts_weekly_equiv.sum().mean()]
    peak = win.assign(_k=win.loading_pct_scenario.fillna(np.inf)).sort_values('_k', ascending=False).iloc[0]
    over = lambda col: win[(win[col] > 100) | win[col].isna()].drop_duplicates(['site', 'tool_group']).shape[0]
    tools_g = win.groupby(['site', 'tool_group']).agg(fleet_b=('tools_base', 'min'), fleet_s=('tools_scenario', 'min'),
                                                      need_b=('need_base', 'max'), need_s=('need_scenario', 'max'),
                                                      short_b=('short_base', 'max'), short_s=('short_scenario', 'max'))
    tools_g['extra'] = tools_g.short_s - tools_g.short_b
    extra = int(tools_g.extra.sum())
    section(site_label(request_site) + (' → ' + site_label(to_site) if operation == 'Qualified load transfer' else '') + ' · ' + request_name,
            period_label + ' · ' + ('both fabs' if include_other else site_label(request_site) + ' only') + ' · ' + request_model.FLEETS[fleet].lower())
    stats([
        (f"{starts[0]:,.0f} → {starts[1]:,.0f}", f'Starts / week · average over {", ".join(qs)}', ''),
        ('no capacity' if pd.isna(peak.loading_pct_scenario) else number(peak.loading_pct_scenario, 1, '%'),
         f'Peak scenario load · {group_label(peak.site, peak.tool_group, True)} {peak.quarter}',
         'bad' if pd.isna(peak.loading_pct_scenario) or peak.loading_pct_scenario > 100 else 'warn' if peak.loading_pct_scenario >= ceiling * 100 else ''),
        (f'{over("loading_pct_base")} → {over("loading_pct_scenario")}', 'Tool groups over 100% in the window', 'bad' if over('loading_pct_scenario') else ''),
        (f'{extra:+d}', f'Extra tools the scenario needs to stay ≤ {ceiling*100:.0f}%', 'warn' if extra > 0 else ''),
    ])
    section('Baseline vs. scenario load by quarter', 'Tool groups the scenario changes, plus any at or above the limit'
            + (' · week range: peak week inside the window' if timing == 'Week range' else ''))
    cols = st.columns(2 if len(qs) > 1 else 1, gap='large')
    for k, q in enumerate(qs):
        f = win[win.quarter == q]
        f = f[(f.change_pp.abs() > .05) | (f.loading_pct_scenario >= ceiling * 100) | f.loading_pct_scenario.isna()]
        f = f.assign(_k=f.change_pp.abs().fillna(999)).sort_values(['_k', 'loading_pct_scenario'], ascending=False).head(8)
        with cols[k % len(cols)]:
            st.markdown(f'<div class="section"><span class="section-title">{q}</span></div>', unsafe_allow_html=True)
            if f.empty:
                st.caption('No tool group changes or reaches the limit.')
            else:
                chart(PC.scenario_delta(f, ceiling, result_scope == 'Both'), f'scenario_q_{q}')
    with st.expander('All tool groups and applied assumptions'):
        frame = win.assign(Fab=win.site.map(rm.SITE_LABELS), Equipment=win.tool_group.map(rm.GROUP_LABELS))
        st.dataframe(frame[['quarter', 'Fab', 'Equipment', 'loading_pct_base', 'loading_pct_scenario', 'change_pp', 'need_base', 'need_scenario']].rename(columns={
            'quarter': 'Quarter', 'loading_pct_base': 'Baseline load (%)', 'loading_pct_scenario': 'Scenario load (%)', 'change_pp': 'Change (pts)',
            'need_base': 'Tools needed · baseline', 'need_scenario': 'Tools needed · scenario'}).round(1), hide_index=True, width='stretch')
        st.write(review['applied'])
    exported = win.assign(request_site=request_site, request_program=program, timing_mode=review['timing_mode'], requested_window=period_label,
                          operation=operation, fleet=fleet, availability=avail)
    st.download_button('Download scenario (CSV)', csv(exported), f'{result_scope}_scenario.csv', 'text/csv')
    ai_panel('ai_scenario', dict(view='Live capacity scenario', site=result_scope, pilot_case=pilot_case, planning_limit=ceiling,
                                 fleet=fleet, availability=avail, demand_basis='requested',
                                 request=dict(operation=operation, site=request_site, program=program, period=period_label,
                                              timing='weeks' if timing == 'Week range' else 'quarters', first=str(first_period), last=str(last_period),
                                              pct=pct, target_wspw=target, recovery_group=recovery_group, recovery_points=points,
                                              tool_group=outage_group if operation == 'Tool unavailability' else None,
                                              tools_unavailable=unavailable if operation == 'Tool unavailability' else None,
                                              to_program=to_program, wafers=wafers),
                                 quarters=qs, starts_before=round(float(starts[0])), starts_after=round(float(starts[1])), extra_tools_to_stay_at_limit=extra,
                                 by_quarter=win.round(2).to_dict('records')), [
        'Which named equipment constrains this request, and which documented response should we examine?',
        'Can we consolidate qualified work into the other fab without simply moving the constraint?',
        'Which tool groups change most in this scenario, and what do the dated operating notes say about them?'])


def scenarios_view():
    scenario_test()


CAP_VIEWS = ['Current fleet', '2027 baseline', 'Scenario analysis', 'Response plan']


def capacity_page():
    page_head('', 'Capacity planning')
    view = st.radio('View', CAP_VIEWS, horizontal=True, key='cap_view', label_visibility='collapsed')
    if view == 'Current fleet':
        fleet_view()
    elif view == '2027 baseline':
        outlook_view()
    elif view == 'Scenario analysis':
        scenarios_view()
    else:
        response_view()


PAGES = {'Demand outlook': demand_page, 'Fab performance': fab_performance_page, 'Capacity planning': capacity_page}
PAGES[page]()
st.markdown('<div class="footer"><span>All fab data, routes, tool counts and plans are synthetic. No company data is used.</span>'
            '<span>Method notes: MODEL_GUIDE.md · VIEW_GUIDE.md</span></div>', unsafe_allow_html=True)
