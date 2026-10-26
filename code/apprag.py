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
import fab_month as M
import forecast_model as fm
import launch_charts as LC
import launch_model as L
import plan_charts as PC
import planning_copilot as rag
import planning_context as PCX
import request_parser as P
import review_model as rm
import scenario_requests as request_model
import wafer_analyst as WA
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


def ai_panel(key, context, examples, title='Ask the planning assistant', expanded=False):
    with st.expander(title, expanded=expanded):
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
def quarter_cell(add, value, ceiling):
    tone = 'bad' if pd.isna(value) or value > 100 else 'warn' if value >= ceiling * 100 else 'ok'
    load = 'no tool' if pd.isna(value) else f'{value:.0f}%'
    tools = f'+{add}' if add else '–'
    return f'<td class="q {tone}"><span class="qa{" zero" if not add else ""}">{tools}</span><span class="ql">{load}</span></td>'


def capital_table(view, show_covered):
    rows = view if show_covered else view[view.action.ne('Covered')]
    rows = rows.assign(_c=rows.action.eq('Covered')).sort_values(['_c', 'net_new', 'pull_in', 'peak_loading_pct'],
                                                                 ascending=[True, False, False, False])
    head = ('<tr>' + ('<th class="l">Fab</th>' if BOTH else '') + '<th class="l">Tool group</th><th>Installed</th>'
            '<th>On order</th><th>Pull in</th><th>Buy · 2027</th>'
            + ''.join(f'<th>{q}<div class="sub">add · load</div></th>' for q in K.Q) + '<th>Fleet by Q4</th></tr>')
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
                    + ''.join(quarter_cell(getattr(r, f'add_{q}'), getattr(r, f'load_{q}'), ceiling) for q in K.Q)
                    + f'<td class="num">{getattr(r, "fleet_2027Q4")}</td></tr>')
    st.markdown(f'<div class="cap-wrap"><table class="cap"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="legend-row"><span>+n = new tools to bring into service that quarter</span>'
                f'<span>load = required hours ÷ hours from installed + on-order tools released by then</span>'
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


def outlook_view():
    top, side = st.columns([3, 1.1], vertical_alignment='bottom')
    with side:
        basis = st.radio('Size the plan to', ['requested', 'expected'], horizontal=True, key='capital_basis',
                         format_func={'requested': 'Requested starts', 'expected': 'Expected starts'}.get,
                         help='Expected = request × the share each program has historically run (Demand page).')
    frame = cached_buy(FP, pilot_case, basis, ceiling)
    view = K.scope(frame, site)
    with top:
        section('2027 tool requirement vs. installed + on-order fleet', '2027 plan · by fab and tool group')
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
        section('New tools by quarter vs. installed + on-order fleet', 'Buy = 2027 total · each quarter shows tools to add and load before adding them')
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
    export = view[list(labels)].assign(planning_limit_pct=ceiling * 100, basis=basis, plan_case=pilot_case)
    st.download_button('Download tool requirement (CSV)', csv(export, labels), 'td_2027_tool_requirement.csv', 'text/csv')
    ai_panel('ai_capital', dict(view='2027 capacity outlook', site=site, pilot_case=pilot_case, planning_limit=ceiling, demand_basis=basis,
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
    q3 = history.iloc[-1]
    req4, exp4 = tot.loc[cur].requested_wspw.iloc[-1], tot.loc[cur].expected_wspw.iloc[-1]
    page_head(f'{ui.SITES[site].upper()} · HISTORY 2024Q1–2026Q3 · 2026Q4 FORECAST · 2027 PLAN', 'Demand outlook')
    stats([
        (number(q3.actual_wspw), '2026Q3 actual starts / week', ''),
        (number(q4['forecast_wspw']), '2026Q4 forecast starts / week', ''),
        (number(req4), '2027Q4 requested starts / week', ''),
        (number(exp4), '2027Q4 expected starts / week', ''),
        (number(100 * (req4 / q4['requested_wspw'] - 1), 0, '%'), 'Request growth · 2026Q4 → 2027Q4', ''),
        (f"{number(metrics['raw_mape'],0,'%')} → {number(metrics['expected_mape'],0,'%')}", 'Backtest error · raw → adjusted', ''),
    ])
    with st.container(key='demand_controls'):
        a, b, _ = st.columns([1, 1.25, 2.4], vertical_alignment='bottom')
        measure = a.radio('Measure', ['Requested', 'Expected'], horizontal=True, key='demand_measure')
        group = b.radio('Group by', ['Program', 'Family', 'Fab'], horizontal=True, key='demand_split')
    col = 'requested_wspw' if measure == 'Requested' else 'expected_wspw'

    left, right = st.columns([1.45, 1], gap='large')
    with left:
        section('Requested vs. actual wafer starts · 2024Q1–2027Q4', 'Quarterly average per week · starts: actual to 2026Q3, forecast from 2026Q4')
        chart(PC.version_trend(history[['requested_wspw', 'actual_wspw']], q4, vf, col, [cur]), 'dm_trend')
    with right:
        section('Demand bridge · 2026Q4 → 2027 average', f'{measure} starts / week by program')
        chart(PC.version_bridge(vf, today, cur, col), 'dm_bridge')

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
            f'request × factor missed by {number(metrics["expected_mape"],1,"%")} across {metrics["rows"]} fab-program-quarters. Expected = request × factor.</p>'
            '<p class="note"><b>2026Q4</b> is in progress, so it has no actuals: each fab/program continues its own last-four-quarter trend, '
            'for requests and for starts. It is the comparison base for 2027 and never enters the fit.</p>'
            '<p class="note"><b>2027 plan.</b> Requests are on the same basis as historical requests (planned run rate ÷ realization factor), '
            'so the series runs on from 2026 without an artificial step.</p>',
            unsafe_allow_html=True)
        out = vf[vf.version == cur].assign(Fab=lambda f: f.site.map(rm.SITE_LABELS), Program=lambda f: f.program.map(D.NAMES))
        cols = {'quarter': 'Quarter', 'Fab': 'Fab', 'Program': 'Program', 'requested_wspw': 'Requested WSPW',
                'expected_wspw': 'Expected WSPW', 'realization_rate': 'Realization factor'}
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
def cached_response(fingerprint, case, basis, ceiling_, max_share):
    return K.response_plan(load_data(fingerprint), case, basis, ceiling_, max_share)


def response_view():
    case = pilot_case
    a, b, _ = st.columns([1.1, 1.1, 2])
    basis = a.selectbox('Size to', ['requested', 'expected'], key='resp_basis',
                        format_func={'requested': 'Requested starts', 'expected': 'Expected starts'}.get)
    max_share = b.selectbox('Most of a program that may move fabs', [25, 50, 100], index=1, key='resp_share', format_func=lambda v: f'{v}%')
    r = cached_response(FP, case, basis, ceiling, max_share)
    steps = r['steps']
    cols = [c for c in steps.columns if '|' in c and (BOTH or c.startswith(site + '|'))]
    steps = steps.assign(net_new=steps[cols].sum(axis=1),
                         cost=[sum(row[c] * C.RELATIVE_COST_INDEX[c.split('|')[1]] for c in cols) for _, row in steps.iterrows()])
    v, cost = steps.net_new.tolist(), steps.cost.tolist()
    over = r['overloads'] if BOTH else r['overloads'][r['overloads'].site == site]
    over_now = r['overloads_current'] if BOTH else r['overloads_current'][r['overloads_current'].site == site]
    first_q = over.quarter.min() if len(over) else None
    groups = over.drop_duplicates(['site', 'tool_group'])
    stats([
        (len(groups), 'Tool groups over 100% on installed + on order' + (f' · from {first_q}' if first_q else ''), 'bad' if len(groups) else ''),
        (v[1], 'Tools needed if nothing else gave way', ''),
        (f'{v[2] - v[1]:+d}', 'From capacity freed by declining programs', ''),
        (f'{v[3] - v[2]:+d}', 'From rebalancing work across fabs', ''),
        (f'{v[-1]} · {cost[-1]:,.0f}', 'Tools to buy · illustrative cost index', 'warn'),
    ])
    left, right = st.columns([1.25, 1], gap='large')
    with left:
        section('Net-new tools: 2026Q4 rate → 2027 plan → response', f'At the {ceiling*100:.0f}% planning limit')
        wf = steps[['step', 'net_new']].copy()
        chart(PC.waterfall(wf, final='Tools to buy'), 'response_waterfall')
    with right:
        section('Tool-hours added vs. freed · 2026Q4 rate → 2027 plan', '2027Q3 · hours / week')
        fv = K.freed_vs_added(data, case, basis, '2027Q3')
        fv = fv if BOTH else fv[fv.site == site]
        chart(PC.freed_added(fv, BOTH), 'freed_added')
    moves = r['transfer']['moves']
    section('Optimized rebalancing across fabs', f'{r["transfer"]["searched"]:,} combinations searched · up to {max_share}% of a program · lowest cost index')
    rows = ''.join(f'<tr><td class="l">{esc(D.NAMES[m["program"]])}</td><td class="l">{site_label(m["from_site"])} → {site_label(m["to_site"])}</td>'
                   f'<td class="num">{m["share_pct"]}%</td><td class="num">{m["avg_wspw"]:,}</td></tr>' for m in moves) or '<tr><td class="l" colspan="4">No move lowers the cost index</td></tr>'
    st.markdown('<div class="cap-wrap narrow"><table class="cap"><thead><tr><th class="l">Program</th><th class="l">Move</th><th>Share</th>'
                f'<th>Starts / week</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    section('Tools to add by quarter · after reallocation and rebalancing', 'Pull in = on-order tools needed before their arrival date')
    buy = r['buy'] if BOTH else r['buy'][r['buy'].site == site]
    capital_table(buy, False)
    st.caption('Cost index: illustrative relative cost per tool (wet clean = 1, EUV = 18), used only to rank responses; not a price.')
    export = buy[['site', 'tool_group', 'installed', 'on_order', 'needed_peak', 'net_new', 'pull_in', 'pull_in_by', *[f'add_{q}' for q in K.Q], 'cost_index']]
    st.download_button('Download response buy list (CSV)', csv(export), f'td_2027_response_{case}.csv', 'text/csv')


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
        (number(s['outs_wspw']), 'Completions / week', ''),
        (number(s['wip']), 'WIP · wafers', ''),
        (f'{s["cycle_days"]:.1f} d', f'Cycle time · p90 {s["cycle_p90"]:.1f} d', ''),
        (number(s['yield_pct'], 1, '%'), 'Line yield', ''),
    ])
    left, right = st.columns([1.2, 1], gap='large')
    with left:
        section('Wafer starts vs. completions vs. 2026Q4 request', 'Weekly · wafers')
        chart(PC.month_flow(s['weekly'], s['plan_wspw']), 'month_flow')
    with right:
        section('WIP and cycle time', 'Weekly')
        chart(PC.month_wip_ct(s['weekly']), 'month_wip_ct')
    left, right = st.columns(2, gap='large')
    with left:
        section(f'Tool loading vs. {ceiling*100:.0f}% planning limit', 'Q4 to date · label = queue per visit')
        chart(PC.loading_queue(tl, ceiling, BOTH), 'month_loading')
    with right:
        section('Tool availability: measured vs. planning assumption', 'Q4 to date · % of calendar time')
        chart(PC.availability(tl, BOTH), 'month_availability')
    left, right = st.columns([1, 1.15], gap='large')
    with left:
        section('Starts by program: actual vs. 2026Q4 request', 'Wafers / week')
        chart(PC.program_vs_plan(s['programs']), 'month_programs')
    with right:
        section('Capacity model vs. lot simulation', '2026Q3 · production loading by tool group')
        check = K.model_check(data)
        check = check if BOTH else check[check.site == site]
        chart(PC.model_check(check), 'model_check')
    st.caption('2026Q4 to date is a simulated continuation of the same fab model (seed 42): rerunning it reproduces the stored 52-week history exactly, '
               'then adds 28 Sep – 25 Oct 2026. Request = 2026Q4 requested starts.')


# ================================================================== CAPACITY SCENARIO PLANNING · SCENARIOS + AI ASSISTANT
TYPES = P.TYPES
SC_EXAMPLES = {
    'Node B +100 at Fab A, 2027Q1 (REQ-2027-014)': (
        'Add 100 wafers per week of DRAM Node B at Fab A in 2027Q1 (REQ-2027-014). Keep the HBM test vehicle qualification milestone unchanged.',
        dict(scenario_type='Demand change', site='FAB_A', program='DRAM_B', quarter='2027Q1', change=100., protected=['HBM_TV'])),
    'Node A +50 at Fab B, 2027Q1 (REQ-2027-021)': (
        'Add 50 wafers per week of DRAM Node A at Fab B in 2027Q1 (REQ-2027-021). Short loops may be deferred; HBM is protected.',
        dict(scenario_type='Demand change', site='FAB_B', program='DRAM_A', quarter='2027Q1', change=50., protected=['HBM_TV'])),
    'Shift 50 WSPW NAND → HBM at Fab A, 2027Q2': (
        'What if we move 50 wafers a week from 3D NAND to the HBM test vehicle at Fab A in 2027Q2, keeping total starts the same?',
        dict(scenario_type='Product mix shift', site='FAB_A', program='NAND_C', to_program='HBM_TV', quarter='2027Q2', wafers=50.)),
    'One EUV scanner down at Fab A, 2027Q3': (
        'One EUV scanner at Fab A is down for 2027Q3. What happens to loading?',
        dict(scenario_type='Tool unavailability', site='FAB_A', group='EUV', quarter='2027Q3', tools=1)),
    'Move 20% of Node A from Fab A to Fab B, 2027Q2': (
        'Move 20% of DRAM Node A from Fab A to Fab B in 2027Q2. Does Fab B become the constraint?',
        dict(scenario_type='Qualified load transfer', site='FAB_A', program='DRAM_A', quarter='2027Q2', pct=20.)),
}


def _sc_set(i):
    ss['sc_type'] = i.get('scenario_type') or 'Demand change'
    ss['sc_site'] = i.get('site') or 'FAB_A'
    ss['sc_quarter'] = i.get('quarter') or '2027Q1'
    if i.get('program') in C.PROGRAMS:
        ss['sc_program'] = i['program']
    if i.get('to_program') in C.PROGRAMS:
        ss['sc_to_program'] = i['to_program']
    if ss['sc_type'] == 'Demand change':
        cur = WA.current_wspw(data, ss['sc_site'], ss.get('sc_program', 'DRAM_B'), ss['sc_quarter'], basis='requested') \
            if ss.get('sc_program') in C.QUALIFIED.get(ss['sc_site'], []) else 0.
        if i.get('requested_wspw') is not None:
            ss['sc_change'] = float(i['requested_wspw']) - cur
        elif i.get('change') is not None:
            ss['sc_change'] = float(i['change'])
    ss['sc_protected'] = [p for p in i.get('protected', []) if p in C.PROGRAMS]
    if i.get('wafers') or i.get('wafers_shift'):
        ss['sc_wafers'] = float(i.get('wafers') or i.get('wafers_shift'))
    if i.get('group') or i.get('tool_group'):
        ss['sc_group'] = i.get('group') or i.get('tool_group')
    if i.get('tools') or i.get('tools_unavailable'):
        ss['sc_tools'] = int(i.get('tools') or i.get('tools_unavailable'))
    if i.get('pct') or i.get('transfer_pct'):
        ss['sc_pct'] = float(i.get('pct') or i.get('transfer_pct'))


def _sc_example():
    ss['sc_extraction'] = None
    ss['sc_error'] = None
    text, inputs = SC_EXAMPLES[ss['sc_example']]
    ss['sc_text'] = text
    _sc_set(inputs)


def _sc_extract():
    ss['sc_error'] = None
    try:
        key = secret('OPENAI_API_KEY')
        out = P.extract(ss['sc_text'], data, note_index(rag.documents_fingerprint(fd.DOCS), key, False), key,
                        secret('OPENAI_MODEL', 'gpt-4o-mini'), basis='requested')
        ss['sc_extraction'] = out
        _sc_set(out['inputs'])
    except Exception as exc:
        ss['sc_error'] = str(exc) if isinstance(exc, ValueError) else ai_errors.explain_error(exc)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_analysis(fingerprint, request_json, limit, ceiling_, fleet):
    return WA.analyze(load_data(fingerprint), json.loads(request_json), 'baseline', limit, ceiling_, 'plan', 'requested', fleet)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_scenario(fingerprint, args_json):
    args = json.loads(args_json)
    return request_model.run(load_data(fingerprint), **args)


def pill(status):
    cls = {'Within limit': 'ok', 'Needs exception': 'warn', 'Not accepted': 'bad', 'Not possible': 'na'}[status]
    return f'<span class="pill {cls}">{esc(status)}</span>'


def analyst_context(r):
    req = r['request']
    return dict(view='Scenario assistant · demand change', site=req['site'], quarter=req['quarter'], planning_limit=r['limit'],
                acceptance_ceiling=r['ceiling'], tools_assumed=request_model.FLEETS[r['fleet']],
                request=dict(program=req['program'], program_name=D.NAMES[req['program']], current_wspw=req['current_wspw'],
                             requested_wspw=req['requested_wspw'], change_wspw=req['delta_wspw'], protected=req['protected']),
                computed_decision=r['decision'].replace('<b>', '').replace('</b>', ''),
                fits_under_acceptance_ceiling_wspw=r['fits_under_ceiling'], fits_under_planning_limit_wspw=r['fits_under_limit'],
                bottleneck=dict(site=req['site'], tool_group=r['bottleneck'], name=rm.GROUP_LABELS[r['bottleneck']]),
                comparison={k: (None if v is None else {kk: round(vv, 1) for kk, vv in v.items()}) for k, v in r['comparison'].items()},
                options=[dict(rank=i + 1, option=o['option'], status=o['status'], started_now_wspw=o['started_now'],
                              taken_from_other_programs_wspw=o['taken_from_others'], peak_tool_group=o['peak_group'],
                              peak_loading_pct=None if pd.isna(o['peak_load']) else round(o['peak_load'], 1), detail=o['how'])
                         for i, o in enumerate(r['options'])],
                programs_sharing_bottleneck=r['affected'][['name', 'priority', 'milestone_date', 'protected', 'base_wspw', 'recommended_wspw']].round(1).to_dict('records') if len(r['affected']) else [],
                ranking_rules=WA.RANK_RULES,
                limits='Loading percentages do not predict queue time or milestone dates.')


def scenarios_view():
    key_ok = bool(secret('OPENAI_API_KEY'))
    ss.setdefault('sc_example', list(SC_EXAMPLES)[0])
    if 'sc_type' not in ss:
        _sc_example()

    # ---- assistant: describe the scenario
    left, right = st.columns([1.3, 1], gap='large')
    with left:
        section('Scenario assistant', 'Describe a scenario; AI fills the inputs below')
        st.selectbox('Example', list(SC_EXAMPLES), key='sc_example', on_change=_sc_example)
        st.text_area('Scenario', key='sc_text', height=90, label_visibility='collapsed')
        a, _ = st.columns([1.4, 2])
        a.button('Fill inputs with AI', type='primary', disabled=not key_ok, on_click=_sc_extract, key='sc_extract')
        if not key_ok:
            st.caption('AI needs OPENAI_API_KEY in the app secrets. Without it, examples fill the inputs.')
        if ss.get('sc_error'):
            st.error(ss['sc_error'])
    with right:
        section('What the AI understood', 'Validated against the plan')
        ext = ss.get('sc_extraction')
        if not ext:
            st.markdown('<p class="note">1 · The AI reads the text and fills the inputs below, citing the request or a note for each constraint.</p>'
                        '<p class="note">2 · Python checks every field against the plan: qualification, quarters, programs, tool groups.</p>'
                        '<p class="note">3 · The capacity model calculates; the AI then explains the computed result with citations.</p>',
                        unsafe_allow_html=True)
        else:
            i = ext['inputs']
            chips = [('Type', i['scenario_type']), ('Fab', rm.SITE_LABELS.get(i['site'], '—')), ('Program', D.NAMES.get(i['program'], '—')),
                     ('Quarter', i['quarter'] or '—')]
            st.markdown('<div class="chips">' + ''.join(f'<span><i>{k}</i>{esc(v)}</span>' for k, v in chips) + '</div>', unsafe_allow_html=True)
            if i['missing']:
                st.caption('Missing — fill by hand: ' + ', '.join(i['missing']))
            for c in i['constraints']:
                st.markdown(f'<p class="note">• {esc(c["text"])} <span class="src">[{esc(c["source_id"] or "no source")}]</span></p>', unsafe_allow_html=True)
            for w in ext['warnings']:
                st.caption('⚠ ' + w)

    # ---- inputs
    section('Inputs', 'Edit anything, then run')
    c1, c2, c3, c4 = st.columns([1.2, .8, .8, 1.5])
    stype = c1.selectbox('Scenario type', TYPES, key='sc_type')
    sc_site = c2.selectbox('Fab', list(C.SITES), format_func=rm.SITE_LABELS.get, key='sc_site')
    quarter = c3.selectbox('Quarter', C.PLAN_QUARTERS, key='sc_quarter')
    fleet = c4.radio('Tools assumed', list(request_model.FLEETS), format_func=request_model.FLEETS.get, key='sc_fleet',
                     help='Planned fleet = installed + on order + the 2027 buys from the 2027 capacity view.')
    other = 'FAB_B' if sc_site == 'FAB_A' else 'FAB_A'
    c1, c2, c3, c4 = st.columns([1.2, .8, .8, 1.5])
    progs = list(C.QUALIFIED[sc_site])
    if stype == 'Qualified load transfer':
        progs = [p_ for p_ in progs if p_ in C.QUALIFIED[other]]
    if stype != 'Tool unavailability' and ss.get('sc_program') not in progs:
        ss['sc_program'] = progs[0]
    args, request = None, None
    if stype == 'Demand change':
        program = c1.selectbox('Program', progs, format_func=D.NAMES.get, key='sc_program')
        current = WA.current_wspw(data, sc_site, program, quarter, basis='requested')
        ss['sc_change'] = max(float(ss.get('sc_change', 100.)), -current)
        change = c2.number_input(f'Change / week (now {current:,.0f})', min_value=-current, max_value=3000., step=10., format='%.0f', key='sc_change')
        ceiling_ = c3.selectbox('Acceptance ceiling', [.90, .95, 1.0], index=1, key='sc_ceiling', format_func=lambda v: f'{v*100:.0f}%',
                                help='Between the planning limit and this ceiling = needs an exception.')
        protected = c4.multiselect('Protected programs', [p_ for p_ in C.PROGRAM_ORDER if p_ != program], format_func=D.NAMES.get, key='sc_protected')
        w = WA.quarter_weeks(quarter)
        request = dict(site=sc_site, program=program, quarter=quarter, requested_wspw=float(current + change),
                       first_week=w[0], last_week=w[-1], protected=protected)
        signature = json.dumps(['demand', request, ceiling, ceiling_, fleet], sort_keys=True)
    else:
        kw = dict(operation=stype, site=sc_site, first=quarter, last=quarter, timing='quarters', ceiling=ceiling, fleet=fleet)
        if stype == 'Product mix shift':
            program = c1.selectbox('Shift from', progs, format_func=D.NAMES.get, key='sc_program')
            targets = [p_ for p_ in C.QUALIFIED[sc_site] if p_ != program]
            if ss.get('sc_to_program') not in targets:
                ss['sc_to_program'] = targets[0]
            to_program = c2.selectbox('Shift to', targets, format_func=D.NAMES.get, key='sc_to_program')
            ss.setdefault('sc_wafers', 50.)
            wafers = c3.number_input('Wafers / week', min_value=5., max_value=1000., step=5., format='%.0f', key='sc_wafers')
            kw.update(program=program, to_program=to_program, wafers=wafers)
        elif stype == 'Tool unavailability':
            groups = [g_ for g_ in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[sc_site][g_] > 0]
            if ss.get('sc_group') not in groups:
                ss['sc_group'] = groups[0]
            group = c1.selectbox('Tool group', groups, format_func=rm.GROUP_LABELS.get, key='sc_group')
            ss.setdefault('sc_tools', 1)
            tools = c2.number_input('Tools down', min_value=1, max_value=50, step=1, key='sc_tools')
            kw.update(group=group, unavailable=int(tools))
        else:
            program = c1.selectbox('Program', progs, format_func=D.NAMES.get, key='sc_program')
            ss.setdefault('sc_pct', 20.)
            pct = c2.number_input(f'% moved to {site_label(other)}', min_value=5., max_value=100., step=5., format='%.0f', key='sc_pct')
            kw.update(program=program, pct=float(pct), to_site=other)
        args = kw
        signature = json.dumps(['scenario', kw], sort_keys=True, default=str)

    if st.button('Run scenario', type='primary', key='sc_run'):
        ss['sc_signature'] = signature
    if ss.get('sc_signature') != signature:
        if ss.get('sc_signature'):
            st.info('Inputs changed. Run the scenario again to update the results.')
        return
    try:
        with st.spinner('Calculating loading for both fabs…'):
            if request:
                if ceiling_ < ceiling:
                    st.error('The acceptance ceiling must be at or above the planning limit.')
                    return
                demand_result(cached_analysis(FP, json.dumps(request), ceiling, ceiling_, fleet))
            else:
                scenario_result(cached_scenario(FP, json.dumps(args, default=str)), args)
    except ValueError as exc:
        st.error(str(exc))


def demand_result(r):
    req = r['request']
    grp = rm.GROUP_LABELS[r['bottleneck']]
    section(f'Result · {D.NAMES[req["program"]]} {req["delta_wspw"]:+,.0f} / week at {rm.SITE_LABELS[req["site"]]}, {req["quarter"]}',
            request_model.FLEETS[r['fleet']])
    st.markdown(f'<div class="decision">{r["decision"]}</div>', unsafe_allow_html=True)
    c, best = r['comparison'], r['recommended']
    rec = req['current_wspw'] + (best['started_now'] if best else 0)
    stats([
        (f'+{r["fits_under_ceiling"]:,.0f}' if req['delta_wspw'] > 0 else '—', f'Fits under the {r["ceiling"]*100:.0f}% ceiling', ''),
        (f'+{r["fits_under_limit"]:,.0f}' if req['delta_wspw'] > 0 else '—', f'Fits within the {r["limit"]*100:.0f}% limit', ''),
        (grp, 'Binding tool group', 'bad' if best is None else ''),
        (f'{c["baseline"]["loading_pct"]:.0f}% → {c["requested"]["loading_pct"]:.0f}%', f'{grp} load · full request', 'bad' if c['requested']['loading_pct'] > r['ceiling'] * 100 else 'warn'),
    ])
    section('Responses tested · ranked', 'Each recalculated through the weekly model · protected programs never touched')
    rows = []
    for i, o in enumerate(r['options']):
        started = '—' if o['started_now'] is None else f"+{o['started_now']:,.0f}"
        taken = '—' if not o['taken_from_others'] else f"{o['taken_from_others']:,.0f}"
        worst = '—' if pd.isna(o['peak_load']) else f"{o['peak_load']:.0f}% {rm.GROUP_LABELS[o['peak_group']]}"
        rows.append(f'<tr><td class="num">{i+1}</td><td class="l"><b>{esc(o["option"])}</b><div class="sub">{esc(o["how"])}</div></td>'
                    f'<td class="l">{pill(o["status"])}</td><td class="num">{started}</td><td class="num">{taken}</td><td class="num">{esc(worst)}</td></tr>')
    st.markdown('<div class="cap-wrap"><table class="cap opts"><thead><tr><th>#</th><th class="l">Response</th><th class="l">Status</th>'
                '<th>Started now</th><th>Taken from others</th><th>Worst load</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>',
                unsafe_allow_html=True)
    if len(r['affected']):
        others = r['affected'][~r['affected'].requester]
        if len(others):
            st.caption(f'Also uses {grp}: ' + ' · '.join(f'{a.name} (priority {a.priority}{", protected " + a.milestone_date if a.protected else ""})'
                                                          for a in others.itertuples()))
    ai_panel('ai_sc_demand', analyst_context(r), [
        'Explain this recommendation for leadership: how many additional wafers can be accepted, which tool group binds, '
        'and which programs share it. Do not claim any milestone delay.',
        'Why was each rejected response rejected?',
        'What would it take to accept the full request?'], title='AI explanation (grounded in this result)', expanded=True)


def scenario_result(review, args):
    scope = 'Both' if args['operation'] == 'Qualified load transfer' else args['site']
    q = args['first']
    brief = rm.decision_brief(review, q, scope)
    before, after = brief['before'], brief['after']
    current = rm.select_scope(review['comparison'], scope, q)
    name = {'Product mix shift': f"{args.get('wafers', 0):,.0f} WSPW {D.NAMES.get(args.get('program'), '')} → {D.NAMES.get(args.get('to_program'), '')}",
            'Tool unavailability': f"{args.get('unavailable', 0)} × {rm.GROUP_LABELS.get(args.get('group'), '')} down",
            'Qualified load transfer': f"{args.get('pct', 0):.0f}% of {D.NAMES.get(args.get('program'), '')} → {site_label(args.get('to_site', 'FAB_B'))}"}[args['operation']]
    section(f'Result · {name} · {rm.SITE_LABELS[args["site"]]}, {q}', request_model.FLEETS[args['fleet']])
    worst = after['worst_load']
    stats([
        (f"{before['starts']:,.0f} → {after['starts']:,.0f}", 'Starts / week', ''),
        (f"{before['at_risk']} → {after['at_risk']}", f'Tool groups ≥ {ceiling*100:.0f}%', 'warn' if after['at_risk'] > before['at_risk'] else ''),
        ('no capacity' if pd.isna(worst) else number(worst, 1, '%'), 'Peak load · ' + group_label(after['worst_site'], after['worst_group'], True),
         'bad' if pd.isna(worst) or worst > 100 else 'warn' if worst >= ceiling * 100 else ''),
        (str(after['over_capacity']), 'Tool groups over 100%', 'bad' if after['over_capacity'] else ''),
    ])
    moved = current[(current.change_pp.abs() > .05) | current.loading_pct_scenario.isna()].copy()
    moved = moved.assign(_k=moved.change_pp.abs().fillna(999)).sort_values('_k', ascending=False).head(10)
    section('Tool groups that change · baseline vs. scenario load', f'{q} · largest changes first')
    if moved.empty:
        st.caption('No tool group changes by more than 0.05 points.')
    else:
        chart(PC.scenario_delta(moved, ceiling, scope == 'Both'), 'sc_delta')
    ai_panel('ai_sc_supply', dict(view='Scenario assistant · ' + args['operation'], site=scope, quarter=q, planning_limit=ceiling,
                                  tools_assumed=request_model.FLEETS[args['fleet']], scenario=name, before=before, after=after,
                                  changed_tool_groups=moved.drop(columns='_k').round(2).to_dict('records'), applied=review['applied']), [
        'Explain what this scenario does to the fab and which tool groups need attention.',
        'Which response would you test next, and why?',
        'Which dated operating notes relate to the tool groups that change most?'], title='AI explanation (grounded in this result)', expanded=True)


def capacity_page():
    frame = cached_buy(FP, pilot_case, 'requested', ceiling)
    t = K.totals(K.scope(frame, site))
    exp = K.totals(K.scope(cached_buy(FP, pilot_case, 'expected', ceiling), site))
    page_head(f'{ui.SITES[site].upper()} · 2027 PLAN · {ceiling*100:.0f}% LIMIT', 'Capacity scenario planning')
    view = st.radio('View', ['2027 capacity', 'Response plan', 'Scenarios'], horizontal=True, key='cap_view', label_visibility='collapsed')
    if view == '2027 capacity':
        outlook_view()
    elif view == 'Response plan':
        response_view()
    else:
        scenarios_view()


PAGES = {'Demand outlook': demand_page, 'Fab performance': fab_performance_page, 'Capacity scenario planning': capacity_page}
PAGES[page]()
st.markdown('<div class="footer"><span>All fab data, routes, tool counts and plans are synthetic. No company data is used.</span>'
            '<span>Method notes: MODEL_GUIDE.md · VIEW_GUIDE.md</span></div>', unsafe_allow_html=True)
