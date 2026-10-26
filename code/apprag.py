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


# ================================================================== CAPACITY SCENARIO PLANNING · AI ASSISTANT (fills the scenario form)
OP_FROM_TYPE = {'Demand change': 'Demand growth', 'Product mix shift': 'Product mix shift',
                'Tool unavailability': 'Tool unavailability', 'Qualified load transfer': 'Qualified load transfer'}
SC_EXAMPLES = {
    'Node B +100 at Fab A, 2027Q1 (REQ-2027-014)': (
        'Add 100 wafers per week of DRAM Node B at Fab A in 2027Q1 (REQ-2027-014). Keep the HBM test vehicle qualification milestone unchanged.',
        dict(scenario_type='Demand change', site='FAB_A', program='DRAM_B', quarter='2027Q1', change_wspw=100.)),
    'Node A +50 at Fab B, 2027Q1 (REQ-2027-021)': (
        'Add 50 wafers per week of DRAM Node A at Fab B in 2027Q1 (REQ-2027-021). Short loops may be deferred; HBM is protected.',
        dict(scenario_type='Demand change', site='FAB_B', program='DRAM_A', quarter='2027Q1', change_wspw=50.)),
    'Shift 50 WSPW NAND → HBM at Fab A, 2027Q2': (
        'What if we move 50 wafers a week from 3D NAND to the HBM test vehicle at Fab A in 2027Q2, keeping total starts the same?',
        dict(scenario_type='Product mix shift', site='FAB_A', program='NAND_C', to_program='HBM_TV', quarter='2027Q2', wafers_shift=50.)),
    'One EUV scanner down at Fab A, 2027Q3': (
        'One EUV scanner at Fab A is down for 2027Q3. What happens to loading?',
        dict(scenario_type='Tool unavailability', site='FAB_A', tool_group='EUV', quarter='2027Q3', tools_unavailable=1)),
    'Move 20% of Node A from Fab A to Fab B, 2027Q2': (
        'Move 20% of DRAM Node A from Fab A to Fab B in 2027Q2. Does Fab B become the constraint?',
        dict(scenario_type='Qualified load transfer', site='FAB_A', program='DRAM_A', quarter='2027Q2', transfer_pct=20.)),
}


def _fill_form(i):
    """Write assistant inputs into the manual scenario form's own widget keys."""
    op = OP_FROM_TYPE.get(i.get('scenario_type'), 'Demand growth')
    s = i.get('site') if i.get('site') in C.SITES else 'FAB_A'
    q = i.get('quarter') if i.get('quarter') in C.PLAN_QUARTERS else '2027Q1'
    ss['scenario_operation'] = op
    ss['scenario_site'] = s
    if ss.get('site') not in ('Both', s):
        ss['site'] = s
    ss['scenario_timing'] = 'Quarter range'
    ss['scenario_first'] = ss['scenario_last'] = q
    prog = i.get('program') if i.get('program') in C.PROGRAMS else None
    if op == 'Product mix shift':
        if prog: ss['mix_from'] = prog
        if i.get('to_program') in C.PROGRAMS: ss['scenario_to_program'] = i['to_program']
        if i.get('wafers_shift'): ss['mix_wafers'] = float(i['wafers_shift'])
    elif op == 'Tool unavailability':
        if i.get('tool_group') in C.TOOL_GROUPS: ss['outage_group_' + s] = i['tool_group']
        if i.get('tools_unavailable'): ss['unavailable_tools'] = int(i['tools_unavailable'])
    elif op == 'Qualified load transfer':
        if prog: ss['scenario_program'] = prog
        if i.get('transfer_pct'): ss['transfer_pct'] = float(i['transfer_pct'])
    else:
        if prog: ss['scenario_program'] = prog
        p_ = prog or ss.get('scenario_program', 'DRAM_B')
        base = float(L.planning_state(data, pilot_case)['wspw'][s].get(p_, [0.] * 4)[C.PLAN_QUARTERS.index(q)])
        target = i.get('requested_wspw')
        if target is None and i.get('change_wspw') is not None:
            target = base + float(i['change_wspw'])
        if target is not None:
            ss['demand_mode'] = 'Target WSPW'
            ss[f'target_{s}_{p_}_{q}'] = max(0., float(target))
    ss['review'] = None


def _assist_example():
    ss['sc_extraction'], ss['sc_error'] = None, None
    text, inputs = SC_EXAMPLES[ss['sc_example']]
    ss['sc_text'] = text
    _fill_form(inputs)


def _assist_extract():
    ss['sc_error'] = None
    try:
        key = secret('OPENAI_API_KEY')
        out = P.extract(ss['sc_text'], data, note_index(rag.documents_fingerprint(fd.DOCS), key, False), key,
                        secret('OPENAI_MODEL', 'gpt-4o-mini'), case=pilot_case, basis='requested')
        ss['sc_extraction'] = out
        _fill_form(out['inputs'])
    except Exception as exc:
        ss['sc_error'] = str(exc) if isinstance(exc, ValueError) else ai_errors.explain_error(exc)


def scenario_assistant():
    key_ok = bool(secret('OPENAI_API_KEY'))
    ss.setdefault('sc_example', list(SC_EXAMPLES)[0])
    ss.setdefault('sc_text', SC_EXAMPLES[ss['sc_example']][0])
    with st.expander('AI assistant · describe a scenario and it fills the form below', expanded=ss.get('sc_open', False)):
        left, right = st.columns([1.3, 1], gap='large')
        with left:
            st.selectbox('Example', list(SC_EXAMPLES), key='sc_example', on_change=_assist_example)
            st.text_area('Scenario', key='sc_text', height=80, label_visibility='collapsed')
            st.button('Fill the form with AI', type='primary', disabled=not key_ok, on_click=_assist_extract, key='sc_extract')
            if not key_ok:
                st.caption('Needs OPENAI_API_KEY in the app secrets. Choosing an example fills the form without it.')
            if ss.get('sc_error'):
                st.error(ss['sc_error'])
        with right:
            ext = ss.get('sc_extraction')
            if not ext:
                st.markdown('<p class="note">The AI reads the text, picks the scenario type and fills the form, citing the request or a '
                            'retrieved note for each constraint. Python checks every field against the plan. You can still edit '
                            'everything before running.</p>', unsafe_allow_html=True)
            else:
                i = ext['inputs']
                chips = [('Type', i['scenario_type']), ('Fab', rm.SITE_LABELS.get(i['site'], '—')),
                         ('Program', D.NAMES.get(i['program'], '—')), ('Quarter', i['quarter'] or '—')]
                st.markdown('<div class="chips">' + ''.join(f'<span><i>{k}</i>{esc(v)}</span>' for k, v in chips) + '</div>', unsafe_allow_html=True)
                if i['missing']:
                    st.caption('Missing — fill by hand: ' + ', '.join(i['missing']))
                for c in i['constraints']:
                    st.markdown(f'<p class="note">• {esc(c["text"])} <span class="src">[{esc(c["source_id"] or "no source")}]</span></p>', unsafe_allow_html=True)
                for w in ext['warnings']:
                    st.caption('⚠ ' + w)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_analysis(fingerprint, request_json, limit, ceiling_, fleet):
    return WA.analyze(load_data(fingerprint), json.loads(request_json), 'baseline', limit, ceiling_, 'plan', 'requested', fleet)


def pill(status):
    cls = {'Within limit': 'ok', 'Needs exception': 'warn', 'Not accepted': 'bad', 'Not possible': 'na'}[status]
    return f'<span class="pill {cls}">{esc(status)}</span>'


def demand_responses(review, program, quarter, target, pct, fleet):
    """Demand growth only: rank the responses (accept, partial, cut lower priority, other fab) for the first quarter."""
    s = review['site']
    current = WA.current_wspw(data, s, program, quarter, basis='requested')
    requested = float(target) if target is not None else current * (1 + pct / 100)
    if requested <= current:
        return
    w = WA.quarter_weeks(quarter)
    protected = [x['program'] for x in PCX.PROGRAMS if x['protected'] and x['program'] != program]
    request = dict(site=s, program=program, quarter=quarter, requested_wspw=requested, first_week=w[0], last_week=w[-1], protected=protected)
    try:
        r = cached_analysis(FP, json.dumps(request), ceiling, .95, fleet)
    except ValueError as exc:
        st.caption(str(exc))
        return
    section(f'Responses tested · {quarter}', 'Ranked: feasible under 95% → protected untouched → most wafers now → least taken from others')
    st.markdown(f'<div class="decision">{r["decision"]}</div>', unsafe_allow_html=True)
    rows = []
    for k, o in enumerate(r['options']):
        started = '—' if o['started_now'] is None else f"+{o['started_now']:,.0f}"
        taken = '—' if not o['taken_from_others'] else f"{o['taken_from_others']:,.0f}"
        worst = '—' if pd.isna(o['peak_load']) else f"{o['peak_load']:.0f}% {rm.GROUP_LABELS[o['peak_group']]}"
        rows.append(f'<tr><td class="num">{k+1}</td><td class="l"><b>{esc(o["option"])}</b><div class="sub">{esc(o["how"])}</div></td>'
                    f'<td class="l">{pill(o["status"])}</td><td class="num">{started}</td><td class="num">{taken}</td><td class="num">{esc(worst)}</td></tr>')
    st.markdown('<div class="cap-wrap"><table class="cap opts"><thead><tr><th>#</th><th class="l">Response</th><th class="l">Status</th>'
                '<th>Started now</th><th>Taken from others</th><th>Worst load</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>',
                unsafe_allow_html=True)
    return r


def scenario_test():
    programs = em.program_options(site, 'All families', include_pilot=False)
    quarter = '2027Q3'
    a, b = st.columns([1, 1.4])
    operation = a.selectbox('Test', request_model.OPERATIONS, key='scenario_operation')
    fleet = b.radio('Tools assumed', list(request_model.FLEETS), format_func=request_model.FLEETS.get, horizontal=True, key='scenario_fleet',
                    help='Planned fleet = installed + on order + the 2027 buys from the 2027 capacity view (sized to requested starts at the planning limit).')
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
        request_priority = 'NORMAL'
    else:
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
    milestone = st.text_input('Urgent experiment / milestone (optional)', key='hot_milestone') if request_priority == 'HOT' and operation != 'Tool unavailability' else ''
    recovery = st.checkbox('Also test an availability recovery', key='recovery_enabled') if operation == 'Demand growth' else False
    recovery_group, points = None, 0.
    if recovery:
        a, b = st.columns(2)
        recovery_group = a.selectbox('Equipment to recover', [g_ for g_ in C.TOOL_GROUP_ORDER if C.INSTALLED_TOOLS[request_site][g_] > 0], format_func=rm.GROUP_LABELS.get, key='recovery_group_' + request_site)
        points = b.number_input('Recovery (percentage points)', min_value=.5, max_value=15., value=5., step=.5, key='recovery_points')
    signature = (site, pilot_case, operation, request_site, program, pct, target, timing, first_period, last_period, ceiling, recovery_group, points,
                 request_priority, milestone, outage_group, unavailable, fleet, to_program, wafers)
    if st.button('Run scenario', type='primary'):
        try:
            review = request_model.run(data, operation, request_site, program, pct, outage_group, unavailable, to_site, first_period, last_period,
                                       'weeks' if timing == 'Week range' else 'quarters', ceiling, recovery_group, points, target, pilot_case,
                                       fleet=fleet, to_program=to_program, wafers=wafers)
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
    request_name = (rm.GROUP_LABELS[outage_group] + ' unavailable' if operation == 'Tool unavailability'
                    else f'{wafers:,.0f} WSPW {ec.PROGRAM_SHORT[program]} → {ec.PROGRAM_SHORT[to_program]}' if operation == 'Product mix shift'
                    else ec.PROGRAM_SHORT[program] + ' · ' + request_priority)
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
            period_label + ' · ' + ('both fabs' if include_other else site_label(request_site) + ' only') + ' · ' + request_model.FLEETS[fleet].lower())
    stats([
        (f"{before['starts']:,.0f} → {after['starts']:,.0f}", 'Starts / week', ''),
        (f"{before['at_risk']} → {after['at_risk']}", f'Groups at or above {ceiling*100:.0f}%', 'warn' if after['at_risk'] > before['at_risk'] else ''),
        ('no capacity' if pd.isna(after['worst_load']) else number(after['worst_load'], 1, '%'),
         'Peak load · ' + group_label(after['worst_site'], after['worst_group'], True), 'bad' if pd.isna(after['worst_load']) or after['worst_load'] > 100 else 'warn'),
        (str(after['over_capacity']), 'Groups over 100%', 'bad' if after['over_capacity'] else ''),
    ])
    if operation == 'Demand growth':
        demand_responses(review, program, first_q, target, pct, fleet)
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
        'Which tool groups change most in this scenario, and what do the dated operating notes say about them?'])


def scenarios_view():
    scenario_assistant()
    scenario_test()


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
