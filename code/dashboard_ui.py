"""Shared, on-page filters and persistent selection state."""
from __future__ import annotations

import html
import pandas as pd
import streamlit as st
import execution_model as em
import fab_config as C
import fab_demo as D
import review_model as rm


PAGE_VIEWS = {
    'Fab performance': ['Production overview', 'Equipment health', 'Historical attainment', 'Program routes'],
    'Demand outlook': ['Demand overview', 'Program & fab detail', 'Forecast validation'],
    'Capacity & scenarios': ['Capacity outlook', 'Scenario comparison', 'Program introduction'],
    'Lot flow & cycle time': ['Cycle-time drivers', 'Lot explorer'],
}
FILTER_KEYS = ('site', 'family', 'programs', 'pilot_case', 'quarter', 'ceiling_pct',
               'priority', 'report_weeks', 'include_short', 'demand_period', 'attainment_period',
               'capacity_groups', 'capacity_risk', 'lot_search', 'lot_min_days', 'lot_max_days')


def reconcile_programs(previous, previous_options, options, scope_changed=False):
    """Keep a deliberate subset; an all-program selection follows the scope."""
    if previous is None:
        return list(options)
    if not previous:
        return []
    if previous_options is not None and set(previous) == set(previous_options):
        return list(options)
    valid = [p for p in previous if p in options]
    return valid if valid or not scope_changed else list(options)


def reset_filters():
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.session_state['_filter_memory'] = {}
    st.session_state.pop('_program_options', None)
    st.session_state.pop('_program_scope', None)
    st.session_state.pop('_planning_quarter', None)
    st.session_state['review'] = None


def remember_filters():
    memory = st.session_state.setdefault('_filter_memory', {})
    for key in FILTER_KEYS:
        if key in st.session_state:
            memory[key] = st.session_state[key]


def render_filters(data, page, view):
    ss = st.session_state
    for key, value in ss.get('_filter_memory', {}).items():
        if key not in ss:
            ss[key] = value
    planning = page in ('Demand outlook', 'Capacity & scenarios')
    study = view == 'Program introduction'
    equipment = view == 'Equipment health'
    routes = view == 'Program routes'
    attainment = view == 'Historical attainment'
    validation = view == 'Forecast validation'
    program_scope = not (equipment or study)
    lot_scope = not planning and not (equipment or routes or attainment)
    weeks = sorted(data['sim_fab_weekly'].week_start.unique())
    first, last = weeks[0], weeks[-1]
    ss.setdefault('quarter', ss.get('_planning_quarter', '2027Q2'))
    ss.setdefault('pilot_case', 'baseline')
    ss.setdefault('family', 'All families')
    ss.setdefault('ceiling_pct', 85)
    ss.setdefault('priority', 'All priorities')
    ss.setdefault('include_short', False)
    with st.container(border=True, key='filter_bar'):
        cols = st.columns([1, 1.7, 1.1, 2] if program_scope else [1, 2, 2], gap='medium')
        site = cols[0].selectbox('Site', ['Both', 'FAB_A', 'FAB_B'], format_func=rm.SITE_LABELS.get, key='site')
        if validation:
            cols[1].caption('Held-out evaluation · 2026Q1–Q3\n\nModel fit · through 2025Q4')
            demand_period = ss.get('demand_period', ('2026Q1', '2027Q4'))
        elif page == 'Demand outlook':
            periods = [f'{year}Q{q}' for year in (2026, 2027) for q in range(1, 5)]
            demand_period = cols[1].select_slider('Chart period', periods, value=(periods[0], periods[-1]), key='demand_period')
        elif attainment:
            periods = sorted(data['wafer_start_history'].quarter.unique())
            cols[1].select_slider('Historical quarters', periods, value=(periods[0], periods[-1]), key='attainment_period')
            demand_period = ('2026Q1', '2027Q4')
        elif planning:
            cols[1].selectbox('Focus quarter', C.PLAN_QUARTERS, key='quarter', help='Highlights one quarter; horizon charts retain the full plan.')
            demand_period = ('2026Q1', '2027Q4')
        elif not routes:
            first, last = cols[1].select_slider('Reporting weeks', weeks, value=(weeks[0], weeks[-1]),
                format_func=lambda w: pd.Timestamp(w).strftime('%d %b %Y'), key='report_weeks')
            demand_period = ('2026Q1', '2027Q4')
        else:
            cols[1].caption('Route reference · follows the selected site and programs')
            demand_period = ('2026Q1', '2027Q4')

        if program_scope:
            family = cols[2].selectbox('Program family', ['All families', 'DRAM', 'NAND', 'HBM', 'Experiments'], key='family')
            options = em.program_options(site, family, include_pilot=routes or (planning and ss['pilot_case'] != 'baseline'))
            scope = (site, family, routes, planning, ss['pilot_case'] if planning else 'baseline')
            if ss.get('_program_scope') != scope:
                ss['programs'] = reconcile_programs(ss.get('programs'), ss.get('_program_options'), options, True)
            ss['_program_scope'], ss['_program_options'] = scope, options
            programs = cols[3].multiselect('Programs', options, format_func=D.NAMES.get, key='programs',
                placeholder='Select programs', help='Applies to page totals and charts. Capacity totals always retain all shared workload.')
            if not options:
                st.caption('This family has no qualified program at this fab.')
        else:
            family = 'All families'
            programs = em.program_options(site, family, include_pilot=planning and ss['pilot_case'] != 'baseline')
            if study:
                cols[2].selectbox('Pilot case', list(D.CASES), format_func=D.CASES.get, key='pilot_case')
            else:
                cols[2].caption('Shared equipment · all programs and priorities')

        actions = st.columns([1, 1, 5])
        if (planning and not validation) or lot_scope:
            with actions[0].popover('More filters', width='stretch'):
                if planning:
                    if not study:
                        st.selectbox('Pilot case', list(D.CASES), format_func=D.CASES.get, key='pilot_case')
                    if page == 'Demand outlook':
                        st.selectbox('Focus quarter', C.PLAN_QUARTERS, key='quarter', help='KPI cards use this quarter; charts follow Chart period.')
                    if page == 'Capacity & scenarios':
                        st.slider('Planning limit (%)', 75, 95, 85, 5, key='ceiling_pct')
                        st.caption('Planning policy; changing it does not add physical capacity.')
                else:
                    st.selectbox('Lot priority', ['All priorities', 'HOT', 'NORMAL'], key='priority')
                    st.checkbox('Include short loops in cycle time', key='include_short')
        actions[1].button('Reset filters', on_click=reset_filters, width='stretch')
        quarter = ss['quarter']
        pilot_case = ss['pilot_case'] if planning else 'baseline'
        priority = ss['priority'] if lot_scope else 'All priorities'
        include_short = ss['include_short'] if lot_scope else False
        if validation:
            period_label = '2026Q1–Q3 validation'
        elif page == 'Demand outlook':
            period_label = '–'.join(ss['demand_period']) + ' · KPI focus ' + quarter
        elif attainment:
            period_label = '–'.join(ss['attainment_period'])
        elif planning:
            period_label = quarter + ' focus · 2027 plan'
        elif routes:
            period_label = 'Route reference'
        else:
            period_label = pd.Timestamp(first).strftime('%d %b %Y') + '–' + (pd.Timestamp(last) + pd.Timedelta(days=6)).strftime('%d %b %Y')
        program_label = ('All qualified programs' if len(programs) == len(em.program_options(site, family, routes or (planning and pilot_case != 'baseline')))
                         else (f'{len(programs)} programs' if len(programs) != 1 else D.NAMES[programs[0]]))
        labels = [rm.SITE_LABELS[site], period_label]
        if program_scope:
            labels += [family, program_label]
        if lot_scope and priority != 'All priorities':
            labels.append(priority)
        if planning:
            labels.append(D.CASES[pilot_case])
        st.markdown('<div class="filter-summary">' + ''.join('<span>' + html.escape(str(label)) + '</span>' for label in labels) + '</div>', unsafe_allow_html=True)
    ss['_planning_quarter'] = quarter
    remember_filters()
    return dict(site=site, programs=programs, family=family, quarter=quarter, pilot_case=pilot_case,
                ceiling=ss['ceiling_pct'] / 100, priority=priority, first=first, last=last,
                include_short=include_short, demand_period=demand_period)
