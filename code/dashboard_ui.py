"""Page navigation and the three planning assumptions every capital number depends on."""
from __future__ import annotations

import streamlit as st

PAGES = ['Demand outlook', 'Fab performance', 'Capacity planning']
SITES = {'Both': 'Both fabs', 'FAB_A': 'Fab A', 'FAB_B': 'Fab B'}
CASES = {'baseline': '2027 plan'}
LIMITS = [75, 80, 85, 90, 95]
# Which global assumptions each page actually uses. Unused ones are disabled, not hidden,
# so the bar never jumps around between pages.
USES = {
    'Demand outlook': {'site'},
    'Fab performance': {'site', 'limit'},
    'Capacity planning': {'site', 'case', 'limit', 'availability'},
}
AVAILABILITY = {'plan': 'Planning assumption', 'measured': 'Measured 2026Q4 to date'}
DEFAULTS = {'page': PAGES[0], 'site': 'Both', 'limit_pct': 85, 'availability': 'plan'}


def reset():
    for key, value in DEFAULTS.items():
        if key != 'page':
            st.session_state[key] = value


def render_header():
    ss = st.session_state
    for key, value in DEFAULTS.items():
        ss.setdefault(key, value)
    if ss['page'] not in PAGES:
        ss['page'] = PAGES[0]
    st.markdown('<div class="brand"><span class="brand-name">Fab Capacity Planner</span>'
                '<span class="brand-year">2027</span>'
                '<span class="brand-note">Synthetic two-fab development network</span></div>', unsafe_allow_html=True)
    page = st.radio('Page', PAGES, horizontal=True, key='page', label_visibility='collapsed')
    uses = USES[page]
    with st.container(key='assumptions'):
        a, c, e, d = st.columns([1.5, 1.1, 1.5, .9], vertical_alignment='bottom')
        site = a.radio('Fab', list(SITES), format_func=SITES.get, horizontal=True, key='site')
        limit = c.selectbox('Planning limit', LIMITS, format_func=lambda v: f'{v}% loading', key='limit_pct',
                            disabled='limit' not in uses,
                            help='Highest loading a tool group is planned to run at. Policy, not physics.')
        availability = e.selectbox('Tool availability', list(AVAILABILITY), format_func=AVAILABILITY.get, key='availability',
                                   disabled='availability' not in uses,
                                   help='Planning assumption = 1 − PM − unscheduled down − engineering (E10 targets). '
                                        'Measured = productive + standby share of calendar time, 2026Q4 to date. '
                                        'Furnaces also carry their calibrated batch efficiency in both.')
        d.button('Reset', on_click=reset, key='reset_assumptions')
    return dict(page=page, site=site, pilot_case='baseline', ceiling=limit / 100, availability=availability)
