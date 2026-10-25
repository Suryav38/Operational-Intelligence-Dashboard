"""Page navigation and the three planning assumptions every capital number depends on."""
from __future__ import annotations

import streamlit as st

PAGES = ['Capital plan', 'Demand', 'Levers', 'Scenarios', '2026 baseline']
SITES = {'Both': 'Both fabs', 'FAB_A': 'Fab A', 'FAB_B': 'Fab B'}
CASES = {'baseline': 'Current plan', 'introduction': 'Plan + EUV pilot', 'slower': 'Plan + pilot, slow ramp'}
LIMITS = [75, 80, 85, 90, 95]
# Which global assumptions each page actually uses. Unused ones are disabled, not hidden,
# so the bar never jumps around between pages.
USES = {
    'Capital plan': {'site', 'case', 'limit'},
    'Demand': {'site', 'case'},
    'Levers': {'site', 'case', 'limit'},
    'Scenarios': {'site', 'case', 'limit'},
    '2026 baseline': {'site'},
}
DEFAULTS = {'page': PAGES[0], 'site': 'Both', 'pilot_case': 'baseline', 'limit_pct': 85}


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
    st.markdown('<div class="brand"><span class="brand-name">TD Capacity Plan</span>'
                '<span class="brand-year">2027</span>'
                '<span class="brand-note">Synthetic two-fab development network</span></div>', unsafe_allow_html=True)
    page = st.radio('Page', PAGES, horizontal=True, key='page', label_visibility='collapsed')
    uses = USES[page]
    with st.container(key='assumptions'):
        a, b, c, d = st.columns([1.35, 1.25, 1, 1.4], vertical_alignment='bottom')
        site = a.radio('Fab', list(SITES), format_func=SITES.get, horizontal=True, key='site')
        case = b.selectbox('Plan case', list(CASES), format_func=CASES.get, key='pilot_case',
                           disabled='case' not in uses,
                           help='Adds the Fab A DRAM EUV pilot (from 17 May 2027) to the requested plan.')
        limit = c.selectbox('Planning limit', LIMITS, format_func=lambda v: f'{v}% loading', key='limit_pct',
                            disabled='limit' not in uses,
                            help='Highest loading a tool group is planned to run at. Policy, not physics.')
        d.button('Reset', on_click=reset, key='reset_assumptions')
    return dict(page=page, site=site, pilot_case=case if 'case' in uses else 'baseline', ceiling=limit / 100)
