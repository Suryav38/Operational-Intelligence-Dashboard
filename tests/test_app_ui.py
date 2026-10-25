from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'code' / 'apprag.py'
PAGES = ['Capital plan', 'Demand', 'Levers', 'Scenarios', '2026 baseline']


def app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_capital_plan_opens_first_and_every_page_renders_for_every_fab():
    at = app()
    assert not at.exception
    assert at.title[0].value == 'Capital plan'
    assert at.radio(key='page').options == PAGES
    for site in ['Both', 'FAB_A', 'FAB_B']:
        at.radio(key='site').set_value(site).run()
        for page in PAGES:
            at.radio(key='page').set_value(page).run()
            assert not at.exception, (site, page)
            assert at.title[0].value == page


def test_unused_assumptions_are_disabled_not_hidden():
    at = app()
    at.radio(key='page').set_value('2026 baseline').run()
    assert at.selectbox(key='pilot_case').disabled and at.selectbox(key='limit_pct').disabled
    at.radio(key='page').set_value('Demand').run()
    assert not at.selectbox(key='pilot_case').disabled and at.selectbox(key='limit_pct').disabled


def test_capital_headline_follows_basis_limit_and_case():
    at = app()
    def headline():
        return next(m.value for m in at.markdown if 'class="headline"' in m.value)
    first = headline()
    assert '55 more tools' in first
    at.radio(key='capital_basis').set_value('expected').run()
    assert 'expected 2027 plan' in headline()
    at.selectbox(key='limit_pct').set_value(90).run()
    at.selectbox(key='pilot_case').set_value('introduction').run()
    assert not at.exception and headline() != first
    at.button(key='reset_assumptions').click().run()
    assert at.selectbox(key='limit_pct').value == 85 and at.selectbox(key='pilot_case').value == 'baseline'


def test_scenario_runs_and_hides_stale_results():
    at = app()
    at.radio(key='page').set_value('Scenarios').run()
    next(b for b in at.button if b.label == 'Run scenario').click().run()
    assert not at.exception and at.session_state['review']['program'] == 'DRAM_B'
    at.number_input(key='scenario_pct').set_value(20).run()
    assert any('Inputs changed' in x.value for x in at.info)
    at.selectbox(key='scenario_first').set_value('2027Q4').run()
    at.selectbox(key='scenario_last').set_value('2027Q1').run()
    next(b for b in at.button if b.label == 'Run scenario').click().run()
    assert any('end quarter' in e.value for e in at.error)


def test_pilot_study_and_ai_panel_without_key():
    at = app()
    assert any(b.label == 'Ask AI' and b.disabled for b in at.button)
    at.radio(key='page').set_value('Scenarios').run()
    at.radio(key='what_if').set_value('EUV pilot study').run()
    assert not at.exception
    at.radio(key='site').set_value('FAB_B').run()
    assert any('qualified only at Fab A' in i.value for i in at.info)
