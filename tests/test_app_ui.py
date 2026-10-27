from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'code' / 'apprag.py'
PAGES = ['Demand outlook', 'Fab performance', 'Capacity planning']
VIEWS = ['Current fleet', '2027 baseline', 'Response plan', 'Scenarios']


def app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_demand_outlook_opens_first_and_every_page_renders_for_every_fab():
    at = app()
    assert not at.exception
    assert at.title[0].value == 'Demand outlook for 2027'
    assert at.radio(key='page').options == PAGES
    for site in ['Both', 'FAB_A', 'FAB_B']:
        at.radio(key='site').set_value(site).run()
        for page in PAGES:
            at.radio(key='page').set_value(page).run()
            assert not at.exception, (site, page)


def test_demand_versions_and_unused_assumptions_disabled():
    at = app()
    assert at.selectbox(key='limit_pct').disabled
    at.radio(key='demand_split').set_value('Family').run()
    at.radio(key='demand_measure').set_value('Realization-adjusted').run()
    assert not at.exception
    assert not any('class="headline"' in m.value for m in at.markdown)


def test_capacity_views_render_at_each_limit():
    at = app()
    at.radio(key='page').set_value('Capacity planning').run()
    for limit in [85, 90]:
        at.selectbox(key='limit_pct').set_value(limit).run()
        for view in VIEWS:
            at.radio(key='cap_view').set_value(view).run()
            assert not at.exception, (limit, view)


def test_manual_scenario_runs_and_hides_stale_results():
    at = app()
    at.radio(key='page').set_value('Capacity planning').run()
    assert at.radio(key='cap_view').value == 'Current fleet'
    at.radio(key='cap_view').set_value('Scenarios').run()
    assert at.radio(key='scenario_fleet').value == 'current'
    next(b for b in at.button if b.label == 'Run scenario').click().run()
    assert not at.exception and at.session_state['review']['program'] == 'DRAM_B'
    assert any('class="decision"' in m.value for m in at.markdown)      # demand growth also ranks responses
    at.number_input(key='scenario_pct').set_value(20).run()
    assert any('Inputs changed' in x.value for x in at.info)


def test_every_scenario_type_runs():
    at = app()
    at.radio(key='page').set_value('Capacity planning').run()
    at.radio(key='cap_view').set_value('Scenarios').run()
    assert not any(b.label == 'Fill the form with AI' for b in at.button)
    for op in ['Product mix shift', 'Tool unavailability', 'Qualified load transfer']:
        at.selectbox(key='scenario_operation').set_value(op).run()
        next(b for b in at.button if b.label == 'Run scenario').click().run()
        assert not at.exception and at.session_state['review']['operation'] == op


def test_fab_performance_has_the_fab_assistant():
    at = app()
    at.radio(key='page').set_value('Fab performance').run()
    assert not at.exception
    assert any('Fab assistant' in e.label for e in at.expander)
    assert any('deposition tools at Fab A' in o for o in at.selectbox(key='ai_fab_example').options)
