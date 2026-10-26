from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'code' / 'apprag.py'
PAGES = ['Demand outlook', 'Fab performance', 'Capacity scenario planning']
VIEWS = ['2027 capacity', 'Response plan', 'Wafer request analyst', 'Other scenarios']


def app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_demand_outlook_opens_first_and_every_page_renders_for_every_fab():
    at = app()
    assert not at.exception
    assert at.title[0].value == 'Demand outlook'
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
    at.radio(key='demand_measure').set_value('Expected').run()
    assert not at.exception
    assert not any('class="headline"' in m.value for m in at.markdown)


def test_capacity_views_render_at_each_limit():
    at = app()
    at.radio(key='page').set_value('Capacity scenario planning').run()
    for limit in [85, 90]:
        at.selectbox(key='limit_pct').set_value(limit).run()
        for view in VIEWS:
            at.radio(key='cap_view').set_value(view).run()
            assert not at.exception, (limit, view)


def test_analyst_calculates_and_flags_stale_inputs():
    at = app()
    at.radio(key='page').set_value('Capacity scenario planning').run()
    assert at.radio(key='cap_view').value == '2027 capacity'
    at.radio(key='cap_view').set_value('Wafer request analyst').run()
    assert any(b.label == 'Extract inputs with AI' and b.disabled for b in at.button)
    assert at.radio(key='an_fleet').value == 'planned'
    at.button(key='an_calc').click().run()
    assert not at.exception
    planned = next(m.value for m in at.markdown if 'class="decision"' in m.value)
    assert 'Accept <b>+100</b>' in planned            # after the 2027 buys, EUV has room
    at.radio(key='an_fleet').set_value('current').run()
    at.button(key='an_calc').click().run()
    decision = next(m.value for m in at.markdown if 'class="decision"' in m.value)
    assert 'EUV lithography' in decision and 'capacity must be added' in decision
    at.number_input(key='an_target').set_value(900.).run()
    assert any('Inputs changed' in i.value for i in at.info)
    at.selectbox(key='analyst_example').set_value(at.selectbox(key='analyst_example').options[1]).run()
    assert at.selectbox(key='an_site').value == 'FAB_B' and at.number_input(key='an_target').value == 730.
    at.button(key='an_calc').click().run()
    assert any('Short-loop experiments' in m.value for m in at.markdown if 'class="decision"' in m.value)


def test_scenario_test_runs_and_hides_stale_results():
    at = app()
    at.radio(key='page').set_value('Capacity scenario planning').run()
    at.radio(key='cap_view').set_value('Other scenarios').run()
    next(b for b in at.button if b.label == 'Run scenario').click().run()
    assert not at.exception and at.session_state['review']['program'] == 'DRAM_B'
    at.number_input(key='scenario_pct').set_value(20).run()
    assert any('Inputs changed' in x.value for x in at.info)
