from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'code' / 'apprag.py'
PAGES = ['Demand outlook', 'Fab performance', 'Capacity scenario planning']
VIEWS = ['2027 capacity', 'Response plan', 'Scenarios']


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


def test_scenario_assistant_demand_change_and_stale_inputs():
    at = app()
    at.radio(key='page').set_value('Capacity scenario planning').run()
    assert at.radio(key='cap_view').value == '2027 capacity'
    at.radio(key='cap_view').set_value('Scenarios').run()
    assert any(b.label == 'Fill inputs with AI' and b.disabled for b in at.button)
    assert at.radio(key='sc_fleet').value == 'planned' and at.selectbox(key='sc_type').value == 'Demand change'
    at.button(key='sc_run').click().run()
    assert not at.exception
    planned = next(m.value for m in at.markdown if 'class="decision"' in m.value)
    assert 'Accept <b>+100</b>' in planned            # after the 2027 buys, EUV has room
    at.radio(key='sc_fleet').set_value('current').run()
    at.button(key='sc_run').click().run()
    decision = next(m.value for m in at.markdown if 'class="decision"' in m.value)
    assert 'EUV lithography' in decision and 'capacity must be added' in decision
    at.number_input(key='sc_change').set_value(60.).run()
    assert any('Inputs changed' in i.value for i in at.info)


def test_scenario_assistant_runs_every_example_type():
    at = app()
    at.radio(key='page').set_value('Capacity scenario planning').run()
    at.radio(key='cap_view').set_value('Scenarios').run()
    seen = set()
    for option in at.selectbox(key='sc_example').options:
        at.selectbox(key='sc_example').set_value(option).run()
        at.button(key='sc_run').click().run()
        assert not at.exception and not at.error
        assert any('class="decision"' in m.value or 'class="stats"' in m.value for m in at.markdown)
        seen.add(at.selectbox(key='sc_type').value)
    assert seen == {'Demand change', 'Product mix shift', 'Tool unavailability', 'Qualified load transfer'}
