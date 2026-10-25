"""User-facing regressions for shared filters, lazy views and scoped exports."""
import base64
import json
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest
import demand_outlook as demand
from ui_helpers import select_view

APP = Path(__file__).resolve().parents[1] / 'code' / 'apprag.py'


def app():
    return AppTest.from_file(str(APP), default_timeout=45).run()


def array(value):
    if isinstance(value, dict) and 'bdata' in value:
        return np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
    return np.asarray(value)


def test_main_filters_are_on_page_and_hidden_tabs_do_not_build():
    at = app()
    assert not at.exception
    assert not at.sidebar.selectbox and not at.sidebar.multiselect
    assert at.selectbox(key='site').value == 'Both'
    assert at.multiselect(key='programs').value
    assert 'route_program' not in [v.key for v in at.selectbox]
    assert len(at.tabs[1].children) == 0
    select_view(at, 0, 'Equipment health')
    assert not at.exception
    assert 'family' not in [v.key for v in at.selectbox]
    assert 'priority' not in [v.key for v in at.selectbox]
    assert len(at.tabs[0].children) == 0
    assert any('Shared equipment activity' in c.value for c in at.caption)


def test_program_subset_and_reporting_window_survive_navigation():
    at = app()
    at.selectbox(key='family').set_value('DRAM').run()
    at.multiselect(key='programs').set_value(['DRAM_A']).run()
    at.select_slider(key='report_weeks').set_range('2026-07-06', '2026-08-03').run()
    at.selectbox(key='priority').set_value('HOT').run()
    at.radio(key='page').set_value('Demand outlook').run()
    assert at.multiselect(key='programs').value == ['DRAM_A']
    at.selectbox(key='pilot_case').set_value('introduction').run()
    assert at.multiselect(key='programs').value == ['DRAM_A']
    at.radio(key='page').set_value('Fab performance').run()
    assert at.multiselect(key='programs').value == ['DRAM_A']
    assert at.select_slider(key='report_weeks').value == ('2026-07-06', '2026-08-03')
    assert at.selectbox(key='priority').value == 'HOT'


def test_demand_shared_filter_updates_chart_values_and_period(data):
    at = app()
    at.radio(key='page').set_value('Demand outlook').run()
    at.multiselect(key='programs').set_value(['DRAM_B']).run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 2
    main = json.loads(at.get('plotly_chart')[0].proto.spec)
    assert not main['layout'].get('updatemenus')
    expected = demand.comparison_totals(demand.comparison_frame(data, programs=['DRAM_B']))
    np.testing.assert_allclose(array(main['data'][0]['y']), expected.requested_wspw)
    at.select_slider(key='demand_period').set_range('2027Q3', '2027Q4').run()
    mix = json.loads(at.get('plotly_chart')[1].proto.spec)
    assert mix['data'][0]['x'] == ['2027Q3', '2027Q4']
    select_view(at, 1, 'Program & fab detail')
    rows = at.dataframe[0].value
    assert set(rows.Quarter) == {'2027Q3', '2027Q4'}
    assert set(rows.Program) == {'DRAM Node B (EUV)'}
    select_view(at, 1, 'Forecast validation')
    assert not at.exception
    assert 'demand_period' not in [v.key for v in at.select_slider]


def test_reset_restores_shared_filter_defaults():
    at = app()
    at.selectbox(key='site').set_value('FAB_A').run()
    at.selectbox(key='family').set_value('DRAM').run()
    at.selectbox(key='priority').set_value('HOT').run()
    next(b for b in at.button if b.label == 'Reset filters').click().run()
    assert not at.exception
    assert at.selectbox(key='site').value == 'Both'
    assert at.selectbox(key='family').value == 'All families'
    assert at.selectbox(key='priority').value == 'All priorities'
    assert len(at.multiselect(key='programs').value) == 5


def test_capacity_filters_do_not_erase_other_product_loading():
    at = app()
    at.radio(key='page').set_value('Capacity & scenarios').run()
    summary = [m.value for m in at.markdown if 'At-risk equipment' in m.value][0]
    at.multiselect(key='programs').set_value(['DRAM_B']).run()
    at.multiselect(key='capacity_groups').set_value(['EUV']).run()
    at.selectbox(key='capacity_detail').set_value('Program contribution').run()
    assert not at.exception
    assert [m.value for m in at.markdown if 'At-risk equipment' in m.value][0] == summary
    contribution = json.loads(at.get('plotly_chart')[1].proto.spec)
    assert 'Other programs' in [trace['name'] for trace in contribution['data']]
    at.multiselect(key='capacity_groups').set_value([]).run()
    assert not at.exception
    assert any('No equipment matches' in i.value for i in at.info)


def test_lot_search_and_limits_apply_to_displayed_rows():
    at = app()
    at.radio(key='page').set_value('Lot flow & cycle time').run()
    select_view(at, 3, 'Lot explorer')
    lot_id = at.dataframe[0].value.iloc[0]['Lot ID']
    at.text_input(key='lot_search').set_value(lot_id).run()
    assert at.dataframe[0].value['Lot ID'].tolist() == [lot_id]
    at.text_input(key='lot_search').set_value('').run()
    at.radio(key='lot_population').set_value('WIP at period end').run()
    at.number_input(key='lot_min_days').set_value(20.).run()
    assert not at.exception
    assert at.dataframe[0].value['Age at cutoff (d)'].ge(20).all()
    at.number_input(key='lot_max_days').set_value(5.).run()
    assert any('Maximum days must be at least' in i.value for i in at.info)


def test_quarter_change_uses_consecutive_quarter_and_missing_stays_missing(data):
    frame = demand.comparison_frame(data)
    changes = demand.comparison_changes(frame, '2027Q2')
    assert changes['qoq_quarter'] == '2027Q1'
    assert abs(changes['qoq_pct'] - 100 * (3000 / 2800 - 1)) < 1e-8
    restricted = frame[frame.quarter.eq('2027Q3')]
    figure = demand.history_outlook_chart(restricted, chart_filters=False)
    assert np.isnan(figure.data[0].y[0])
    assert np.isnan(figure.data[1].y[-1])
