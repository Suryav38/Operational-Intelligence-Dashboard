from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'code'/'apprag.py'

def app():return AppTest.from_file(str(APP),default_timeout=40).run()


def test_performance_is_first_and_no_ai_controls():
    at=app()
    assert not at.exception
    assert at.title[0].value=='Fab performance'
    assert at.radio(key='page').options==['Fab performance','Capacity & scenarios','Lot flow & cycle time']
    assert not at.text_area
    assert not any('AI' in b.label for b in at.button)
    assert any('Weekly cycle time'==s.value for s in at.subheader)
    assert any('Equipment time allocation'==s.value for s in at.subheader)


def test_named_program_filters_and_empty_qualified_family():
    at=app()
    at.selectbox(key='family').set_value('DRAM').run()
    assert at.multiselect(key='programs').value==['DRAM_A','DRAM_B']
    at.multiselect(key='programs').set_value(['DRAM_B']).run()
    at.selectbox(key='priority').set_value('HOT').run()
    assert not at.exception
    at.selectbox(key='site').set_value('FAB_B').run()
    assert at.multiselect(key='programs').value==['DRAM_A']
    at.selectbox(key='family').set_value('HBM').run()
    assert not at.exception
    assert at.multiselect(key='programs').value==[]
    assert any('at least one program' in i.value for i in at.info)


def test_lot_drill_down_and_earlier_week_filters():
    at=app();at.radio(key='page').set_value('Lot flow & cycle time').run()
    assert not at.exception
    at.select_slider(key='report_weeks').set_range('2026-07-06','2026-08-03').run()
    at.radio(key='lot_population').set_value('WIP at period end').run()
    assert not at.exception
    frame=at.dataframe[-1].value
    assert frame['Completed'].isna().all()
    assert frame['Output wafers'].isna().all()
    assert frame['Route progress (%)'].isna().all()
    assert frame['Age at cutoff (d)'].gt(0).all()


def test_scenario_recalculates_and_hides_stale_results():
    at=app();at.radio(key='page').set_value('Capacity & scenarios').run()
    assert not at.exception
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert at.session_state['review']['program']=='DRAM_B'
    at.number_input(key='scenario_pct').set_value(20).run()
    assert any('Inputs changed' in x.value for x in at.info)
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert at.session_state['review']['pct']==20
    at.checkbox(key='recovery_enabled').set_value(True).run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert len(at.session_state['review']['changes'])==2


def test_target_and_reversed_period_validation():
    at=app();at.radio(key='page').set_value('Capacity & scenarios').run()
    at.radio(key='demand_mode').set_value('Target WSPW').run()
    at.number_input(key='target_FAB_A_DRAM_B_2027Q2').set_value(1000).run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert at.session_state['review']['scenario_state']['wspw']['FAB_A']['DRAM_B'][1]==1000
    at.selectbox(key='scenario_first').set_value('2027Q4').run()
    at.selectbox(key='scenario_last').set_value('2027Q1').run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert any('end quarter' in e.value for e in at.error)


def test_capacity_focus_retains_all_program_risk():
    at=app();at.radio(key='page').set_value('Capacity & scenarios').run()
    def risk():return next(m.value for m in at.markdown if 'At-risk equipment' in m.value)
    before=risk()
    at.multiselect(key='programs').set_value(['SHORT']).run()
    assert not at.exception
    assert risk()==before
    at.toggle(key='all_groups').set_value(True).run()
    assert not at.exception


def test_oee_calculator_validates_example_and_exposes_upload():
    at=app()
    at.number_input(key='oee_good').set_value(301).run()
    assert any('Good count' in e.value for e in at.error)
    at.radio(key='oee_source').set_value('Upload machine records').run()
    assert not at.exception
    assert not at.number_input
