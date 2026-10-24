from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'code'/'apprag.py'

def app(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    return AppTest.from_file(str(APP),default_timeout=25).run()

def test_capacity_and_production_pages_render(monkeypatch):
    at=app(monkeypatch)
    assert not at.exception
    at.toggle(key='all_groups').set_value(True).run()
    assert not at.exception
    at.selectbox(key='site').set_value('FAB_B').run()
    assert not at.exception
    at.radio(key='page').set_value('Fab performance').run()
    assert not at.exception
    assert at.title[0].value=='Fab performance'
    at.selectbox(key='site').set_value('FAB_A').run()
    assert not at.exception

def test_scenario_recalculates_and_hides_stale_results(monkeypatch):
    at=app(monkeypatch)
    at.radio(key='page').set_value('Scenario review').run()
    assert not at.exception
    assert next(b for b in at.button if b.label=='Extract request').disabled
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert at.session_state['review']['pct']==10
    at.number_input(key='scenario_pct').set_value(20).run()
    assert any('Inputs changed' in x.value for x in at.info)
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert at.session_state['review']['pct']==20
    assert not at.exception
    at.checkbox(key='recovery_enabled').set_value(True).run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert len(at.session_state['review']['changes'])==2

def test_reversed_period_is_explained(monkeypatch):
    at=app(monkeypatch)
    at.radio(key='page').set_value('Scenario review').run()
    at.selectbox(key='scenario_first').set_value('2027Q4').run()
    at.selectbox(key='scenario_last').set_value('2027Q1').run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert any('end quarter' in e.value for e in at.error)

def test_reviewed_ai_fields_copy_across_sites(monkeypatch):
    at=app(monkeypatch)
    at.radio(key='page').set_value('Scenario review').run()
    note='Increase NAND Gen C wafer starts by 5% at Fab B during 2027Q3 only.'
    at.text_area(key='planning_note').set_value(note).run()
    at.session_state['ai_draft']=dict(program='NAND_C',site='FAB_B',change_pct=5,
        first_quarter='2027Q3',last_quarter='2027Q3',source_quote=note,open_questions=[])
    at.session_state['ai_note_source']=note
    at.run()
    next(b for b in at.button if b.label=='Copy reviewed request to controls').click().run()
    assert not at.exception
    assert at.selectbox(key='site').value=='FAB_B'
    assert at.selectbox(key='scenario_program').value=='NAND_C'
    assert at.number_input(key='scenario_pct').value==5
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert at.selectbox(key='inspect_quarter').value=='2027Q3'
