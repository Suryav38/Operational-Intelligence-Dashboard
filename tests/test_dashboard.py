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
    at.selectbox(key='site').set_value('FAB_A').run()
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

def test_visible_ai_explanation_and_stale_context(monkeypatch):
    import assistant as ai
    monkeypatch.setenv('OPENAI_API_KEY','unit-test-not-a-real-key')
    monkeypatch.setattr(ai,'explain_scenario',lambda context,key,model:{
        'summary':'EUV loading rises from 98.2% to 106.0%.',
        'operating_implication':'The request needs a capacity response.',
        'suggested_checks':['Confirm experiment timing with Engineering.'],
        'evidence_groups':['EUV','DEP']})
    at=AppTest.from_file(str(APP),default_timeout=25).run()
    at.radio(key='page').set_value('Scenario review').run()
    assert any('AI request reader' in s.value for s in at.subheader)
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    next(b for b in at.button if b.label=='Explain results with AI').click().run()
    assert not at.exception
    assert at.session_state['ai_explanation']['result']['evidence_groups']==['EUV','DEP']
    assert any('AI-generated interpretation' in c.value for c in at.caption)
    at.selectbox(key='inspect_quarter').set_value('2027Q3').run()
    assert not at.exception
    assert any('Generate a new AI explanation' in c.value for c in at.caption)

def test_combined_views_and_document_preview(monkeypatch):
    at=app(monkeypatch)
    assert at.selectbox(key='site').value=='Both'
    assert not at.exception
    at.selectbox(key='quarter').set_value('2027Q3').run()
    at.radio(key='page').set_value('Fab performance').run()
    assert not at.exception
    at.radio(key='page').set_value('AI copilot').run()
    assert not at.exception
    assert at.selectbox(key='quarter').value=='2027Q3'
    assert next(b for b in at.button if b.label=='Ask AI').disabled
    next(b for b in at.button if b.label=='Preview evidence').click().run()
    assert not at.exception
    assert at.session_state['copilot_preview']['docs']

def test_copilot_ui_uses_scope_and_hides_stale_answers(monkeypatch):
    import planning_copilot as rag
    monkeypatch.setenv('OPENAI_API_KEY','unit-test-not-a-real-key')
    seen={}
    def answer(self,question,api_key,model):
        seen['site']=self.site;seen['quarter']=self.quarter
        return {'answer':'The current equipment constraints need review [S1].',
                'sources':[{'id':'S1','kind':'structured','title':'Current dashboard','data':{'site':self.site}}],
                'trace':[{'step':'Read dashboard','sources':['S1']}],
                'retrieval_mode':'keyword','retrieval_notice':None}
    monkeypatch.setattr(rag.Copilot,'answer',answer)
    at=AppTest.from_file(str(APP),default_timeout=25).run()
    at.radio(key='page').set_value('AI copilot').run()
    at.checkbox(key='semantic_retrieval').set_value(False).run()
    next(b for b in at.button if b.label=='Ask AI').click().run()
    assert not at.exception
    assert seen=={'site':'Both','quarter':'2027Q2'}
    assert any('AI insight' in h.value for h in at.subheader)
    at.selectbox(key='quarter').set_value('2027Q3').run()
    assert not at.exception
    assert any('Context changed' in c.value for c in at.caption)
