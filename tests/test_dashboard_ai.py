from ui_helpers import select_view
from types import SimpleNamespace
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
import fab_data as fd
import fab_demo as D
import planning_copilot as rag
import dashboard_ai as DA


class Message:
    def __init__(self,content=None,calls=()):self.content=content;self.tool_calls=list(calls);self.refusal=None
    def model_dump(self,**kwargs):
        return {'role':'assistant','content':self.content,'tool_calls':[{'id':c.id,'type':'function','function':{'name':c.function.name,'arguments':c.function.arguments}} for c in self.tool_calls]}


def context():return dict(view='Pilot introduction DES',site='FAB_A',quarter='2027Q2',pilot_case='introduction',planning_limit=.85,period=['2027-04-05','2027-07-25'])


def test_new_ai_uses_view_context_and_model_tools_with_citations(data):
    call=SimpleNamespace(id='calculation',function=SimpleNamespace(name='read_pilot_study',arguments=json.dumps({'case':'introduction'})))
    messages=iter([Message(calls=[call]),Message('The pilot adds 1460 input-wafer equivalents of average WIP [S2].')])
    captured=[]
    def create(**kwargs):captured.append(kwargs);return SimpleNamespace(choices=[SimpleNamespace(message=next(messages))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent=DA.ViewCopilot(data,rag.DocumentIndex(fd.DOCS),context())
    out=agent.answer('What changes when the DRAM pilot is introduced?','mock',client=client)
    assert out['sources'][0]['data']['pilot_case']=='introduction'
    assert any(s['kind']=='document' for s in out['sources'])
    assert out['sources'][-1]['data']['matched_lots']==1613
    assert captured[0]['store'] is False
    assert 'read_pilot_study' in [t['function']['name'] for t in captured[0]['tools']]


def test_ai_reads_deployed_tables_and_current_pilot_loading(data):
    agent=DA.ViewCopilot(data,rag.DocumentIndex(fd.DOCS),context())
    rows=agent.execute('query_data',{'query':"SELECT COUNT(DISTINCT week_start) AS weeks FROM sim_fab_weekly"})
    assert rows['data']['rows'][0]['weeks']==52
    capacity=agent.execute('read_capacity',{'site':'FAB_A','quarter':'2027Q2','case':'introduction'})
    assert capacity['data']['case']=='introduction'
    assert capacity['data']['summary']['starts']>2050
    before=data['plan_2027'].copy(deep=True)
    source=agent.execute('calculate_capacity_request',{'operation':'Tool unavailability','site':'FAB_A','timing':'weeks','first':'2027-05-17','last':'2027-05-17','group':'EUV','unavailable':1})
    assert source['data']['basis'].endswith('introduction pilot overlay')
    assert source['data']['after']['over_capacity']>=1
    assert source['data']['after']['worst_load'] is None
    import pandas as pd
    pd.testing.assert_frame_equal(data['plan_2027'],before)
    with pytest.raises(ValueError,match='not qualified'):
        agent.execute('calculate_capacity_request',{'operation':'Qualified load transfer','site':'FAB_A','to_site':'FAB_B','program':'DRAM_B','pct':10,'timing':'quarters','first':'2027Q2','last':'2027Q2'})


def test_dated_unstructured_notes_and_archives_are_retrievable():
    index=rag.DocumentIndex(fd.DOCS)
    hits=index.search('SYN-OPS-003 2026-06-15 scanner recovery',4)
    assert any('SYN-OPS-003' in h['source'] for h in hits)
    pilot=index.search('DRAM_P pilot 17 May 2027 qualified FAB_A',4)
    assert any('pilot-introduction' in h['source'] for h in pilot)
    archived=[p for p in fd.DOCS.rglob('*.txt') if 'v6_records' not in p.parts]
    assert all(p.read_text().startswith('ARCHIVED PROTOTYPE NOTE') for p in archived)


def app():return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'code'/'apprag.py'),default_timeout=45).run()


def test_oee_removed_inventory_and_six_routes_visible():
    at=app();assert not at.exception
    assert [t.label for t in at.tabs]==['Production overview','Equipment health','Historical attainment','Program routes']
    assert 'Machine OEE' not in str(at)
    select_view(at,0,'Equipment health')
    assert any('Equipment states'==s.value for s in at.subheader)
    select_view(at,0,'Program routes')
    assert set(at.selectbox(key='route_program').options)==set(D.NAMES.values())
    at.selectbox(key='route_program').set_value(D.PILOT).run()
    assert not at.exception
    assert any('Ordered visits' in s.value for s in at.subheader)


def test_ai_evidence_preview_obeys_actual_performance_filters():
    at=app();at.selectbox(key='family').set_value('DRAM').run()
    at.multiselect(key='programs').set_value(['DRAM_B']).run()
    at.selectbox(key='priority').set_value('HOT').run()
    at.button(key='ai_performance_preview').click().run()
    assert not at.exception
    evidence=at.session_state['ai_performance_evidence']
    assert evidence['context']['programs']==['DRAM_B'] and evidence['context']['priority']=='HOT'
    assert len(evidence['context']['weekly'])==52
    assert at.button(key='ai_performance_ask').disabled


def test_live_ui_binding_uses_new_view_copilot_without_network(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','unit-test-only')
    seen={}
    def answer(self,question,key,model):
        seen.update(self.context)
        return dict(answer='Review the displayed weekly flow and operating records [S1].',sources=[dict(id='S1',kind='structured',title='Selected view',data=self.context)],trace=[dict(step='Read selected view',sources=['S1'])],retrieval_mode='keyword',retrieval_notice=None)
    monkeypatch.setattr(DA.ViewCopilot,'answer',answer)
    at=app();at.checkbox(key='ai_performance_semantic').set_value(False).run()
    at.button(key='ai_performance_ask').click().run()
    assert not at.exception
    assert seen['view']=='Historical fab performance' and len(seen['weekly'])==52
    assert at.session_state['ai_performance_answer']['result']['sources'][0]['id']=='S1'


def test_pilot_overlay_new_program_and_unknown_forecast_are_explicit():
    at=app();at.radio(key='page').set_value('Demand outlook').run()
    at.selectbox(key='pilot_case').set_value('introduction').run()
    assert not at.exception
    assert D.PILOT in at.multiselect(key='programs').value
    assert at.radio(key='demand_measure').disabled
    assert any('no realization history' in c.value for c in at.caption)
    at.radio(key='page').set_value('Capacity & scenarios').run()
    assert not at.exception
    assert any(t.label=='Program introduction' for t in at.tabs)
    select_view(at,2,'Program introduction')
    assert any('Pilot impact detail'==s.value for s in at.subheader)
    select_view(at,2,'Scenario comparison')
    at.selectbox(key='scenario_operation').set_value('Tool unavailability').run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert any('No capacity' in m.value for m in at.markdown)
