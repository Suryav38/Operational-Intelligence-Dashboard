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
    archived=[p for p in fd.DOCS.rglob('*.txt') if not {'v6_records','requests','tool_logs'} & set(p.parts)]
    assert all(p.read_text().startswith('ARCHIVED PROTOTYPE NOTE') for p in archived)


def app():return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'code'/'apprag.py'),default_timeout=45).run()







def test_uncited_draft_gets_one_repair_turn(data):
    messages=iter([Message('EUV is the binding tool group at 89%.'),Message('EUV is the binding tool group at 89% [S1].')])
    def create(**kwargs):return SimpleNamespace(choices=[SimpleNamespace(message=next(messages))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent=DA.ViewCopilot(data,rag.DocumentIndex(fd.DOCS),dict(view='test',site='FAB_A',euv_loading_pct=89.3))
    out=agent.answer('Which tool group binds?','mock',client=client)
    assert out['answer'].endswith('[S1].')
    assert any('rejected' in t['step'] for t in out['trace'])


def test_ai_calculations_use_the_screen_fleet_and_availability():
    data=fd.load()
    args={'operation':'Tool unavailability','site':'FAB_A','timing':'quarters','first':'2027Q3','last':'2027Q3','group':'EUV','unavailable':1}
    peaks={}
    for fleet in ['planned','current']:
        agent=DA.ViewCopilot(data,rag.DocumentIndex(fd.DOCS),{'site':'FAB_A','quarter':'2027Q3','planning_limit':.85,'fleet':fleet,'availability':'plan'})
        out=agent.execute('calculate_capacity_request',dict(args,fleet='current' if fleet=='planned' else 'planned'))  # model cannot override
        assert out['data']['fleet']==fleet
        euv=[r for r in out['data']['selected_start_period'] if r['tool_group']=='EUV'][0]
        peaks[fleet]=euv['loading_pct_scenario']
    assert peaks['planned']<100 and (peaks['current'] is None or peaks['current']>100)
