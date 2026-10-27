from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import fab_capacity as fc
import fab_data as fd
import planning_copilot as rag
import review_model as rm

def test_combined_does_not_pool_equipment_capacity(data):
    s=fc.base_state(data);t=fc.loading_table(s)
    a=rm.capacity_summary(t,s,'FAB_A','2027Q2');b=rm.capacity_summary(t,s,'FAB_B','2027Q2')
    both=rm.capacity_summary(t,s,'Both','2027Q2')
    assert both['starts']==a['starts']+b['starts']==3645
    assert both['groups']==19
    assert both['at_risk']==a['at_risk']+b['at_risk']
    assert both['worst_load']==max(a['worst_load'],b['worst_load'])
    assert len(rm.select_scope(t,'Both','2027Q2'))==19

def test_combined_production_uses_counts_and_cohorts(data):
    a=rm.production_metrics(data,'FAB_A');b=rm.production_metrics(data,'FAB_B');p=rm.production_metrics(data,'Both')
    assert len(p['weekly'])==52
    assert p['wip']==pytest.approx(a['wip']+b['wip'])
    assert p['avg_outs']==pytest.approx(a['avg_outs']+b['avg_outs'])
    all_full=pd.concat([a['full'],b['full']])
    assert p['cycle_days']==pytest.approx(all_full.cycle_time_days.mean())
    complete=pd.concat([a['completed'],b['completed']])
    assert p['line_yield']==pytest.approx(complete.wafers_out.sum()/complete.wafers_in.sum()*100)

def test_combined_ai_context_preserves_site_specific_facts(data):
    r=rm.run_review(data,'FAB_A','DRAM_B',10,'2027Q2','2027Q2')
    c=rm.ai_review_context(r,'2027Q2','Both')
    assert len(c['equipment'])==19
    assert c['facts']['fab_a_euv_after']=='170.3%'
    assert c['facts']['starts_after']=='3,772 wafers/week'
    snap=rag.dashboard_snapshot(data,'Both','2027Q2',review=r)
    assert len(snap['capacity'])==19
    assert snap['scope']['basis']=='reviewed hypothetical scenario'

def test_document_retrieval_and_provenance():
    index=rag.DocumentIndex(fd.DOCS)
    assert len({c['source'] for c in index.chunks})==len(list(fd.DOCS.rglob('*.txt')))
    hits=index.search('Fab A EUV scanner availability and second scanner',4)
    assert hits
    assert any('EUV' in x['text'] for x in hits)
    assert all(x['retrieval']=='keyword' for x in hits)
    assert all(x['text'] and x['character_end']>x['character_start'] for x in hits)

def test_hybrid_embeddings_are_used_with_keyword_retrieval():
    calls=[]
    def create(**kw):
        calls.append(kw)
        texts=kw['input'] if isinstance(kw['input'],list) else [kw['input']]
        return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=[1.,0.] if 'euv' in t.lower() else [0.,1.]) for i,t in enumerate(texts)])
    client=SimpleNamespace(embeddings=SimpleNamespace(create=create))
    idx=rag.DocumentIndex(fd.DOCS).enable_semantic(client)
    hits=idx.search('EUV scanner',3)
    assert len(calls)==2
    assert all(h['retrieval']=='hybrid semantic + keyword' for h in hits)

def test_sql_is_read_only_and_bounded():
    before=hashlib.sha256(fd.DB_PATH.read_bytes()).digest()
    out=rag.read_sql(fd.DB_PATH,"SELECT loading_pct FROM sim_tool_group_q3_2026 WHERE site='FAB_A' AND tool_group='EUV'")
    assert out['rows'][0]['loading_pct']==pytest.approx(86.12319554)
    assert rag.read_sql(fd.DB_PATH,'SELECT * FROM sim_lots',limit=3)['truncated']
    for sql in ['DELETE FROM sim_lots',"ATTACH DATABASE '/tmp/other.db' AS x","SELECT load_extension('anything')",'SELECT 1; DELETE FROM sim_lots']:
        with pytest.raises(ValueError):rag.read_sql(fd.DB_PATH,sql)
    assert hashlib.sha256(fd.DB_PATH.read_bytes()).digest()==before

class Message:
    def __init__(self,content=None,tool_calls=()):
        self.content=content;self.tool_calls=list(tool_calls);self.refusal=None
    def model_dump(self,**kw):
        return {'role':'assistant','content':self.content,'tool_calls':[
            {'id':c.id,'type':'function','function':{'name':c.function.name,'arguments':c.function.arguments}} for c in self.tool_calls]}

def test_copilot_runs_structured_tool_and_returns_citations(data):
    call=SimpleNamespace(id='call_test',function=SimpleNamespace(name='query_structured_data',arguments=json.dumps({'query':"SELECT loading_pct FROM sim_tool_group_q3_2026 WHERE site='FAB_A' AND tool_group='EUV'"})))
    messages=iter([Message(tool_calls=[call]),Message('Historical EUV loading is 86.12319554% [S2].')])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:SimpleNamespace(choices=[SimpleNamespace(message=next(messages))]))))
    agent=rag.Copilot(data,rag.DocumentIndex(fd.DOCS),'FAB_A','2027Q2')
    out=agent.answer('What was historical EUV loading?','mock',client=client)
    assert '[S2]' in out['answer']
    assert any(s['kind']=='document' for s in out['sources'])
    assert any(s.get('sql') for s in out['sources'])

def test_copilot_rejects_unknown_citations_and_unsupported_numbers():
    sources=[{'id':'S1','kind':'structured','data':{'load_pct':79.2}}]
    for answer in ['Load is 79.2% [S99].','Load is 99.9% [S1].','Load is 79.2%.']:
        with pytest.raises(ValueError):rag.validate_answer(answer,sources)

def test_hypothetical_tool_keeps_dashboard_and_source_plan_unchanged(data):
    before=data['plan_2027'].copy(deep=True)
    agent=rag.Copilot(data,rag.DocumentIndex(fd.DOCS),'Both','2027Q2')
    source=agent.execute('compare_hypothetical_scenario',{'changes':[{'type':'move_load','program':'NAND_C','site':'FAB_B','to_site':'FAB_A','pct':25,'from_quarter':'2027Q2'}]})
    assert source['data']['is_hypothetical']
    assert len(source['data']['selected_quarter_comparison'])==19
    pd.testing.assert_frame_equal(data['plan_2027'],before)

def test_site_comparison_includes_both_fabs_for_each_selected_group(data):
    import executive_charts as charts
    table=fc.loading_table(fc.base_state(data))
    figure=charts.equipment_compare(table,'Both','2027Q2',.85)
    fab_b=next(t for t in figure.data if t.name=='Fab B')
    index=list(fab_b.y).index('Planarization')
    assert pd.notna(fab_b.x[index])

def test_new_no_capacity_constraint_is_not_lost_in_scenario_merge(data):
    agent=rag.Copilot(data,rag.DocumentIndex(fd.DOCS),'Both','2027Q3')
    source=agent.execute('compare_hypothetical_scenario',{'changes':[{'type':'route_change','program':'NAND_C','tool_group':'EUV','passes_delta':1,'from_quarter':'2027Q3'}]})
    rows=source['data']['selected_quarter_comparison']
    new=next(r for r in rows if r['site']=='FAB_B' and r['tool_group']=='EUV')
    assert new['status']=='No capacity'
    assert new['loading_pct_scenario'] is None
    assert any(r['site']=='FAB_B' and r['tool_group']=='EUV' for r in source['data']['full_year_changed_cells'])
