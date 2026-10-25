from types import SimpleNamespace
import pandas as pd
import pytest
import assistant as ai
import fab_capacity as fc
import review_model as rm

def test_known_euv_capacity(data):
    state=fc.base_state(data)
    table=fc.loading_table(state)
    row=rm.select_scope(table,'FAB_A','2027Q1').query("tool_group == 'EUV'").iloc[0]
    assert row.required_h_wk == pytest.approx((750+150)*7/60,abs=.051)
    assert row.available_h_wk == pytest.approx(168*.70)
    assert row.loading_pct == pytest.approx(89.3,abs=.051)
    detail=rm.program_workload(state,'FAB_A','EUV','2027Q1')
    assert detail['Required h / week'].sum() == pytest.approx(105.0)

def test_focus_does_not_remove_other_programs_or_resources(data):
    state=fc.base_state(data)
    table=fc.loading_table(state)
    snapshot=table.copy(deep=True)
    focus=rm.select_scope(table,'FAB_A','2027Q2').query('tool_group in @rm.FOCUS_GROUPS')
    summary=rm.capacity_summary(table,state,'FAB_A','2027Q2')
    assert len(focus)==3
    assert summary['groups']==10
    assert summary['starts']==2050
    pd.testing.assert_frame_equal(table,snapshot)

def test_one_quarter_change_preserves_baseline_and_other_periods(data):
    review=rm.run_review(data,'FAB_A','DRAM_B',10,'2027Q2','2027Q2')
    base=review['base_state']['wspw']['FAB_A']['DRAM_B']
    scen=review['scenario_state']['wspw']['FAB_A']['DRAM_B']
    assert base==[750,830,975,1125]
    assert scen==pytest.approx([750,913,975,1125])
    unchanged=review['comparison'].query("site == 'FAB_B' or quarter != '2027Q2'")
    assert (unchanged.change_pp==0).all()
    assert len(review['comparison'])==len(review['baseline'])

@pytest.mark.parametrize('program,site,pct,first,last',[
    ('DRAM_B','FAB_B',10,'2027Q2','2027Q2'),
    ('DRAM_B','FAB_A',float('nan'),'2027Q2','2027Q2'),
    ('DRAM_B','FAB_A',-101,'2027Q2','2027Q2'),
    ('DRAM_B','FAB_A',10,'2027Q4','2027Q1')])
def test_invalid_scenarios_are_rejected(data,program,site,pct,first,last):
    with pytest.raises(ValueError): rm.run_review(data,site,program,pct,first,last)

def test_availability_uses_percentage_points(data):
    r=rm.run_review(data,'FAB_A','DRAM_B',0,'2027Q2','2027Q2',recovery_group='EUV',recovery_points=5)
    row=rm.select_scope(r['scenario'],'FAB_A','2027Q2').query("tool_group == 'EUV'").iloc[0]
    assert row.available_h_wk==pytest.approx(168*.75)
    assert r['scenario_state']['avail']['FAB_A']['EUV'][0]==pytest.approx(.70)

def test_risk_classification_includes_missing_capacity():
    assert rm.risk_label(float('nan'))=='No capacity'
    t=pd.DataFrame([dict(site='FAB_B',tool_group='EUV',quarter='2027Q1',loading_pct=float('nan'))])
    assert fc.constraints(t).iloc[0].status.startswith('Cannot run')

@pytest.mark.parametrize('site',['FAB_A','FAB_B'])
def test_production_metrics_use_matching_cohorts(data,site):
    p=rm.production_metrics(data,site)
    assert len(p['weekly'])==52
    assert str(p['window_start'])=='2025-09-29'
    assert str(p['window_end'])=='2026-09-27'
    assert p['line_yield']==pytest.approx(p['completed'].wafers_out.sum()/p['completed'].wafers_in.sum()*100)
    assert 'SHORT' not in set(p['full'].program)
    assert p['cycle_days']==pytest.approx(p['full'].cycle_time_days.mean())
    assert p['breakdown'].sum()==pytest.approx(p['cycle_days'],abs=.03)
    assert p['wip']==pytest.approx(p['weekly'].wip_wafers.mean())

def request(**overrides):
    fields=dict(program='DRAM_B',site='FAB_A',change_pct=10,first_quarter='2027Q2',last_quarter='2027Q2',
                source_quote='Increase DRAM Node B by 10%',open_questions=[])
    fields.update(overrides)
    return ai.StartRequest(**fields)

def test_ai_fields_are_validated_before_copying():
    assert ai.validate_request(request()).change_pct==10
    for r in [request(site='FAB_B'),request(change_pct=None),request(open_questions=['Which quarter?'])]:
        with pytest.raises(ValueError): ai.validate_request(r)

def test_ai_extract_uses_typed_output_and_source_evidence():
    captured={}
    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(refusal=None,parsed=request()))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    r=ai.extract_request('Increase DRAM Node B by 10% at Fab A during 2027Q2 only.','mock',client=client)
    assert r.change_pct==10
    assert captured['response_format'] is ai.StartRequest
    assert captured['store'] is False
    with pytest.raises(ValueError,match='verifiable quote'):
        ai.extract_request('A completely different note.','mock',client=client)

def test_decision_brief_accounts_for_existing_constraints(data):
    r=rm.run_review(data,'FAB_A','DRAM_B',10,'2027Q2','2027Q2')
    b=rm.decision_brief(r,'2027Q2')
    assert b['after']['starts']==pytest.approx(b['before']['starts']+83)
    assert b['after']['at_risk']>=b['before']['at_risk']
    assert 'due dates' in b['limit']

def test_ai_context_has_exact_model_facts_and_period(data):
    r=rm.run_review(data,'FAB_A','DRAM_B',10,'2027Q2','2027Q2')
    c=rm.ai_review_context(r,'2027Q2')
    assert c['facts']['euv_before']=='99.2%'
    assert c['facts']['euv_after']=='107.4%'
    assert c['facts']['dep_after']=='96.5%'
    assert c['facts']['starts_after']=='2,133 wafers/week'
    assert len(c['equipment'])==10
    assert c['quarter_is_affected'] is True
    assert rm.ai_review_context(r,'2027Q3')['quarter_is_affected'] is False

def test_ai_text_allows_only_source_fact_references():
    assert ai.render_grounded_text('Load changes to {{load}}.',{'load':'107.4%'})=='Load changes to 107.4%.'
    with pytest.raises(ValueError,match='unknown model fact'):
        ai.render_grounded_text('Load is {{invented}}.',{})
    with pytest.raises(ValueError,match='unverified quantity'):
        ai.render_grounded_text('This will save 200 hours.',{})
    with pytest.raises(ValueError,match='unverified quantity'):
        ai.render_grounded_text('Load is {load}.',{'load':'106%'})

def test_ai_scenario_explanation_is_grounded_and_typed(data):
    context=rm.ai_review_context(rm.run_review(data,'FAB_A','DRAM_B',10,'2027Q2','2027Q2'),'2027Q2')
    response=ai.ScenarioExplanation(
        summary='EUV loading moves from {{euv_before}} to {{euv_after}} in {{quarter}}.',
        operating_implication='Deposition also exceeds the {{planning_limit}} planning limit at {{dep_after}}.',
        suggested_checks=['Confirm experiment timing with Engineering.','Check recoverable availability with Operations.'],
        evidence_groups=['EUV','DEP'])
    captured={}
    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(refusal=None,parsed=response))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    out=ai.explain_scenario(context,'mock',client=client)
    assert '99.2% to 107.4%' in out['summary']
    assert '96.5%' in out['operating_implication']
    assert captured['response_format'] is ai.ScenarioExplanation
    assert captured['store'] is False
    context['equipment']=[x for x in context['equipment'] if x['code']!='EUV']
    with pytest.raises(ValueError,match='outside this review'):
        ai.explain_scenario(context,'mock',client=client)
