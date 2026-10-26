from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import fab_config as C
import fab_demo as D
from fab_flow import FabFlow
import launch_model as L
import demand_outlook as demand
import scenario_requests as R


def test_dram_heavy_baseline_and_accelerating_dram_b(data):
    plan=data['plan_2027'];plan=plan.assign(technology=plan.program.map(D.TECHNOLOGY))
    totals=plan.groupby('quarter').requested_wspw.sum()
    dram=plan[plan.technology=='DRAM'].groupby('quarter').requested_wspw.sum()
    # The 75% DRAM mix is a run-rate assumption; requests sit higher because DRAM over-requests most.
    run=pd.DataFrame([dict(quarter=q,program=p,wspw=v[i]) for s in C.JULY_RUN_RATE_WSPW_2027 for p,v in C.JULY_RUN_RATE_WSPW_2027[s].items() for i,q in enumerate(C.PLAN_QUARTERS)])
    run_dram=run[run.program.map(D.TECHNOLOGY)=='DRAM'].groupby('quarter').wspw.sum()/run.groupby('quarter').wspw.sum()
    np.testing.assert_allclose(run_dram,.75)
    assert run.groupby('quarter').wspw.sum().tolist()==[2800,3000,3300,3600]
    assert totals.tolist()==[3130,3645,4030,4440]
    assert C.JULY_RUN_RATE_WSPW_2027['FAB_A']['DRAM_B']==[750,830,975,1125]
    assert C.RUN_RATE_WSPW_2027['FAB_A']['DRAM_B']==[825,1080,1270,1460]
    flow=data['weekly_program_flow'];starts=flow.groupby('program').wafer_starts.sum()
    share=sum(n for p,n in starts.items() if D.TECHNOLOGY[p]=='DRAM')/starts.sum()
    assert share==pytest.approx(.75,abs=.01)


def test_ordered_routes_match_declared_visits_and_qualifications(data):
    route=data['route_steps']
    for program,spec in D.PROGRAMS.items():
        steps=route[route.program==program]
        assert steps.step_no.tolist()==list(range(1,len(steps)+1))
        assert steps.groupby('tool_group').size().to_dict()=={g:n for g,n in spec['passes'].items() if n}
        litho=steps[steps.tool_group.isin(['EUV','DUV'])]
        assert len(litho)==spec['mask_layers']
        assert litho.layer.nunique()==spec['mask_layers']
    pilot=route[route.program==D.PILOT]
    assert set(pilot.qualified_fabs)=={'FAB_A'}
    assert pilot[pilot.tool_group=='EUV'].shape[0]==8


def test_weekly_wafer_balance_and_state_hours(data):
    assert data['sim_fab_weekly'].flow_balance_error.eq(0).all()
    assert data['launch_weekly'].flow_balance_error.eq(0).all()
    for key in ['sim_fab_weekly','launch_weekly']:
        f=data[key];np.testing.assert_allclose(f.opening_current_wafers+f.wafer_starts-f.wafer_outs-f.scrap_wafers,f.wip_current_wafers)
        assert f.wip_wafers.ge(f.wip_current_wafers).all()
    states=['productive_h','standby_h','engineering_h','sched_down_h','unsched_down_h']
    for key in ['sim_tool_group_weekly','launch_equipment']:
        f=data[key];np.testing.assert_allclose(f[states].sum(axis=1),f.calendar_h,atol=1e-6)
        assert (f.productive_h<=f.calendar_h+1e-6).all()


def test_quarter_history_reconciles_to_thirteen_reporting_weeks(data):
    flow=data['weekly_program_flow'].groupby(['site','program','week_start'],as_index=False).wafer_starts.sum()
    weeks=sorted(flow.week_start.unique())
    for block,quarter in enumerate(['2025Q4','2026Q1','2026Q2','2026Q3']):
        expected=flow[flow.week_start.isin(weeks[block*13:(block+1)*13])].groupby(['site','program']).wafer_starts.mean()
        actual=data['wafer_start_history'].query('quarter==@quarter').set_index(['site','program']).actual_wspw
        np.testing.assert_allclose(actual.sort_index(),expected.sort_index(),atol=1e-8)


def test_pre_release_prefix_and_cohort_are_identical(data):
    frame=data['launch_weekly'].query('week_start<@D.LAUNCH_DATE')
    cols=['wafer_starts','wafer_outs','wip_wafers','wip_current_wafers']
    for seed in D.SEEDS:
        baseline=frame.query("seed==@seed and case=='baseline'").sort_values('week_start')[cols].reset_index(drop=True)
        for case in ['introduction','slower']:
            other=frame.query('seed==@seed and case==@case').sort_values('week_start')[cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(baseline,other)
    initial=data['launch_initial_wip']
    assert initial.remaining_state_hours.ge(0).all()
    assert initial.remaining_route_steps.eq(initial.steps_total-initial.steps_done).all()
    assert set(initial.operation_state).issubset({'queued','processing','transport','engineering_hold'})


def test_paired_completion_changes_are_queue_changes_not_random_processing(data):
    cohort=set(data['launch_initial_wip'].query('seed==7').lot_id)
    base=data['launch_lots_baseline'].set_index('lot_id')
    after=data['launch_lots_introduction'].set_index('lot_id')
    ids=sorted(cohort&set(base.index)&set(after.index))
    a=base.loc[ids];b=after.loc[ids]
    assert a.end_time.notna().all() and b.end_time.notna().all()
    for column in ['process_days','transport_days','hold_days']:
        np.testing.assert_allclose(a[column],b[column],atol=2e-8)
    shift=(pd.to_datetime(b.end_time)-pd.to_datetime(a.end_time)).dt.total_seconds()/86400
    np.testing.assert_allclose(shift,b.queue_days-a.queue_days,atol=3e-8)


def test_true_batch_charges_one_machine_cycle_for_three_lots():
    sim=FabFlow('FAB_A','2027-04-05',weeks=1,seed=9,warmup_weeks=0)
    operation=next(op for op in sim.routes['SHORT'] if op.tool_group=='FURN')
    sim.routes={'SHORT':[operation]}
    sim.env.process(sim.batch_dispatch())
    for i in range(3):sim.env.process(sim.lot('SHORT','batch-test-'+str(i)))
    sim.env.run(until=20)
    assert len(sim.finished)==3
    assert len(sim.batches)==1 and sim.batches[0]['lots']==3
    assert sim.toolacc[(0,'FURN')]['productive']==pytest.approx(6)
    assert all(r['process_h']==6 for r in sim.finished)
    assert len({r['end_h'] for r in sim.finished})==1


def test_saved_batches_obey_compatibility_and_size(data):
    batches=data['launch_batches']
    assert batches.lots.between(1,3).all()
    assert batches.apply(lambda r:len(r.lot_ids.split('|'))==r.lots,axis=1).all()
    pilot=batches[batches.recipe_id.str.startswith('PILOT')]
    assert pilot.lot_ids.str.split('|').apply(lambda ids:all('-DRAM_P-' in i for i in ids)).all()


def test_pilot_summary_matches_underlying_replicates_and_pairs(data):
    summary=L.study_summary(data,'introduction');pairs=data['launch_lot_comparison'].query("case=='introduction'")
    assert summary['replications']==5
    assert summary['mean_delay_days']==pytest.approx(pairs.additional_days.mean())
    assert summary['matched_lots']==1613 and summary['unfinished_lots']==0
    assert summary['additional_wip']==pytest.approx(1460)
    assert L.study_summary(data,'slower')['additional_wip']<summary['additional_wip']
    baseline=L.study_summary(data,'baseline');assert baseline['mean_delay_days']==0


def test_pilot_loading_uses_its_route_but_has_no_fitted_demand_estimate(data):
    _,base=L.capacity(data,'baseline');_,case=L.capacity(data,'introduction')
    q='2027Q2';rate=float(data['pilot_plan_quarterly'].query("case=='introduction' and quarter==@q").requested_wspw.iloc[0])
    a=base.query("site=='FAB_A' and tool_group=='EUV' and quarter==@q").iloc[0]
    b=case.query("site=='FAB_A' and tool_group=='EUV' and quarter==@q").iloc[0]
    assert b.required_h_wk-a.required_h_wk==pytest.approx(rate*8/60,abs=.11)
    forecast=demand.plan(data,pilot_case='introduction')
    assert forecast.loc[(forecast.program==D.PILOT)&forecast.requested_wspw.gt(0),'expected_wspw'].isna().all()
    assert pd.isna(demand.quarterly(forecast).loc[q,'expected_wspw'])


def test_outage_and_transfer_are_bounded_and_use_current_pilot_overlay(data):
    outage=R.run(data,'Tool unavailability','FAB_A',group='EUV',unavailable=1,pilot_case='introduction')
    row=outage['comparison'].query("site=='FAB_A' and tool_group=='EUV' and quarter=='2027Q2'").iloc[0]
    assert pd.isna(row.loading_pct_scenario) and row.status=='No capacity'
    assert outage['base_state']['wspw']['FAB_A'][D.PILOT][1]>0
    transfer=R.run(data,'Qualified load transfer','FAB_A',program='DRAM_A',pct=10,to_site='FAB_B')
    for q in C.PLAN_QUARTERS:
        before=rm_summary(transfer,'baseline',q);after=rm_summary(transfer,'scenario',q)
        assert before==pytest.approx(after)
    with pytest.raises(ValueError,match='not qualified'):R.run(data,'Qualified load transfer','FAB_A',program='DRAM_B',to_site='FAB_B')
    with pytest.raises(ValueError,match='exceeds'):R.run(data,'Tool unavailability','FAB_A',group='EUV',unavailable=4)


def rm_summary(review,key,quarter):
    state=review['base_state' if key=='baseline' else 'scenario_state'];i=C.PLAN_QUARTERS.index(quarter)
    return sum(sum(values[i] for values in programs.values()) for programs in state['wspw'].values())
