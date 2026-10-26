import fab_config
import numpy as np
import pandas as pd
import pytest
import execution_model as em
import review_model as rm


def test_reconstructed_default_matches_published_weekly_totals(data):
    p=em.execution(data)
    cols=['wafer_starts','wafer_outs','wip_wafers']
    expected=data['sim_fab_weekly'].groupby('week_start')[cols].sum()
    actual=p['weekly'].set_index('week_start')[cols]
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
    assert p['cycle_days']==pytest.approx(rm.production_metrics(data,'Both')['cycle_days'])


def test_program_and_priority_partition_flow_and_wip(data):
    overall=em.execution(data)
    a=em.execution(data,priority='HOT');b=em.execution(data,priority='NORMAL')
    cols=['wafer_starts','wafer_outs','wip_wafers']
    np.testing.assert_allclose(overall['weekly'][cols],a['weekly'][cols]+b['weekly'][cols])
    total=sum((em.execution(data,programs=[p])['weekly'][cols] for p in em.program_options()),start=pd.DataFrame(0,index=range(52),columns=cols))
    np.testing.assert_allclose(overall['weekly'][cols],total)


def test_filtered_cohort_and_weighted_combined_cycle_time(data):
    p=em.execution(data,programs=['NAND_C'],priority='HOT',first='2026-07-13',last='2026-08-10')
    assert set(p['completed'].program)=={'NAND_C'}
    assert set(p['completed'].priority)=={'HOT'}
    assert p['completed'].ended.ge(pd.Timestamp('2026-07-13')).all()
    assert p['completed'].ended.lt(pd.Timestamp('2026-08-17')).all()
    assert p['cycles'].cycle_time_days.mean()==pytest.approx(p['cycle_days'])
    for _,row in p['weekly'].iterrows():
        lo=pd.Timestamp(row.week_start);hi=lo+pd.Timedelta(days=7)
        expected=p['cycles'].loc[p['cycles'].ended.ge(lo)&p['cycles'].ended.lt(hi),'cycle_time_days'].mean()
        if pd.isna(expected):assert pd.isna(row.avg_cycle_time_days)
        else:assert row.avg_cycle_time_days==pytest.approx(expected)


def test_historical_wip_does_not_backcast_route_progress(data):
    p=em.execution(data,last='2026-07-06')
    wip=p['wip_lots']
    assert len(wip)>0
    assert wip.progress_pct.isna().all()
    assert wip.started.lt(p['cutoff']).all()
    assert (wip.ended.isna()|wip.ended.ge(p['cutoff'])).all()
    expected=(p['cutoff']-wip.started).dt.total_seconds()/86400
    np.testing.assert_allclose(wip.age_days,expected)


def test_empty_and_short_loop_cohorts_are_explicit(data):
    empty=em.execution(data,programs=[])
    assert empty['weekly'].wafer_outs.sum()==0
    assert pd.isna(empty['cycle_days']) and pd.isna(empty['yield_pct'])
    short=em.execution(data,programs=['SHORT'])
    assert short['cycles'].empty
    assert not em.execution(data,programs=['SHORT'],include_short=True)['cycles'].empty


def test_attainment_uses_filtered_start_rates(data):
    out=em.historical_attainment(data,'FAB_A',['DRAM_B'])
    expected=data['wafer_start_history'].query("site=='FAB_A' and program=='DRAM_B'")
    assert out.actual_wspw.sum()==expected.actual_wspw.sum()
    np.testing.assert_allclose(out.attainment_pct,100*out.actual_wspw/out.requested_wspw)
    assert em.program_options('FAB_B','HBM')==[]


def test_queue_and_hold_claim_is_measured(data):
    p=em.execution(data);parts=p['breakdown']
    assert (parts.queue_days+parts.hold_days)/parts.sum()==pytest.approx(.37145,abs=.001)


def test_target_wspw_applies_exactly_to_each_selected_quarter(data):
    review=rm.run_review(data,'FAB_A','DRAM_B',0,'2027Q2','2027Q4',target_wspw=1000)
    B=fab_config.REQUESTED_WSPW_2027['FAB_A']['DRAM_B']
    assert review['scenario_state']['wspw']['FAB_A']['DRAM_B']==[B[0],1000,1000,1000]
    assert review['base_state']['wspw']['FAB_A']['DRAM_B']==B
    for value in [-1,float('nan')]:
        with pytest.raises(ValueError):rm.run_review(data,'FAB_A','DRAM_B',0,'2027Q2','2027Q2',target_wspw=value)

