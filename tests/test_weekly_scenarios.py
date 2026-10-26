from ui_helpers import select_view
import pandas as pd
import pytest
import fab_capacity as fc
import weekly_scenarios as w
import fab_config as C

A=C.REQUESTED_WSPW_2027['FAB_A']
B,H=A['DRAM_B'],A['HBM_TV']
Q2_A=sum(v[1] for v in A.values())
from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_calendar_covers_year_and_clips_boundary_weeks():
    calendar=w.calendar()
    assert len(calendar)==53 and calendar.period_days.sum()==365
    assert calendar.iloc[0].period_start==pd.Timestamp('2027-01-01')
    assert calendar.iloc[-1].period_end==pd.Timestamp('2028-01-01')
    assert calendar.iloc[0].period_days==3 and calendar.iloc[-1].period_days==5
    assert (calendar.period_end.iloc[:-1].to_numpy()==calendar.period_start.iloc[1:].to_numpy()).all()


def test_full_week_matches_known_quarter_scenario(data):
    r=w.run(data,'FAB_A','DRAM_B',10,'2027-04-05','2027-04-05')
    assert w.summary(r,'FAB_A','2027-04-05')['starts']==Q2_A
    assert w.summary(r,'FAB_A','2027-04-05',True)['starts']==pytest.approx(Q2_A+.1*B[1])
    row=r['weekly_comparison'].query("site=='FAB_A' and tool_group=='EUV' and week_start=='2027-04-05'").iloc[0]
    assert row.loading_pct_base==round(100*(B[1]+H[1])*7/60/117.6,1)
    assert row.loading_pct_scenario==round(100*(1.1*B[1]+H[1])*7/60/117.6,1)
    assert row.available_h_period_base==pytest.approx(117.6)


def test_only_selected_week_and_fab_change(data):
    frame=w.run(data,'FAB_A','DRAM_B',10,'2027-04-05','2027-04-05')['weekly_comparison']
    unchanged=frame[(frame.site=='FAB_B')|(frame.week_start!='2027-04-05')]
    assert unchanged.change_pp.eq(0).all()
    assert len(frame)==19*53 and frame.inside_request_window.sum()==19


def test_equipment_release_is_prorated_at_quarter_boundary(data):
    frame,_=w.week_table(fc.base_state(data))
    row=frame.query("site=='FAB_A' and tool_group=='EUV' and week_start=='2027-09-27'").iloc[0]
    assert row.available_h_period==pytest.approx(1*24*.7*4+2*24*.7*3)
    assert row.required_h_period==pytest.approx(((B[2]+H[2])*7/60)*4/7+((B[3]+H[3])*7/60)*3/7)


def test_partial_week_uses_partial_hours(data):
    frame,_=w.week_table(fc.base_state(data))
    row=frame.query("site=='FAB_A' and tool_group=='EUV' and week_start=='2026-12-28'").iloc[0]
    assert row.required_h_period==pytest.approx((B[0]+H[0])*7/60*3/7)
    assert row.available_h_period==pytest.approx(117.6*3/7)
    assert row.loading_pct==round(100*(B[0]+H[0])*7/60/117.6,1)


def test_target_crosses_quarter_without_changing_other_weeks(data):
    r=w.run(data,'FAB_A','DRAM_B',0,'2027-06-28','2027-06-28',target_wspw=1000)
    row=r['weekly_scenario'].query("site=='FAB_A' and tool_group=='EUV' and week_start=='2027-06-28'").iloc[0]
    assert row.required_h_period==pytest.approx(((1000+H[1])*7/60)*3/7+((1000+H[2])*7/60)*4/7)
    assert r['weekly_comparison'].query("week_start!='2027-06-28'").change_pp.eq(0).all()


def test_availability_recovery_is_masked_to_request_window(data):
    r=w.run(data,'FAB_A','DRAM_B',0,'2027-04-05','2027-04-05',recovery_group='EUV',recovery_points=5)
    frame=r['weekly_comparison'].query("site=='FAB_A' and tool_group=='EUV'").set_index('week_start')
    assert frame.loc['2027-04-05','available_h_period_scenario']==pytest.approx(126)
    assert frame.loc['2027-04-12','available_h_period_scenario']==pytest.approx(117.6)


def test_reversed_and_outside_windows_are_rejected(data):
    for first,last in [('2027-10-04','2027-01-04'),('2028-01-03','2028-01-03')]:
        with pytest.raises(ValueError,match='end week'):w.run(data,'FAB_A','DRAM_B',10,first,last)


def test_weekly_hours_reconcile_to_whole_year(data):
    frame,_=w.week_table(fc.base_state(data));euv=frame.query("site=='FAB_A' and tool_group=='EUV'")
    expected=sum((dram+hbm)*7/60*days/7 for dram,hbm,days in zip(B,H,[90,91,92,92]))
    assert euv.required_h_period.sum()==pytest.approx(expected)
    assert euv.available_h_period.sum()==pytest.approx(273*24*.7+92*2*24*.7)


def test_weekly_run_preserves_source_plan(data):
    original=data['plan_2027'].copy(deep=True)
    w.run(data,'FAB_A','DRAM_B',10,'2027-04-05','2027-04-05')
    pd.testing.assert_frame_equal(data['plan_2027'],original)


def app():return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'code'/'apprag.py'),default_timeout=40).run()




