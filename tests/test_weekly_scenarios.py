from ui_helpers import select_view
import pandas as pd
import pytest
import fab_capacity as fc
import weekly_scenarios as w
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
    assert w.summary(r,'FAB_A','2027-04-05')['starts']==2050
    assert w.summary(r,'FAB_A','2027-04-05',True)['starts']==2133
    row=r['weekly_comparison'].query("site=='FAB_A' and tool_group=='EUV' and week_start=='2027-04-05'").iloc[0]
    assert row.loading_pct_base==99.2 and row.loading_pct_scenario==107.4
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
    assert row.required_h_period==pytest.approx(((975+200)*7/60)*4/7+((1125+225)*7/60)*3/7)


def test_partial_week_uses_partial_hours(data):
    frame,_=w.week_table(fc.base_state(data))
    row=frame.query("site=='FAB_A' and tool_group=='EUV' and week_start=='2026-12-28'").iloc[0]
    assert row.required_h_period==pytest.approx(105.0*3/7)
    assert row.available_h_period==pytest.approx(117.6*3/7)
    assert row.loading_pct==89.3


def test_target_crosses_quarter_without_changing_other_weeks(data):
    r=w.run(data,'FAB_A','DRAM_B',0,'2027-06-28','2027-06-28',target_wspw=1000)
    row=r['weekly_scenario'].query("site=='FAB_A' and tool_group=='EUV' and week_start=='2027-06-28'").iloc[0]
    assert row.required_h_period==pytest.approx(((1000+170)*7/60)*3/7+((1000+200)*7/60)*4/7)
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
    expected=sum((dram+hbm)*7/60*days/7 for dram,hbm,days in [(750,150,90),(830,170,91),(975,200,92),(1125,225,92)])
    assert euv.required_h_period.sum()==pytest.approx(expected)
    assert euv.available_h_period.sum()==pytest.approx(273*24*.7+92*2*24*.7)


def test_weekly_run_preserves_source_plan(data):
    original=data['plan_2027'].copy(deep=True)
    w.run(data,'FAB_A','DRAM_B',10,'2027-04-05','2027-04-05')
    pd.testing.assert_frame_equal(data['plan_2027'],original)


def app():return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'code'/'apprag.py'),default_timeout=40).run()


def test_week_ui_names_fab_equipment_and_hot_without_capacity_change():
    at=app();at.radio(key='page').set_value('Capacity & scenarios').run();select_view(at,2,'Scenario comparison')
    at.radio(key='scenario_timing').set_value('Week range').run()
    at.selectbox(key='scenario_start_week').set_value('2027-04-05').run()
    at.selectbox(key='scenario_end_week').set_value('2027-04-05').run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    before=at.session_state['review']['weekly_scenario'].copy()
    at.selectbox(key='scenario_priority').set_value('HOT').run()
    assert any('Inputs changed' in i.value for i in at.info)
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert at.session_state['review']['request_priority']=='HOT'
    pd.testing.assert_frame_equal(before,at.session_state['review']['weekly_scenario'])
    assert any('Fab A' in s.value and 'HOT request' in s.value for s in at.subheader)
    risk=next(m.value for m in at.markdown if m.value.startswith('**At-risk equipment:**'))
    assert 'EUV' in risk and 'Deposition' in risk and 'Fab B' not in risk
    assert any('Priority changes dispatch order' in i.value for i in at.info)
    assert not at.checkbox(key='scenario_include_other').value


def test_quarter_results_default_to_affected_fab_with_other_fab_optional():
    at=app();at.radio(key='page').set_value('Capacity & scenarios').run();select_view(at,2,'Scenario comparison')
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert any('Results: Fab A only' in c.value for c in at.caption)
    at.checkbox(key='scenario_include_other').set_value(True).run()
    assert not at.exception
    assert any('Combined fab comparison' in c.value for c in at.caption)


def test_week_ui_validates_date_order():
    at=app();at.radio(key='page').set_value('Capacity & scenarios').run();select_view(at,2,'Scenario comparison')
    at.radio(key='scenario_timing').set_value('Week range').run()
    at.selectbox(key='scenario_start_week').set_value('2027-10-04').run()
    at.selectbox(key='scenario_end_week').set_value('2027-01-04').run()
    next(b for b in at.button if b.label=='Compare scenario').click().run()
    assert not at.exception
    assert any('end week' in e.value for e in at.error)
