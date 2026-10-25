import pandas as pd
import numpy as np
import pytest
import demand_outlook as demand
import fab_capacity as fc
import forecast_model as fm
from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_next_year_totals_reconcile_to_requested_plan(data):
    frame=demand.plan(data)
    totals=demand.quarterly(frame)
    assert totals.requested_wspw.tolist()==[2800,3000,3300,3600]
    assert totals.requested_wspw.mean()==3175
    assert len(frame)==32
    assert set(frame.quarter)=={'2027Q1','2027Q2','2027Q3','2027Q4'}
    expected=fm.corrected_plan(data['plan_2027'],fm.fit(data['wafer_start_history']))
    assert totals.expected_wspw.tolist()==expected.groupby('quarter').corrected_wspw.sum().tolist()


def test_combined_equals_sum_of_fabs_and_programs(data):
    total=demand.quarterly(demand.plan(data))
    by_fab=sum((demand.quarterly(demand.plan(data,s)) for s in ['FAB_A','FAB_B']))
    pd.testing.assert_frame_equal(total,by_fab)
    by_program=sum((demand.quarterly(demand.plan(data,programs=[p])) for p in fm.C.PROGRAM_ORDER))
    pd.testing.assert_frame_equal(total,by_program)
    filtered=demand.plan(data,'FAB_B',['NAND_C'])
    assert set(filtered.program)=={'NAND_C'} and set(filtered.site)=={'FAB_B'}
    assert filtered.requested_wspw.tolist()==[120,130,135,140]


def test_expected_starts_are_pooled_program_calibration_not_new_site_models(data):
    frame=demand.plan(data,programs=['DRAM_A'])
    assert frame.realization_rate.nunique()==1
    np.testing.assert_allclose(frame.expected_wspw,(frame.requested_wspw*frame.realization_rate).round())
    assert set(demand.plan(data,programs=['DRAM_B']).route_technology)=={'Includes EUV'}
    assert set(demand.plan(data,programs=['DRAM_A']).route_technology)=={'DUV only'}


def test_holdout_predictions_never_train_on_holdout_actuals(data):
    held,metrics=demand.validation(data)
    changed={**data,'wafer_start_history':data['wafer_start_history'].copy()}
    mask=changed['wafer_start_history'].quarter>fm.TRAIN_END
    changed['wafer_start_history'].loc[mask,'actual_wspw']*=2
    changed_held,changed_metrics=demand.validation(changed)
    np.testing.assert_allclose(held.expected_wspw,changed_held.expected_wspw)
    assert changed_metrics['expected_mape']!=metrics['expected_mape']
    expected=(100*(held.expected_wspw-held.actual_wspw).abs()/held.actual_wspw).mean()
    assert metrics['expected_mape']==pytest.approx(expected)
    assert all(held.quarter>fm.TRAIN_END)


def test_empty_programs_and_future_views_do_not_change_capacity(data):
    before=data['plan_2027'].copy(deep=True)
    state=fc.base_state(data)
    assert demand.plan(data,programs=[]).empty
    held,metrics=demand.validation(data,programs=[])
    assert held.empty and metrics['rows']==0
    demand.plan(data,'FAB_A',['DRAM_B'])
    pd.testing.assert_frame_equal(data['plan_2027'],before)
    assert fc.base_state(data)['wspw']==state['wspw']


def test_all_chart_breakdowns_preserve_selected_quarter_totals(data):
    frame=demand.plan(data,'Both',['DRAM_A','DRAM_B'])
    for group in demand.BREAKDOWNS:
        for measure,col in demand.MEASURES.items():
            fig=demand.mix_chart(frame,group,measure)
            plotted=np.array([trace.y for trace in fig.data],dtype=float).sum(axis=0)
            np.testing.assert_allclose(plotted,demand.quarterly(frame)[col])
    assert len(demand.ramp_chart(frame,'Requested plan','2027Q2').data[0].y)==3


def test_demand_page_filters_and_quarter_scope():
    app=Path(__file__).resolve().parents[1]/'code'/'apprag.py'
    at=AppTest.from_file(str(app),default_timeout=40).run()
    at.radio(key='page').set_value('Demand outlook').run()
    assert not at.exception
    assert at.title[0].value=='2027 demand outlook'
    assert not at.select_slider
    at.selectbox(key='family').set_value('DRAM').run()
    at.multiselect(key='programs').set_value(['DRAM_B']).run()
    at.selectbox(key='quarter').set_value('2027Q4').run()
    at.radio(key='demand_measure').set_value('Expected starts').run()
    at.selectbox(key='demand_breakdown').set_value('Route technology').run()
    assert not at.exception
    assert len(at.dataframe[0].value)==4
    at.radio(key='page').set_value('Capacity & scenarios').run()
    assert not at.exception
    assert at.selectbox(key='quarter').value=='2027Q4'


def test_demand_empty_selection_is_explained():
    app=Path(__file__).resolve().parents[1]/'code'/'apprag.py'
    at=AppTest.from_file(str(app),default_timeout=40).run()
    at.radio(key='page').set_value('Demand outlook').run()
    at.multiselect(key='programs').set_value([]).run()
    assert not at.exception
    assert any('at least one program' in info.value for info in at.info)
