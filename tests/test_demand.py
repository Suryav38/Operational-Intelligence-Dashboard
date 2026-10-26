from ui_helpers import select_view
import pandas as pd
import numpy as np
import pytest
import demand_outlook as demand
import fab_capacity as fc
import forecast_model as fm
import fab_config as C
from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_next_year_totals_reconcile_to_requested_plan(data):
    frame=demand.plan(data)
    totals=demand.quarterly(frame)
    assert totals.requested_wspw.tolist()==[3130,3645,4030,4440]
    assert totals.requested_wspw.mean()==3811.25
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
    assert filtered.requested_wspw.tolist()==[110,100,105,110]


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




def test_year_boundary_is_continuous_and_q4_is_a_labelled_forecast(data):
    rows=demand.comparison_frame(data)
    totals=demand.comparison_totals(rows)
    # Requests run on from 2026Q3 through the Q4 forecast into 2027Q1 without a step down.
    q3,q4,q1=totals.loc['2026Q3','requested_wspw'],totals.loc['2026Q4','requested_wspw'],totals.loc['2027Q1','requested_wspw']
    assert q3==2942 and q4==2994 and q1==3130
    assert 1<=q1/q4<1.06  # the October plan grows from Q4; no artificial step down
    # Starts: 2026Q3 actual -> Q4 forecast -> 2027Q1 expected also continue.
    bridge=data['demand_bridge_2026q4']
    assert bridge.forecast_wspw.sum()==2789
    assert 1<=totals.loc['2027Q1','expected_wspw']/bridge.forecast_wspw.sum()<1.06
    assert totals.loc['2026Q4','actual_wspw']!=totals.loc['2026Q4','actual_wspw']
    assert totals.loc['2027Q1':,'actual_wspw'].isna().all()
    assert set(rows.query("quarter=='2026Q4'").record_kind)=={'q4_forecast'}
    assert '2026Q4' not in set(data['wafer_start_history'].quarter)

    pilot=demand.comparison_totals(demand.comparison_frame(data,programs=['DRAM_P'],pilot_case='introduction'))
    assert pilot.loc['2026Q1':'2026Q4','requested_wspw'].isna().all()
    assert pilot.loc['2027Q2':'2027Q4','expected_wspw'].isna().all()
    assert np.isnan(demand.comparison_changes(demand.comparison_frame(data,programs=['DRAM_P'],pilot_case='introduction'),'2027Q2')['yoy_pct'])


def test_chart_dropdowns_filter_program_data_and_preserve_unknown_actuals(data):
    frame=demand.comparison_frame(data)
    a=demand.year_comparison_chart(frame);b=demand.history_outlook_chart(frame)
    assert list(a.data[0].marker.pattern.shape)==['','','','/']
    assert np.isnan(b.data[1].y[3]) and all(np.isnan(x) for x in b.data[1].y[4:])
    for fig in [a,b]:
        buttons={v.label:v for v in fig.layout.updatemenus[0].buttons}
        assert {'All selected programs','DRAM incl. HBM','NAND','HBM only','DRAM Node B (EUV)'}.issubset(buttons)
        request=buttons['DRAM Node B (EUV)'].args[0]['y']
        expected=demand.comparison_totals(frame[frame.program=='DRAM_B'])
        plotted=request[0] if len(request[0])==8 else list(request[0])+list(request[1])
        np.testing.assert_allclose(plotted,expected.requested_wspw)
        assert fig.layout.legend.itemclick=='toggle'
        assert fig.to_json()


def test_q4_reference_does_not_train_realization_model(data):
    original=demand.plan(data)
    modified={**data,'demand_bridge_2026q4':data['demand_bridge_2026q4'].copy()}
    modified['demand_bridge_2026q4']['requested_wspw']*=2
    modified['demand_bridge_2026q4']['actual_wspw']=99999
    pd.testing.assert_frame_equal(original,demand.plan(modified))
