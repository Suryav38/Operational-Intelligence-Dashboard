import json
from types import SimpleNamespace
import pytest
import fab_config as C
import planning_copilot as rag
import fab_data as fd
import wafer_analyst as A


def req(**kw):
    q = kw.get('quarter', '2027Q1')
    w = A.quarter_weeks(q)
    base = dict(site='FAB_A', program='DRAM_B', quarter=q, requested_wspw=850., protected=['HBM_TV'], first_week=w[0], last_week=w[-1])
    base.update(kw)
    return base


def test_quarter_weeks_exclude_straddling_weeks():
    weeks = A.quarter_weeks('2027Q1')
    assert weeks[0] == '2026-12-28' and weeks[-1] == '2027-03-22' and len(weeks) == 13


def test_euv_full_request_is_declined_and_protected_program_untouched(data):
    r = A.analyze(data, req(requested_wspw=925.))
    assert r['request']['current_wspw'] == 825 and r['bottleneck'] == 'EUV'
    # EUV: 117.6 h/wk; October plan (825+180)*7/60 = 117.25 h = 99.7% -> no room under a 95% ceiling
    assert r['comparison']['baseline']['required_h_wk'] == pytest.approx((825 + 180) * 7 / 60)
    assert r['fits_under_ceiling'] == 0 and r['recommended'] is None
    statuses = {o['option']: o['status'] for o in r['options']}
    assert statuses['Accept full request'] == 'Not accepted'
    assert statuses['Reduce another program'] == 'Not possible'
    assert statuses['Use qualified tools at the other fab'] == 'Not possible'
    assert 'capacity must be added' in r['decision']


def test_lower_priority_program_gives_way(data):
    r = A.analyze(data, req(site='FAB_B', program='DRAM_A', requested_wspw=730.))
    best = r['recommended']
    assert best['option'] == 'Reduce Short-loop experiments' and best['started_now'] == 50
    assert best['taken_from_others'] > 0


def test_dual_qualified_request_can_use_the_other_fab(data):
    r = A.analyze(data, req(program='SHORT', requested_wspw=240.))
    assert r['recommended']['option'] == 'Run the excess at Fab B' and r['recommended']['started_now'] == 100


def test_invalid_requests_are_rejected(data):
    with pytest.raises(ValueError, match='not qualified'):
        A.analyze(data, req(site='FAB_B'))
    with pytest.raises(ValueError, match='inside the quarter'):
        A.analyze(data, req(first_week='2027-05-03'))


def test_october_plan_is_july_times_submitted_changes():
    assert C.RUN_RATE_WSPW_2027['FAB_A']['DRAM_B'][1] == round(830 * C.OCTOBER_CHANGE['DRAM_B'][1] / 5) * 5
