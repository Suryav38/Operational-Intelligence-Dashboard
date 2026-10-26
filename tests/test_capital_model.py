import numpy as np
import capital_model as K
import fab_config as C
import launch_model as L


def test_buy_list_matches_capacity_model_tools_needed(data):
    frame = K.buy_list(data, 'baseline', 'requested', .85)
    _, table = L.capacity(data, 'baseline', .85)
    peak = table.groupby(['site', 'tool_group']).tools_needed_at_ceiling.max()
    for r in frame.itertuples():
        assert r.needed_peak == peak[(r.site, r.tool_group)]
        committed = C.INSTALLED_TOOLS[r.site][r.tool_group] + r.on_order
        assert r.net_new == max(0, r.needed_peak - committed)


def test_on_order_euv_is_pulled_in_before_any_new_scanner(data):
    row = K.buy_list(data).query("site=='FAB_A' and tool_group=='EUV'").iloc[0]
    assert row.pull_in == 1 and row.need_by == '2027Q1'
    assert row.net_new == row.needed_peak - (row.installed + row.on_order)


def test_expected_basis_never_needs_more_tools_than_requested(data):
    req = K.buy_list(data, basis='requested').net_new.sum()
    exp = K.buy_list(data, basis='expected').net_new.sum()
    assert exp <= req


def test_pilot_raises_the_buy(data):
    assert K.buy_list(data, 'introduction').net_new.sum() > K.buy_list(data, 'baseline').net_new.sum()


def test_levers_are_monotone_and_transfer_conserves_starts(data):
    steps, transfer, higher = K.lever_steps(data, 'baseline', .85, 5, 50)
    v = steps.net_new.tolist()
    assert v == sorted(v, reverse=True) and v[0] == K.buy_list(data).net_new.sum()
    assert higher == .9
    state = K.plan_state(data, 'baseline', 'expected')
    moved = K.apply_moves(state, transfer['moves'])
    assert np.isclose(K._wspw(state).sum(), K._wspw(moved).sum())
    assert all(m['program'] in K.TRANSFERABLE and m['share_pct'] <= 50 for m in transfer['moves'])


def test_model_check_reproduces_simulation_except_batch_furnace(data):
    check = K.model_check(data)
    off = check[check.difference_pts.abs() > 3]
    assert list(zip(off.site, off.tool_group)) == [('FAB_B', 'FURN')]


def test_queue_multiplier():
    assert round(K.queue_multiplier(.85, .90), 2) == 1.59


def test_response_plan_steps_and_freed_capacity(data):
    r = K.response_plan(data)
    t = r['steps'].tools.tolist()
    assert t[1] >= t[2] >= t[3]               # reallocation and rebalancing never add tools
    fv = K.freed_vs_added(data).set_index(['site', 'tool_group'])
    assert fv.loc[('FAB_A', 'EUV'), 'freed_h_wk'] == 0   # NAND/short loops never use EUV
    assert fv.loc[('FAB_A', 'ETCH'), 'freed_h_wk'] > 0
