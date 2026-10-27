"""Capital view of the 2027 plan: which tools to add, by when, and which levers avoid buying.

Everything here reuses the analytical capacity model (route hours vs. available hours).
The added logic is bookkeeping a capital planner needs:

    tools needed (site, group, quarter) = ceil(required h/wk / (168 x availability x planning limit))
    committed tools                     = installed today + every tool already on order
    net-new tools                       = max over 2027 of (tools needed - committed), never below 0
    pull-in tools                       = on-order tools that are needed before their scheduled quarter

Tool counts are not costs. An EUV scanner and a wet bench both count as one tool here.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pandas as pd

import fab_config as C
import fab_demo as D
import forecast_model as fm
import launch_model as L

Q = C.PLAN_QUARTERS
SITES = list(C.SITES)
GROUPS = C.TOOL_GROUP_ORDER
PROGRAMS = list(D.PROGRAMS)
# A program can move between fabs only if its route is qualified at both.
TRANSFERABLE = [p for p in C.PROGRAM_ORDER if all(p in C.QUALIFIED[s] for s in SITES)]
HOURS = np.array([[D.hours_per_wafer(p, g) for g in GROUPS] for p in PROGRAMS])  # program x group


# ------------------------------------------------------------------ plan inputs
def realization_rates(data):
    return fm.fit(data['wafer_start_history']).set_index('program')['realization_rate'].astype(float)


AVAILABILITY = {'plan': 'Planning assumption', 'measured': 'Measured 2026Q4 to date'}


def measured_availability(data):
    """(productive + standby) ÷ calendar over 2026Q4 to date, by (site, tool group)."""
    t = data['sim_tool_group_weekly_q4'].groupby(['site', 'tool_group'])[['calendar_h', 'productive_h', 'standby_h']].sum()
    return ((t.productive_h + t.standby_h) / t.calendar_h).to_dict()


def apply_availability(state, data, availability='plan'):
    """Planning assumption (default) or measured 2026Q4 availability; furnaces keep their batch efficiency."""
    if availability == 'measured':
        for (s, g), a in measured_availability(data).items():
            state['avail'][s][g] = [float(a) * C.batch_efficiency(s, g)] * len(Q)
    elif availability != 'plan':
        raise ValueError('Availability must be plan or measured.')
    state['availability'] = availability
    return state


def plan_state(data, case='baseline', basis='requested', availability='plan'):
    """Requested plan, or requested x historical realization factor ('expected').

    The pilot has no history, so it stays at its requested ramp in both bases.
    """
    state = L.planning_state(data, case)
    if basis == 'expected':
        rates = realization_rates(data)
        for s in SITES:
            for p in C.PROGRAM_ORDER:
                state['wspw'][s][p] = [v * float(rates[p]) for v in state['wspw'][s][p]]
    elif basis != 'requested':
        raise ValueError('Basis must be requested or expected.')
    state['basis'] = basis
    return apply_availability(state, data, availability)


def _wspw(state):
    return np.array([[[state['wspw'][s].get(p, [0.] * 4)[i] for i in range(4)] for p in PROGRAMS] for s in SITES])


def _per_tool_hours(state):
    return np.array([[[168 * state['avail'][s][g][i] for i in range(4)] for g in GROUPS] for s in SITES])


def installed():
    return np.array([[C.INSTALLED_TOOLS[s][g] for g in GROUPS] for s in SITES], dtype=float)


def timed_tools():
    """Tools released to production in each quarter (installed + arrivals by their date)."""
    out = np.repeat(installed()[:, :, None], 4, axis=2)
    for s, g, quarter, n in C.TOOL_ARRIVALS:
        if quarter in Q:
            out[SITES.index(s), GROUPS.index(g), Q.index(quarter):] += n
    return out


def committed():
    """Installed + everything on order, regardless of arrival date."""
    out = installed()
    for s, g, quarter, n in C.TOOL_ARRIVALS:
        out[SITES.index(s), GROUPS.index(g)] += n
    return out


def tools_needed(wspw, per_tool, ceiling):
    """Works on a single plan (site, program, quarter) or a stack (n, site, program, quarter)."""
    required = np.einsum('...spq,pg->...sgq', wspw, HOURS)
    return np.ceil(required / (per_tool * ceiling) - 1e-9), required


# ------------------------------------------------------------------ buy list
def buy_list(data, case='baseline', basis='requested', ceiling=.85, state=None, availability='plan'):
    state = state or plan_state(data, case, basis, availability)
    per = _per_tool_hours(state)
    need, required = tools_needed(_wspw(state), per, ceiling)
    timed, inst, comm = timed_tools(), installed(), committed()
    rows = []
    for i, s in enumerate(SITES):
        for j, g in enumerate(GROUPS):
            n = need[i, j]
            if not n.any() and not inst[i, j]:
                continue
            timed_gap = np.clip(n - timed[i, j], 0, None)
            early = np.clip(np.minimum(n, comm[i, j]) - timed[i, j], 0, None)
            arrivals = [(q, k) for site, grp, q, k in C.TOOL_ARRIVALS if site == s and grp == g]
            loading = np.where(timed[i, j] > 0, 100 * required[i, j] / np.where(timed[i, j] > 0, timed[i, j] * per[i, j], 1), np.nan)
            short = [Q[k] for k in range(4) if timed_gap[k] > 0]
            # Tools must stay once installed: required fleet by quarter is the running max of need.
            fleet = np.maximum.accumulate(n)
            new_cum = np.clip(fleet - comm[i, j], 0, None)            # new tools that must be in service by each quarter
            new_add = np.diff(np.concatenate([[0.], new_cum]))         # new tools to bring in during each quarter
            pull_cum = np.minimum(np.clip(fleet - timed[i, j], 0, None), comm[i, j] - timed[i, j])
            pull_q = next((Q[k] for k in range(4) if pull_cum[k] > 0), None)
            in_service = np.maximum(fleet, timed[i, j])                 # after the response: buys + pull-ins in place
            after = np.where(in_service > 0, 100 * required[i, j] / np.where(in_service > 0, in_service * per[i, j], 1), np.nan)
            rows.append(dict(pull_in_by=pull_q,
                **{f'add_{q}': int(new_add[k]) for k, q in enumerate(Q)},
                **{f'fleet_{q}': int(fleet[k]) for k, q in enumerate(Q)},
                site=s, tool_group=g, installed=int(inst[i, j]),
                on_order=int(sum(k for _, k in arrivals)),
                on_order_label=', '.join(f'{k} in {q}' for q, k in arrivals) or '—',
                needed_peak=int(n.max()), peak_quarter=Q[int(n.argmax())],
                net_new=int(max(0, n.max() - comm[i, j])), pull_in=int(early.max()),
                need_by=short[0] if short else None,
                peak_loading_pct=float(np.nanmax(loading)) if np.isfinite(loading).any() else np.nan,
                **{f'gap_{q}': int(timed_gap[k]) for k, q in enumerate(Q)},
                **{f'load_{q}': float(loading[k]) for k, q in enumerate(Q)},
                **{f'after_{q}': float(after[k]) for k, q in enumerate(Q)}))
    frame = pd.DataFrame(rows)
    frame['action'] = np.select([frame.net_new.gt(0) & frame.pull_in.gt(0), frame.net_new.gt(0), frame.pull_in.gt(0)],
                                ['Buy + pull in', 'Buy', 'Pull in'], 'Covered')
    return frame


def totals(frame):
    action = frame[frame.action.ne('Covered')]
    need_by = sorted(action.need_by.dropna())
    largest = action.sort_values(['net_new', 'pull_in'], ascending=False)
    return dict(net_new=int(frame.net_new.sum()), pull_in=int(frame.pull_in.sum()),
                groups=int(len(action)), groups_total=int(len(frame)),
                first_need=need_by[0] if need_by else None,
                largest=[(r.site, r.tool_group, int(r.net_new)) for r in largest.head(3).itertuples() if r.net_new > 0])


def scope(frame, site):
    return frame if site == 'Both' else frame[frame.site == site]


# ------------------------------------------------------------------ levers
COST = np.array([C.RELATIVE_COST_INDEX[g] for g in GROUPS])


def cost_index(net_new):
    return float((np.asarray(net_new) * COST[None, :]).sum())


def optimize_transfers(state, ceiling, max_share=50, step=5, weighted=False):
    """Grid-search a constant 2027 share of each dual-qualified program moved between fabs.

    Objective: fewest net-new tools across both fabs; ties go to the smallest move.
    Network starts are conserved. EUV programs are never moved (not qualified at Fab B).
    """
    W = _wspw(state)
    per = _per_tool_hours(state)
    comm = committed()
    idx = [PROGRAMS.index(p) for p in TRANSFERABLE]
    shares = np.arange(-max_share, max_share + 1e-9, step) / 100  # + = Fab A -> Fab B
    combos = np.array(list(itertools.product(shares, repeat=len(idx))))
    stack = np.broadcast_to(W, (len(combos),) + W.shape).copy()
    moved = np.zeros(len(combos))
    for col, k in enumerate(idx):
        x = combos[:, col][:, None]
        a_to_b = np.clip(x, 0, None) * W[0, k]
        b_to_a = np.clip(-x, 0, None) * W[1, k]
        stack[:, 0, k] += b_to_a - a_to_b
        stack[:, 1, k] += a_to_b - b_to_a
        moved += (a_to_b + b_to_a).mean(axis=1)
    need, _ = tools_needed(stack, per, ceiling)
    gaps = np.clip(need.max(axis=-1) - comm, 0, None)
    count = gaps.sum(axis=(1, 2))
    net = (gaps * COST[None, None, :]).sum(axis=(1, 2)) if weighted else count
    best = int(np.lexsort((moved, count, net))[0])
    zero = int(np.flatnonzero((combos == 0).all(axis=1))[0])
    moves = []
    for col, p in enumerate(TRANSFERABLE):
        x = combos[best, col]
        if abs(x) < 1e-9:
            continue
        src, dst = ('FAB_A', 'FAB_B') if x > 0 else ('FAB_B', 'FAB_A')
        wafers = abs(x) * np.mean(W[SITES.index(src), PROGRAMS.index(p)])
        moves.append(dict(program=p, from_site=src, to_site=dst, share_pct=round(100 * abs(x)), avg_wspw=round(wafers)))
    return dict(before=float(net[zero]), after=float(net[best]), tools_before=int(count[zero]), tools_after=int(count[best]), moves=moves,
                moved_wspw=round(float(moved[best])), max_share=max_share, step=step,
                searched=len(combos))


def apply_moves(state, moves):
    import copy
    out = copy.deepcopy(state)
    for m in moves:
        share = m['share_pct'] / 100
        for i in range(4):
            amount = state['wspw'][m['from_site']][m['program']][i] * share
            out['wspw'][m['from_site']][m['program']][i] -= amount
            out['wspw'][m['to_site']][m['program']][i] += amount
    return out


def _net_new(state, ceiling):
    need, _ = tools_needed(_wspw(state), _per_tool_hours(state), ceiling)
    return np.clip(need.max(axis=-1) - committed(), 0, None)


def lever_steps(data, case='baseline', ceiling=.85, raise_pts=5, max_share=50):
    """Sequential capital-avoidance levers. Order matters; each step keeps the previous ones."""
    requested = plan_state(data, case, 'requested')
    expected = plan_state(data, case, 'expected')
    higher = min(.95, ceiling + raise_pts / 100)
    transfer = optimize_transfers(expected, higher, max_share)
    moved = apply_moves(expected, transfer['moves'])
    steps = [
        ('Requested plan', f'Program requests at a {ceiling*100:.0f}% limit', _net_new(requested, ceiling)),
        ('Plan to expected starts', 'Requests × historical realization factor', _net_new(expected, ceiling)),
        (f'Run tools to {higher*100:.0f}%', f'Planning limit +{(higher-ceiling)*100:.0f} pts', _net_new(expected, higher)),
        ('Rebalance qualified work', 'Best move of dual-qualified programs', _net_new(moved, higher)),
    ]
    rows = []
    for name, detail, arr in steps:
        rows.append(dict(step=name, detail=detail, net_new=int(arr.sum()),
                         **{f'{s}|{g}': int(arr[i, j]) for i, s in enumerate(SITES) for j, g in enumerate(GROUPS)}))
    return pd.DataFrame(rows), transfer, higher


def exposure(data, case='baseline', ceiling=.85):
    """Buy only for expected starts, then run the full request: which groups break?"""
    expected = plan_state(data, case, 'expected')
    requested = plan_state(data, case, 'requested')
    need_exp, _ = tools_needed(_wspw(expected), _per_tool_hours(expected), ceiling)
    fleet = np.maximum(need_exp.max(axis=-1), committed())  # tools bought for the expected plan
    _, req_hours = tools_needed(_wspw(requested), _per_tool_hours(requested), ceiling)
    with np.errstate(divide='ignore', invalid='ignore'):
        load = 100 * req_hours / (fleet[:, :, None] * _per_tool_hours(requested))
    load = np.where(np.isfinite(load), load, -1.)  # groups with no tools and no work
    peak = np.where(load.max(axis=-1) >= 0, load.max(axis=-1), np.nan)
    rows = [dict(site=s, tool_group=g, peak_pct=float(peak[i, j]))
            for i, s in enumerate(SITES) for j, g in enumerate(GROUPS) if np.isfinite(peak[i, j])]
    frame = pd.DataFrame(rows)
    return frame, int(frame.peak_pct.gt(100 * ceiling + 1e-9).sum()), int(frame.peak_pct.gt(100).sum())


def queue_multiplier(u_from, u_to):
    """Kingman/M-M-1 shape: queue time grows with u/(1-u)."""
    f = lambda u: u / (1 - u)
    return f(u_to) / f(u_from)


# ------------------------------------------------------------------ model check
def model_check(data):
    """Capacity-model loading for the last 13 simulated weeks vs. loading the simulation produced."""
    flow = data['weekly_program_flow']
    weeks = sorted(flow.week_start.unique())[-13:]
    starts = flow[flow.week_start.isin(weeks)].groupby(['site', 'program']).wafer_starts.sum() / len(weeks)
    sim = data['sim_tool_group_q3_2026']
    rows = []
    for r in sim.itertuples():
        required = sum(starts.get((r.site, p), 0.) * D.hours_per_wafer(p, r.tool_group) for p in C.PROGRAM_ORDER)
        available = C.INSTALLED_TOOLS[r.site][r.tool_group] * 168 * C.production_availability(r.tool_group)
        rows.append(dict(site=r.site, tool_group=r.tool_group, model_pct=100 * required / available,
                         model_batch_pct=100 * required / (available * C.batch_efficiency(r.site, r.tool_group)),
                         simulated_pct=float(r.loading_pct), queue_h=float(r.avg_queue_h),
                         assumed_nonproductive_pct=100 * (1 - C.production_availability(r.tool_group)),
                         observed_nonproductive_pct=float(r.engineering_pct + r.sched_down_pct + r.unsched_down_pct)))
    out = pd.DataFrame(rows)
    out['difference_pts'] = out.simulated_pct - out.model_pct
    return out


def headline(frame, ceiling, basis):
    from review_model import GROUP_LABELS, SITE_LABELS
    t = totals(frame)
    plan = 'requested' if basis == 'requested' else 'expected'
    limit = f'{ceiling*100:.0f}%'
    article = 'an' if limit.startswith('8') else 'a'
    if not t['groups']:
        return f'The {plan} 2027 plan fits installed and on-order tools at {article} {limit} loading limit.'
    names = ', '.join(f'{SITE_LABELS[s]} {GROUP_LABELS[g].lower()} ({n})' for s, g, n in t['largest'])
    parts = [f"At {article} {limit} loading limit the {plan} 2027 plan needs <b>{t['net_new']} more tools</b> than are installed or on order"]
    if t['pull_in']:
        parts.append(f"plus <b>{t['pull_in']}</b> on-order tools pulled in earlier")
    text = ', '.join(parts) + '.'
    if names:
        text += f' Largest gaps: {names}.'
    if t['first_need']:
        text += f" The first shortfall is in <b>{t['first_need']}</b>."
    return text


# ------------------------------------------------------------------ response plan (2026Q4 rate -> 2027 plan)
def overloads(state):
    """Tool groups that cannot run the plan on installed + on-order tools (released by date): load > 100%."""
    _, required = tools_needed(_wspw(state), _per_tool_hours(state), 1.)
    avail = timed_tools() * _per_tool_hours(state)
    rows = []
    for i, s in enumerate(SITES):
        for j, g in enumerate(GROUPS):
            for k, q in enumerate(Q):
                req, av = required[i, j, k], avail[i, j, k]
                if req > 0 and (av == 0 or req > av):
                    rows.append(dict(site=s, tool_group=g, quarter=q, load_pct=np.nan if av == 0 else 100 * req / av,
                                     short_h_wk=req - av))
    return pd.DataFrame(rows, columns=['site', 'tool_group', 'quarter', 'load_pct', 'short_h_wk'])


def _today_state(data, basis, availability='plan'):
    """Today's 2026Q4 rate held flat through 2027: requests, or forecast starts for the expected basis."""
    state = plan_state(data, 'baseline', 'requested', availability)
    col = 'requested_wspw' if basis == 'requested' else 'forecast_wspw'
    q4 = data['demand_bridge_2026q4'].set_index(['site', 'program'])[col]
    for s in SITES:
        for p in C.PROGRAM_ORDER:
            state['wspw'][s][p] = [float(q4.get((s, p), 0.))] * 4
    return state


def _increases_only(previous, current):
    """Current plan where programs grew, previous plan where they shrank (nothing gives way)."""
    out = copy.deepcopy(current)
    for s in SITES:
        for p in C.PROGRAM_ORDER:
            out['wspw'][s][p] = [max(a, b) for a, b in zip(previous['wspw'][s][p], current['wspw'][s][p])]
    return out


def freed_vs_added(data, case='baseline', basis='requested', quarter='2027Q3', availability='plan'):
    """Hours per week the 2027 plan adds (growing programs) and gives back (shrinking programs) versus today's 2026Q4 rate."""
    base = _today_state(data, basis, availability)
    new = plan_state(data, case, basis, availability)
    qi = Q.index(quarter)
    rows = []
    for i, s in enumerate(SITES):
        for j, g in enumerate(GROUPS):
            added = freed = 0.
            for p in C.PROGRAM_ORDER:
                d = (new['wspw'][s][p][qi] - base['wspw'][s][p][qi]) * D.hours_per_wafer(p, g)
                added += max(d, 0.)
                freed += max(-d, 0.)
            if added or freed:
                rows.append(dict(site=s, tool_group=g, added_h_wk=added, freed_h_wk=freed, net_h_wk=added - freed,
                                 offset_pct=100 * min(freed, added) / added if added else np.nan))
    return pd.DataFrame(rows)


def response_plan(data, case='baseline', basis='requested', ceiling=.85, max_share=50, availability='plan'):
    """Cheapest response to the latest plan: declines given back, then cross-fab rebalancing, then buy the rest."""
    current = _today_state(data, basis, availability)
    requested = plan_state(data, case, basis, availability)
    up_only = _increases_only(current, requested)
    transfer = optimize_transfers(requested, ceiling, max_share, weighted=True)
    moved = apply_moves(requested, transfer['moves'])
    steps = [('2026Q4 rate', "Today's run rate held flat", current),
             ('Growth only', 'If nothing else gave way', up_only),
             ('Reallocate freed capacity', 'Declining programs release shared tools', requested),
             ('Rebalance fabs', 'Cost-optimal move of dual-qualified work', moved)]
    rows = []
    for name, detail, st in steps:
        arr = _net_new(st, ceiling)
        rows.append(dict(step=name, detail=detail, tools=int(arr.sum()), cost_index=cost_index(arr),
                         **{f'{s}|{g}': int(arr[i, j]) for i, s in enumerate(SITES) for j, g in enumerate(GROUPS)}))
    final = buy_list(data, case, basis, ceiling, state=moved)
    final['cost_index'] = final.net_new * final.tool_group.map(C.RELATIVE_COST_INDEX)
    return dict(steps=pd.DataFrame(rows), transfer=transfer, buy=final, final_state=moved,
                overloads=overloads(requested), overloads_current=overloads(current))


# ------------------------------------------------------------------ planned fleet (follow the buy plan)
def planned_extras(data, basis='requested', ceiling=.85, availability='plan'):
    """extra_tools entries that take every group to its planned 2027 fleet: pull-ins + new buys, by quarter.

    Cumulative extras = max(0, planned fleet - tools released by date). Negative steps cancel a pulled-in
    tool when its original order arrives, so it is never counted twice.
    """
    b = buy_list(data, 'baseline', basis, ceiling, availability=availability)
    timed = timed_tools()
    extras = []
    for r in b.itertuples():
        i, j = SITES.index(r.site), GROUPS.index(r.tool_group)
        prev = 0
        for k, q in enumerate(Q):
            cum = max(0, getattr(r, f'fleet_{q}') - int(timed[i, j, k]))
            if cum != prev:
                extras.append([r.site, r.tool_group, q, int(cum - prev)])
            prev = cum
    return extras


def with_fleet(state, data, fleet='current', basis='requested', ceiling=.85, availability='plan'):
    """Planned fleet = the buy list sized at the same availability the scenario uses."""
    if fleet == 'planned':
        state['extra_tools'] = list(state.get('extra_tools', [])) + planned_extras(data, basis, ceiling, availability)
    elif fleet != 'current':
        raise ValueError('Fleet must be current or planned.')
    return state


def furnace_sensitivity(data, basis='requested', ceiling=.85, availability='plan', lots=(2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5)):
    """Peak 2027 furnace fleet needed vs. effective lots per 6-hour run (a furnace holds up to 4)."""
    rows = []
    for n in lots:
        state = plan_state(data, 'baseline', basis, availability)
        for s in SITES:
            base = state['avail'][s]['FURN'][0] / C.batch_efficiency(s, 'FURN')
            state['avail'][s]['FURN'] = [base * C.batch_efficiency(s, 'FURN', n)] * len(Q)
        b = buy_list(data, 'baseline', basis, ceiling, state=state)
        for r in b[b.tool_group == 'FURN'].itertuples():
            rows.append(dict(site=r.site, lots_per_run=n, needed_peak=int(r.needed_peak), committed=int(r.installed + r.on_order),
                             net_new=int(r.net_new)))
    return pd.DataFrame(rows)
