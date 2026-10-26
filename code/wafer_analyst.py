"""Wafer loading analyst: can we accept more wafers for a program, and what does it do to the others?

Everything here is deterministic Python. An LLM may *prepare* the request (request_parser.py)
and *explain* the result, but every number comes from this module.

Request  ->  weekly loading (route hours vs. available hours, both fabs, exact calendar overlap)
         ->  how much of the increase fits under the acceptance ceiling, week by week
         ->  a small set of permitted responses, each recalculated and ranked by explicit rules

Decision bands for a tool group in a week:
    <= planning limit            within policy
    planning limit .. ceiling    needs an exception (queue time rises, see Levers)
    > ceiling                    not accepted

Loading percentages do not predict queue time or milestone dates. Statements about delay
need the lot simulation, which this screen does not rerun.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

import fab_config as C
import fab_demo as D
import launch_model as L
import review_model as rm
import weekly_scenarios as W

Q = C.PLAN_QUARTERS
STEP = 5  # accepted quantities are rounded down to 5 wafers/week


# ------------------------------------------------------------------ inputs
def programs(data):
    return data['program_planning'].set_index('program')


def quarter_weeks(quarter):
    cal = W.calendar()
    # Weeks wholly inside the quarter; a week straddling two quarters mixes two plan rates.
    return [r.week_start for r in cal.itertuples()
            if W.quarter_of(r.period_start) == quarter == W.quarter_of(r.period_end - pd.Timedelta(days=1))]


def observed_availability(data, weeks=4):
    """Production availability measured over 2026Q4 to date: (productive + standby) / calendar."""
    t = data['sim_tool_group_weekly_q4']
    recent = sorted(t.week_start.unique())[-weeks:]
    t = t[t.week_start.isin(recent)].groupby(['site', 'tool_group'])[['calendar_h', 'productive_h', 'standby_h']].sum()
    return ((t.productive_h + t.standby_h) / t.calendar_h).to_dict()


BASES = {'run_rate': 'Planned run rate', 'requested': 'Requested plan'}


def base_state(data, case='baseline', availability='plan', basis='run_rate'):
    """Loading basis: the planned run rate (what will actually start) or the requested plan."""
    state = L.planning_state(data, case)
    if basis == 'run_rate':
        for s, programs in C.RUN_RATE_WSPW_2027.items():
            for p, values in programs.items():
                state['wspw'][s][p] = [float(v) for v in values]
    elif basis != 'requested':
        raise ValueError('Basis must be run_rate or requested.')
    if availability == 'observed':
        for (s, g), a in observed_availability(data).items():
            state['avail'][s][g] = [float(a)] * len(Q)
    return state


def current_wspw(data, site, program, quarter, case='baseline', basis='run_rate'):
    return float(base_state(data, case, 'plan', basis)['wspw'][site].get(program, [0.] * 4)[Q.index(quarter)])


def validate(data, request, case='baseline'):
    errors = []
    site, program, quarter = request.get('site'), request.get('program'), request.get('quarter')
    if site not in C.SITES:
        errors.append('Choose Fab A or Fab B.')
    if program not in C.PROGRAMS:
        errors.append('Choose an established program (the pilot ramp is set by the plan case).')
    elif site in C.SITES and program not in C.QUALIFIED[site]:
        errors.append(f"{C.PROGRAMS[program]['name']} is not qualified at {rm.SITE_LABELS[site]}.")
    if quarter not in Q:
        errors.append('Choose a 2027 quarter.')
    target = request.get('requested_wspw')
    if target is None or not math.isfinite(float(target)) or float(target) < 0:
        errors.append('Requested starts must be a non-negative weekly rate.')
    weeks = quarter_weeks(quarter) if quarter in Q else []
    first, last = request.get('first_week'), request.get('last_week')
    if weeks and (first not in weeks or last not in weeks or first > last):
        errors.append('Choose a start and end week inside the quarter, in order.')
    if program in (request.get('protected') or []):
        errors.append('The requesting program cannot also be protected.')
    return errors


# ------------------------------------------------------------------ calculation helpers
def _weekly(state, changed=None, first=None, last=None, case='baseline'):
    frame, _ = W.week_table(state, changed, first, last, case)
    return frame


def _quarter_days(week_start, quarter):
    row = W.calendar().set_index('week_start').loc[week_start]
    i = Q.index(quarter)
    return sum(f for j, f in W.segments(row.period_start, row.period_end) if j == i) * 7


def headroom(weekly, site, program, weeks, quarter, cap):
    """Largest extra weekly starts of `program` at `site` that keeps every tool group <= cap in every week."""
    rows = weekly[(weekly.site == site) & weekly.week_start.isin(weeks)]
    best, binding = math.inf, None
    for r in rows.itertuples():
        h = D.hours_per_wafer(program, r.tool_group) * _quarter_days(r.week_start, quarter) / 7
        if h <= 0:
            continue
        room = (cap * r.available_h_period - r.required_h_period) / h if r.available_h_period else 0.
        if room < best:
            best, binding = room, (r.tool_group, r.week_start)
    if best is math.inf:
        return math.inf, None
    return max(0., math.floor(best / STEP) * STEP), binding


def room_by_quarter(state, site, program, cap, case):
    weekly = _weekly(state, case=case)
    return {q: headroom(weekly, site, program, quarter_weeks(q), q, cap)[0] for q in Q}


def _with(state, changes):
    out = copy.deepcopy(state)
    for site, program, qi, delta in changes:
        out['wspw'][site][program][qi] = max(0., out['wspw'][site][program][qi] + delta)
    return out


def _peak(weekly, weeks, sites):
    rows = weekly[weekly.week_start.isin(weeks) & weekly.site.isin(sites)]
    rows = rows.assign(_l=rows.loading_pct.fillna(np.inf)).sort_values('_l', ascending=False)
    return rows.iloc[0]


def _status(load, limit, ceiling):
    if pd.isna(load) or load > ceiling * 100 + 1e-9:
        return 'Not accepted'
    if load > limit * 100 + 1e-9:
        return 'Needs exception'
    return 'Within limit'


def _score(weekly, base_weekly, weeks, sites, limit, ceiling):
    rows = weekly[weekly.week_start.isin(weeks) & weekly.site.isin(sites)]
    base = base_weekly[base_weekly.week_start.isin(weeks) & base_weekly.site.isin(sites)]
    merged = rows.merge(base, on=['site', 'tool_group', 'week_start'], suffixes=('', '_base'))
    worse = merged[merged.loading_pct > merged.loading_pct_base + .05]
    peak = _peak(weekly, weeks, sites)
    worst_new = worse.assign(_l=worse.loading_pct.fillna(np.inf)).sort_values('_l', ascending=False)
    top = worst_new.iloc[0] if len(worst_new) else peak
    return dict(peak_site=top.site, peak_group=top.tool_group, peak_load=float(top.loading_pct),
                status=_status(top.loading_pct, limit, ceiling) if len(worst_new) else 'Within limit',
                groups_above_limit=int(len(worse[worse.loading_pct > limit * 100].drop_duplicates(['site', 'tool_group']))))


# ------------------------------------------------------------------ main analysis
def analyze(data, request, case='baseline', limit=.85, ceiling=.95, availability='plan', basis='run_rate'):
    errors = validate(data, request, case)
    if errors:
        raise ValueError(' '.join(errors))
    site, program, quarter = request['site'], request['program'], request['quarter']
    first, last = request['first_week'], request['last_week']
    qi = Q.index(quarter)
    protected = list(request.get('protected') or [])
    plan = programs(data)
    state = base_state(data, case, availability, basis)
    current = float(state['wspw'][site][program][qi])
    target = float(request['requested_wspw'])
    delta = target - current
    weeks = [w for w in quarter_weeks(quarter) if first <= w <= last]
    other = 'FAB_B' if site == 'FAB_A' else 'FAB_A'
    base_weekly = _weekly(state, case=case)

    fit, binding = headroom(base_weekly, site, program, weeks, quarter, ceiling)
    fit_limit, _ = headroom(base_weekly, site, program, weeks, quarter, limit)
    accepted_now = min(delta, fit) if delta > 0 else delta

    # Critical week and bottleneck: where the full request needs the most hours above the ceiling.
    full_state = _with(state, [(site, program, qi, delta)])
    full_weekly = _weekly(state, full_state, first, last, case)
    window = full_weekly[full_weekly.week_start.isin(weeks) & (full_weekly.site == site)].merge(
        base_weekly[['site', 'tool_group', 'week_start', 'required_h_period', 'loading_pct']],
        on=['site', 'tool_group', 'week_start'], suffixes=('', '_base'))
    window['added_h'] = window.required_h_period - window.required_h_period_base
    window['deficit_h'] = window.required_h_period - ceiling * window.available_h_period
    touched = window[window.added_h > 1e-9]
    crit = touched.assign(_k=touched.loading_pct.fillna(np.inf)).sort_values(['_k', 'period_days'], ascending=False).iloc[0] if len(touched) else window.iloc[0]
    bottleneck = crit.tool_group
    scale = 7 / crit.period_days

    options = []

    def add_option(name, changes, sites, started_now, taken, how, later=None, phrase=None):
        st_ = _with(state, changes)
        wk = _weekly(state, st_, first, last, case)
        scored, score_weeks = wk, weeks
        if later:
            lq, rate = later
            lweeks = quarter_weeks(Q[lq])
            lw = _weekly(state, _with(state, [(site, program, lq, rate)]), lweeks[0], lweeks[-1], case)
            scored = pd.concat([wk[wk.week_start.isin(weeks)], lw[lw.week_start.isin(lweeks)]])
            score_weeks = weeks + lweeks
        score = _score(scored, base_weekly, score_weeks, sites, limit, ceiling)
        options.append(dict(option=name, started_now=started_now, deferred=max(0., delta - started_now) if delta > 0 else 0.,
                            taken_from_others=taken, how=how, changes=changes, weekly=wk, later=later,
                            phrase=phrase or name[0].lower() + name[1:], **score))

    add_option('Accept full request', [(site, program, qi, delta)], [site], delta, 0.,
               f'Run {target:,.0f} WSPW of {D.NAMES[program]} as requested.')
    rooms = room_by_quarter(state, site, program, ceiling, case)
    if delta > 0 and 0 < fit < delta:
        deferred = delta - fit
        total = deferred * len(weeks)  # wafers that still need a home
        later = next((q for q in Q[qi + 1:] if rooms[q] * len(quarter_weeks(q)) >= total), None)
        if later:
            rate = math.ceil(total / len(quarter_weeks(later)) / STEP) * STEP
            add_option('Accept part, defer the rest', [(site, program, qi, fit)], [site], fit, 0.,
                       f'Start +{fit:,.0f} WSPW now. Start the remaining {total:,.0f} wafers in {later} '
                       f'(+{rate:,.0f} WSPW), the first quarter with room under the {ceiling*100:.0f}% ceiling.',
                       later=(Q.index(later), rate), phrase=f'and defer the rest to {later}')
        else:
            add_option('Accept part, decline the rest', [(site, program, qi, fit)], [site], fit, 0.,
                       f'Start +{fit:,.0f} WSPW now. The remaining {deferred:,.0f} WSPW has no room under the '
                       f'{ceiling*100:.0f}% ceiling in any later 2027 quarter.', phrase='and decline the rest')

    # Offset: cut a lower-priority, unprotected program at the same fab that uses the bottleneck tools.
    if delta > fit:
        my_rank = int(plan.loc[program, 'priority'])
        candidates = []
        for p in C.QUALIFIED[site]:
            if p == program or p in protected or bool(plan.loc[p, 'protected']) or int(plan.loc[p, 'priority']) <= my_rank:
                continue
            if state['wspw'][site][p][qi] <= 0:
                continue
            candidates.append(p)
        best = None
        for p in sorted(candidates, key=lambda p: (-int(plan.loc[p, 'priority']), p)):
            need = 0.
            for r in touched.itertuples():
                excess = r.required_h_period - ceiling * r.available_h_period
                if excess <= 0:
                    continue
                h = D.hours_per_wafer(p, r.tool_group) * _quarter_days(r.week_start, quarter) / 7
                need = math.inf if h <= 0 else max(need, excess / h)
            if need == math.inf:
                continue
            cut = math.ceil(need / STEP) * STEP
            if cut <= state['wspw'][site][p][qi]:
                best = (p, cut)
                break
        if best:
            p, cut = best
            add_option(f'Reduce {D.NAMES[p]}', [(site, program, qi, delta), (site, p, qi, -cut)], [site], delta, cut,
                       f'Run the full increase and cut {D.NAMES[p]} by {cut:,.0f} WSPW (priority {int(plan.loc[p, "priority"])}, '
                       f'{plan.loc[p, "priority_label"].lower()}).')
        else:
            users = [D.NAMES[p] for p in C.QUALIFIED[site] if p != program and D.hours_per_wafer(p, bottleneck) > 0]
            blocked = [D.NAMES[p] for p in C.QUALIFIED[site] if p != program and D.hours_per_wafer(p, bottleneck) > 0
                       and (p in protected or bool(plan.loc[p, 'protected']) or int(plan.loc[p, 'priority']) <= my_rank)]
            options.append(dict(option='Reduce another program', started_now=None, deferred=None, taken_from_others=None, changes=[], weekly=None,
                                peak_site=site, peak_group=bottleneck, peak_load=np.nan, status='Not possible', groups_above_limit=None,
                                how=(f'Programs sharing {rm.GROUP_LABELS[bottleneck]} here: {", ".join(users) or "none"}. '
                                     + (f'Protected or same/higher priority, so not cut: {", ".join(blocked)}. ' if blocked else '')
                                     + ('No remaining lower-priority program frees enough hours on every tool group that goes over.'
                                        if len(blocked) < len(users) else ''))))

    # Alternate qualified tools: run the increase at the other fab, or move a dual-qualified program there.
    if delta > fit:
        if program in C.QUALIFIED[other]:
            stay = min(delta, fit)
            move = delta - stay
            if move > 0:
                add_option(f'Run the excess at {rm.SITE_LABELS[other]}', [(site, program, qi, stay), (other, program, qi, move)], [site, other],
                           delta, 0., f'Keep +{stay:,.0f} WSPW at {rm.SITE_LABELS[site]} and start {move:,.0f} WSPW on qualified tools at {rm.SITE_LABELS[other]}.')
        else:
            movable = [p for p in C.QUALIFIED[site] if p != program and p in C.QUALIFIED[other] and p not in protected
                       and not bool(plan.loc[p, 'protected']) and D.hours_per_wafer(p, bottleneck) > 0]
            done = False
            for p in movable:
                need = 0.
                for r in touched.itertuples():
                    excess = r.required_h_period - ceiling * r.available_h_period
                    if excess <= 0:
                        continue
                    h = D.hours_per_wafer(p, r.tool_group) * _quarter_days(r.week_start, quarter) / 7
                    need = math.inf if h <= 0 else max(need, excess / h)
                if need == math.inf:
                    continue
                shift = math.ceil(need / STEP) * STEP
                if shift <= state['wspw'][site][p][qi]:
                    add_option(f'Move {D.NAMES[p]} to {rm.SITE_LABELS[other]}',
                               [(site, program, qi, delta), (site, p, qi, -shift), (other, p, qi, shift)], [site, other], delta, shift,
                               f'Run the full increase; move {shift:,.0f} WSPW of {D.NAMES[p]} (qualified at both fabs) to {rm.SITE_LABELS[other]}.')
                    done = True
                    break
            if not done:
                reason = (f'{D.NAMES[program]} is not qualified at {rm.SITE_LABELS[other]}' +
                          (f', and {rm.SITE_LABELS[other]} has no {rm.GROUP_LABELS[bottleneck]} tools.'
                           if not C.INSTALLED_TOOLS[other].get(bottleneck) else
                           ', and no dual-qualified program can free enough bottleneck hours.'))
                options.append(dict(option='Use qualified tools at the other fab', started_now=None, deferred=None, taken_from_others=None,
                                    changes=[], weekly=None, peak_site=other, peak_group=bottleneck, peak_load=np.nan,
                                    status='Not possible', groups_above_limit=None, how=reason))

    ranked = rank(options)
    best = next((o for o in ranked if o['status'] in ('Within limit', 'Needs exception')), None)

    # Who shares the bottleneck, before and after the recommended option.
    rec_weekly = best['weekly'] if best else full_weekly
    wk = crit.week_start
    affected = []
    for p in list(C.PROGRAM_ORDER) + [D.PILOT]:
        h = D.hours_per_wafer(p, bottleneck)
        base_rate = state['wspw'][site].get(p, [0.] * 4)[qi]
        if p == D.PILOT:
            base_rate = D.pilot_rate(pd.Timestamp(wk) + pd.Timedelta(days=3), case) if site == 'FAB_A' else 0.
        if h <= 0 or base_rate <= 0:
            continue
        rec_rate = base_rate + sum(dlt for s, pr, i, dlt in (best['changes'] if best else []) if s == site and pr == p and i == qi)
        meta = plan.loc[p] if p in plan.index else None
        affected.append(dict(program=p, name=D.NAMES[p], priority=int(meta.priority) if meta is not None else None,
                             milestone=meta.milestone if meta is not None else '', milestone_date=meta.milestone_date if meta is not None else '',
                             protected=bool(meta.protected) or p in protected if meta is not None else p in protected,
                             base_wspw=base_rate, recommended_wspw=rec_rate,
                             base_hours=base_rate * h, recommended_hours=rec_rate * h,
                             requester=p == program))
    affected = pd.DataFrame(affected)
    if len(affected):
        affected['share_pct'] = 100 * affected.recommended_hours / affected.recommended_hours.sum()

    def at(frame, s, g, w):
        r = frame[(frame.site == s) & (frame.tool_group == g) & (frame.week_start == w)].iloc[0]
        return dict(required_h_wk=r.required_h_period * 7 / r.period_days, available_h_wk=r.available_h_period * 7 / r.period_days,
                    loading_pct=float(r.loading_pct))

    comparison = dict(baseline=at(base_weekly, site, bottleneck, wk), requested=at(full_weekly, site, bottleneck, wk),
                      recommended=at(rec_weekly, site, bottleneck, wk) if best else None)
    week_label = W.calendar().set_index('week_start').loc[wk, 'label']
    result = dict(request=dict(request, current_wspw=current, delta_wspw=delta, protected=protected),
                  case=case, limit=limit, ceiling=ceiling, availability=availability, basis=basis,
                  weeks=weeks, critical_week=wk, critical_week_label=week_label, bottleneck=bottleneck,
                  binding=binding, fits_under_ceiling=fit, fits_under_limit=fit_limit, accepted_now=accepted_now,
                  deficit_h_wk=max(0., float(crit.deficit_h) * scale), comparison=comparison,
                  options=ranked, recommended=best, affected=affected, room_by_quarter=rooms,
                  base_weekly=base_weekly, request_weekly=full_weekly)
    result['decision'] = decision_text(result)
    return result


RANK_RULES = [
    'Feasible: no tool group above the acceptance ceiling in any week of the window (both fabs when work moves).',
    'Protected programs are never reduced or moved.',
    'Most of the requested increase started in the requested weeks.',
    'Fewest wafers taken from, or moved for, other programs.',
    'Fewest tool groups pushed above the planning limit.',
]


def rank(options):
    order = {'Within limit': 0, 'Needs exception': 1, 'Not accepted': 2, 'Not possible': 3}
    feasible = lambda o: o['status'] in ('Within limit', 'Needs exception')
    return sorted(options, key=lambda o: (not feasible(o), -(o['started_now'] or 0) if feasible(o) else 0,
                                          o['taken_from_others'] or 0, order[o['status']], o['groups_above_limit'] or 0))


def decision_text(r):
    req = r['request']
    name = D.NAMES[req['program']]
    delta = r['delta_wspw'] if 'delta_wspw' in r else req['delta_wspw']
    best = r['recommended']
    site = rm.SITE_LABELS[req['site']]
    if delta <= 0:
        return f'This request lowers {name} starts; it frees capacity and needs no exception.'
    if best is None:
        return (f'None of the tested responses keeps {site} under the {r["ceiling"]*100:.0f}% acceptance ceiling. '
                f'The binding tool group is {rm.GROUP_LABELS[r["bottleneck"]]}; capacity must be added or the request declined.')
    text = f'Accept <b>+{best["started_now"]:,.0f}</b> of the requested +{delta:,.0f} {name} wafers/week in {req["quarter"]}'
    if best['option'] != 'Accept full request':
        text += f' — <b>{best["phrase"]}</b>'
    text += f'. {rm.GROUP_LABELS[r["bottleneck"]]} at {site} is the binding tool group'
    c = r['comparison']
    after = c['recommended']['loading_pct'] if c['recommended'] else np.nan
    text += f' ({c["baseline"]["loading_pct"]:.0f}% → {after:.0f}% in the week of {r["critical_week_label"]}).'
    if best['status'] == 'Needs exception':
        text += f' This is above the {r["limit"]*100:.0f}% planning limit, so it needs an exception.'
    prot = r['affected'][r['affected'].protected] if len(r['affected']) else []
    if len(prot):
        names = ', '.join(f'{p.name} ({p.milestone_date})' for p in prot.itertuples())
        text += f' Protected programs sharing these tools keep their starts: {names}.'
    return text
