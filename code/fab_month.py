"""Quarter-to-date fab performance: 2026Q4 from 28 Sep to 25 Oct 2026 (simulated continuation) against plan.

The point of this page is the operating conditions the 2027 scenarios start from: what the
line actually ran, how loaded each tool group was, and whether tools delivered the
availability the capacity model assumes.
"""
from __future__ import annotations

import pandas as pd

import fab_config as C
import execution_model as em

WEEKS = 4


def window(data, weeks=WEEKS):
    return sorted(data['sim_fab_weekly_q4'].week_start.unique())[-weeks:]


def _sites(site):
    return list(C.SITES) if site == 'Both' else [site]


def summary(data, site='Both', weeks=WEEKS):
    wk = window(data, weeks)
    sites = _sites(site)
    fab = data['sim_fab_weekly_q4']
    fab = fab[fab.week_start.isin(wk) & fab.site.isin(sites)]
    weekly = fab.groupby('week_start', as_index=False)[['wafer_starts', 'wafer_outs', 'scrap_wafers', 'wip_wafers']].sum()
    ct = fab.assign(w=fab.wafer_outs).groupby('week_start').apply(
        lambda g: (g.avg_cycle_time_days * g.w).sum() / g.w.sum() if g.w.sum() else float('nan'), include_groups=False)
    weekly['cycle_time_days'] = weekly.week_start.map(ct)

    quarter = '2026Q4'
    plan = data['demand_bridge_2026q4']
    plan = plan[plan.site.isin(sites)]
    plan_wspw = float(plan.requested_wspw.sum())

    lots = data['sim_lots_q4'].copy()
    lots['ended'] = pd.to_datetime(lots.end_time, errors='coerce')
    start, end = pd.Timestamp(wk[0]), pd.Timestamp(wk[-1]) + pd.Timedelta(days=7)
    done = lots[lots.site.isin(sites) & lots.ended.ge(start) & lots.ended.lt(end) & lots.status.eq('COMPLETE')]
    full = done[done.program.ne('SHORT')]

    flow = data['weekly_program_flow_q4']
    flow = flow[flow.week_start.isin(wk) & flow.site.isin(sites)]
    by_program = flow.groupby('program').wafer_starts.sum() / len(wk)
    planned = plan.groupby('program').requested_wspw.sum()
    programs = pd.DataFrame({'actual_wspw': by_program, 'plan_wspw': planned}).fillna(0.)
    programs['attainment_pct'] = 100 * programs.actual_wspw / programs.plan_wspw.where(programs.plan_wspw > 0)

    return dict(weeks=wk, start=start, end=end - pd.Timedelta(days=1), plan_quarter=quarter, weekly=weekly,
                starts_wspw=float(weekly.wafer_starts.mean()), outs_wspw=float(weekly.wafer_outs.mean()),
                plan_wspw=plan_wspw, attainment_pct=100 * float(weekly.wafer_starts.mean()) / plan_wspw if plan_wspw else float('nan'),
                wip=float(weekly.wip_wafers.mean()), cycle_days=float(full.cycle_time_days.mean()),
                cycle_p90=float(full.cycle_time_days.quantile(.9)) if len(full) else float('nan'),
                hold_share_pct=100 * float(full.hold_days.sum() / full.cycle_time_days.sum()) if len(full) else float('nan'),
                queue_share_pct=100 * float(full.queue_days.sum() / full.cycle_time_days.sum()) if len(full) else float('nan'),
                yield_pct=100 * float(done.wafers_out.sum() / done.wafers_in.sum()) if len(done) else float('nan'),
                completed_lots=int(len(done)), programs=programs.reset_index().rename(columns={'index': 'program'}))


def tools(data, site='Both', weeks=WEEKS):
    wk = window(data, weeks)
    t = data['sim_tool_group_weekly_q4']
    t = t[t.week_start.isin(wk) & t.site.isin(_sites(site))]
    rows = []
    for (s, g), r in t.groupby(['site', 'tool_group']):
        cal = r.calendar_h.sum()
        nonprod = r.engineering_h.sum() + r.sched_down_h.sum() + r.unsched_down_h.sum()
        visits = r.lot_steps.sum()
        rows.append(dict(site=s, tool_group=g, tools=float(r.tools.mean()),
                         observed_avail_pct=100 * (cal - nonprod) / cal,
                         assumed_avail_pct=100 * C.production_availability(g),
                         loading_pct=100 * r.productive_h.sum() / (cal - nonprod),
                         queue_h=float((r.avg_queue_h * r.lot_steps).sum() / visits) if visits else 0.,
                         unsched_down_pct=100 * r.unsched_down_h.sum() / cal))
    out = pd.DataFrame(rows)
    out['avail_gap_pts'] = out.observed_avail_pct - out.assumed_avail_pct
    return out
