"""Lot-cohort metrics with consistent site, program, priority and week filters."""
from __future__ import annotations
import pandas as pd
import fab_config as C

FAMILIES={'DRAM_A':'DRAM','DRAM_B':'DRAM','NAND_C':'NAND','HBM_TV':'HBM','SHORT':'Experiments'}
COMPONENTS=['process_days','queue_days','transport_days','hold_days']

def program_options(site='Both',family='All families'):
    sites=list(C.SITES) if site=='Both' else [site]
    return [p for p in C.PROGRAM_ORDER if any(p in C.QUALIFIED[s] for s in sites)
            and (family=='All families' or FAMILIES[p]==family)]

def prepare_lots(data):
    lots=data['sim_lots'].copy()
    lots['started']=pd.to_datetime(lots.start_time)
    lots['ended']=pd.to_datetime(lots.end_time,errors='coerce')
    return lots

def execution(data,site='Both',programs=None,priority='All priorities',first=None,last=None,include_short=False):
    weeks=sorted(data['sim_fab_weekly'].week_start.unique())
    first=first or weeks[0];last=last or weeks[-1]
    if first not in weeks or last not in weeks or first>last:
        raise ValueError('Choose an ordered range within the simulated reporting weeks.')
    weeks=[w for w in weeks if first<=w<=last]
    start=pd.Timestamp(first);cutoff=pd.Timestamp(last)+pd.Timedelta(days=7)
    sites=list(C.SITES) if site=='Both' else [site]
    programs=list(C.PROGRAM_ORDER) if programs is None else list(programs)
    lots=prepare_lots(data)
    lots=lots[lots.site.isin(sites)&lots.program.isin(programs)].copy()
    if priority!='All priorities':lots=lots[lots.priority==priority].copy()
    done=lots[(lots.status=='COMPLETE')&lots.ended.ge(start)&lots.ended.lt(cutoff)].copy()
    cycles=done if include_short else done[done.program!='SHORT']
    wip=lots[lots.started.lt(cutoff)&(lots.ended.isna()|lots.ended.ge(cutoff))].copy()
    wip['age_days']=(cutoff-wip.started).dt.total_seconds()/86400
    wip['progress_pct']=100*wip.steps_done/wip.steps_total
    # Progress is observed only at the final simulation snapshot; it is not backcast.
    if cutoff < pd.Timestamp(sorted(data['sim_fab_weekly'].week_start.unique())[-1])+pd.Timedelta(days=7):
        wip['progress_pct']=float('nan')
    weekly=[]
    for s in sites:
        sl=lots[lots.site==s]
        for w in weeks:
            lo=pd.Timestamp(w);hi=lo+pd.Timedelta(days=7)
            completed=done[(done.site==s)&done.ended.ge(lo)&done.ended.lt(hi)]
            ct=completed if include_short else completed[completed.program!='SHORT']
            active=sl[sl.started.lt(hi)&(sl.ended.isna()|sl.ended.ge(hi))]
            weekly.append(dict(site=s,week_start=w,
                wafer_starts=sl.loc[sl.started.ge(lo)&sl.started.lt(hi),'wafers_in'].sum(),
                wafer_outs=completed.wafers_out.sum(),wip_wafers=active.wafers_in.sum(),
                completed_lots=len(completed),ct_lots=len(ct),
                avg_cycle_time_days=ct.cycle_time_days.mean()))
    by_site=pd.DataFrame(weekly)
    combined=by_site.groupby('week_start',as_index=False)[['wafer_starts','wafer_outs','wip_wafers','completed_lots','ct_lots']].sum()
    ct_by_week=cycles.assign(week_start=cycles.ended.dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')).groupby('week_start').cycle_time_days.mean()
    combined['avg_cycle_time_days']=combined.week_start.map(ct_by_week)
    denominator=done.wafers_in.sum()
    return dict(lots=lots,completed=done,cycles=cycles,wip_lots=wip,weekly=combined,by_site_week=by_site,
        avg_outs=combined.wafer_outs.mean(),avg_starts=combined.wafer_starts.mean(),
        avg_wip=combined.wip_wafers.mean(),cycle_days=cycles.cycle_time_days.mean(),
        p90_days=cycles.cycle_time_days.quantile(.9),
        yield_pct=100*done.wafers_out.sum()/denominator if denominator else float('nan'),
        breakdown=cycles[COMPONENTS].mean(),first=start,cutoff=cutoff,include_short=include_short)

def historical_attainment(data,site='Both',programs=None):
    frame=data['wafer_start_history'].copy()
    if site!='Both':frame=frame[frame.site==site]
    if programs is not None:frame=frame[frame.program.isin(programs)]
    out=frame.groupby('quarter',as_index=False)[['requested_wspw','actual_wspw']].sum()
    out['attainment_pct']=100*out.actual_wspw/out.requested_wspw.replace(0,float('nan'))
    return out

def equipment_health(data,site='Both',groups=None):
    frame=data['sim_tool_group_q3_2026'].copy()
    if site!='Both':frame=frame[frame.site==site]
    if groups is not None:frame=frame[frame.tool_group.isin(groups)]
    frame['downtime_pct']=frame.sched_down_pct+frame.unsched_down_pct
    frame['production_available_pct']=frame.productive_pct+frame.standby_pct
    return frame
