"""Regenerate reproducible synthetic history, pilot study, planning tables and notes."""
from pathlib import Path
import argparse,json,sqlite3,time
import numpy as np
import pandas as pd
import fab_config as C
import fab_demo as D
from fab_flow import FabFlow,rng_for

ROOT=Path(__file__).resolve().parents[1];TABLES=ROOT/'data/fab/tables';DOCS=ROOT/'data/fab/documents'

def save(name,frame):
    frame.to_csv(TABLES/(name+'.csv'),index=False,float_format='%.9f')


def quarter_history():
    rows=[]
    for i,q in enumerate(C.HISTORY_QUARTERS):
        total=2200+60*i
        for family,fraction in [('DRAM',.75),('NAND',.15),('Experiments',.10)]:
            active=[(s,p,w) for s,programs in C.ACTUAL_WSPW_2026Q3.items() for p,w in programs.items() if D.TECHNOLOGY[p]==family and q>=C.PROGRAM_START[p]]
            target=int(round(total*fraction));weights=np.array([r[2] for r in active],float);exact=target*weights/weights.sum();counts=np.floor(exact).astype(int)
            for j in np.argsort(-(exact-counts))[:target-counts.sum()]:counts[j]+=1
            for (site,program,_),actual in zip(active,counts):
                rng=rng_for(61,q,site,program);request=int(round(actual/C.TRUE_REQUEST_BIAS[program]*rng.uniform(.97,1.03)))
                rows.append(dict(quarter=q,site=site,program=program,requested_wspw=request,actual_wspw=int(actual)))
    return pd.DataFrame(rows)


def plans(seeds=D.SEEDS):
    rows=[]
    for site,programs in C.REQUESTED_WSPW_2027.items():
        for p,values in programs.items():
            for q,w in zip(C.PLAN_QUARTERS,values):rows.append(dict(site=site,program=p,quarter=q,requested_wspw=w))
    frame=pd.DataFrame(rows);save('plan_2027',frame);save('wafer_start_history',quarter_history())
    save('demand_bridge_2026q4',D.demand_bridge())
    save('installed_tools',pd.DataFrame([dict(site=s,tool_group=g,tools=n) for s,groups in C.INSTALLED_TOOLS.items() for g,n in groups.items()]))
    save('tool_arrivals',pd.DataFrame(C.TOOL_ARRIVALS,columns=['site','tool_group','quarter','tools']))
    route=D.route_steps();save('route_steps',route)
    save('routes',pd.DataFrame([dict(program=p,program_name=v['name'],mask_layers=v['mask_layers'],tool_group=g,passes=n,time_mult=v['time_mult'].get(g,1.),tool_hours_per_wafer=D.hours_per_wafer(p,g)) for p,v in D.PROGRAMS.items() for g,n in v['passes'].items()]))
    save('program_catalog',pd.DataFrame([dict(program=p,program_name=v['name'],family=D.FAMILIES[p],technology_family=D.TECHNOLOGY[p],mask_layers=v['mask_layers'],route_version='SYN-V6-1',lot_size=25,assumed_target_days=D.TARGET_DAYS[p],qualified_fabs='|'.join(s for s in C.SITES if D.qualified(s,p)),first_release=D.LAUNCH_DATE if p==D.PILOT else 'Established program',provenance='Synthetic engineering assumptions') for p,v in D.PROGRAMS.items()]))
    save('qualification',pd.DataFrame([dict(site=s,program=p,qualified=D.qualified(s,p)) for s in C.SITES for p in D.PROGRAMS]))
    weekly=[];quarterly=[]
    for case in D.CASES:
        for q in C.PLAN_QUARTERS:
            lo=pd.Timestamp(int(q[:4]),3*(int(q[-1])-1)+1,1);hi=lo+pd.DateOffset(months=3)
            average=np.mean([D.pilot_rate(d,case) for d in pd.date_range(lo,hi-pd.Timedelta(days=1))])
            quarterly.append(dict(case=case,site='FAB_A',program=D.PILOT,quarter=q,requested_wspw=average))
        lo=pd.Timestamp('2027-01-01');hi=pd.Timestamp('2028-01-01');monday=lo-pd.Timedelta(days=lo.weekday())
        while monday<hi:
            start=max(lo,monday);end=min(hi,monday+pd.Timedelta(days=7))
            rate=np.mean([D.pilot_rate(d,case) for d in pd.date_range(start,end-pd.Timedelta(days=1))])
            weekly.append(dict(case=case,site='FAB_A',program=D.PILOT,week_start=monday.strftime('%Y-%m-%d'),period_start=start,period_end=end,requested_wspw=rate))
            monday+=pd.Timedelta(days=7)
    save('pilot_plan_quarterly',pd.DataFrame(quarterly));save('pilot_plan_weekly',pd.DataFrame(weekly))
    specification=dict(schema_version='v6',provenance='Entirely synthetic, independent of Micron',historical_period=[D.HISTORY_START,'2026-09-27'],planning_horizon='2027',technology_start_mix={'DRAM_including_HBM':.75,'NAND':.15,'experiments':.10},
        notes='75% wafer-start share is a demo assumption, not inferred from revenue. Fleet is illustrative and sized for the declared workload.',
        prior_year_bridge=dict(quarter='2026Q4',requested_wspw=D.PRIOR_Q4_REQUEST_WSPW,source='User-provided planning reference',allocation='Proportional to baseline 2027Q1 fab/program requests',actuals_available=False,used_for_training=False),
        study=dict(site='FAB_A',start=D.STUDY_START,weeks=D.STUDY_WEEKS,launch_date=D.LAUNCH_DATE,pilot=D.PILOT,seeds=list(seeds),cases=D.CASES),
        dispatch='Non-preemptive HOT before NORMAL; same process, move and hold distributions for both priorities',
        batching='FURN: compatible recipe only, at most 3 lots, 2 h maximum formation wait, fixed 6 h machine cycle',
        limitations=['No real fab validation','No chamber/operator/rework network','Non-preemptive capacity holds','Targets are illustrative, not customer promises','Replicate ranges quantify model randomness, not real-world predictive accuracy'])
    (ROOT/'data/fab/specification.json').write_text(json.dumps(specification,indent=2))


def history():
    incidents=[dict(event_id='SYN-OPS-001',site='FAB_A',tool_group='ETCH',start_time='2026-01-12T08:00:00',duration_hours=12,affected_tools=2,state='sched_down',reason='Preventive maintenance capacity hold'),
        dict(event_id='SYN-OPS-002',site='FAB_B',tool_group='DEP',start_time='2026-03-16T08:00:00',duration_hours=24,affected_tools=2,state='engineering',reason='Deposition recipe qualification window'),
        dict(event_id='SYN-OPS-003',site='FAB_A',tool_group='EUV',start_time='2026-06-15T08:00:00',duration_hours=36,affected_tools=1,state='unsched_down',reason='Scanner recovery and verification capacity hold'),
        dict(event_id='SYN-OPS-004',site='FAB_A',tool_group='DEP',start_time='2026-08-10T08:00:00',duration_hours=18,affected_tools=3,state='engineering',reason='Engineering characterization campaign'),
        dict(event_id='SYN-OPS-005',site='FAB_B',tool_group='FURN',start_time='2026-09-07T08:00:00',duration_hours=24,affected_tools=1,state='sched_down',reason='Furnace maintenance capacity hold')]
    results=[]
    for site in C.SITES:
        start=time.time();out=FabFlow(site,D.HISTORY_START,D.HISTORY_WEEKS,seed=42,historical=True,incidents=incidents).run().results();results.append(out)
        print(f'History {site}: {len(out["lots"])} lots, {time.time()-start:.1f}s',flush=True)
    lots=pd.concat([r['lots'] for r in results],ignore_index=True);weekly=pd.concat([r['weekly'] for r in results],ignore_index=True)
    equipment=pd.concat([r['equipment'] for r in results],ignore_index=True)
    save('sim_lots',lots);save('sim_fab_weekly',weekly);save('sim_tool_group_weekly',equipment)
    save('weekly_program_flow',pd.concat([r['program_weekly'] for r in results],ignore_index=True))
    save('operating_events',pd.concat([r['incidents'] for r in results],ignore_index=True))
    recent=equipment[equipment.week_start>='2026-06-29'];summary=[]
    for (site,group),rows in recent.groupby(['site','tool_group']):
        capacity=rows.calendar_h.sum();prod=rows.productive_h.sum();eng=rows.engineering_h.sum();sd=rows.sched_down_h.sum();ud=rows.unsched_down_h.sum();visits=rows.lot_steps.sum()
        summary.append(dict(site=site,tool_group=group,tools=C.INSTALLED_TOOLS[site][group],productive_pct=100*prod/capacity,standby_pct=100*rows.standby_h.sum()/capacity,engineering_pct=100*eng/capacity,sched_down_pct=100*sd/capacity,unsched_down_pct=100*ud/capacity,loading_pct=100*prod/(capacity-eng-sd-ud),avg_queue_h=np.average(rows.avg_queue_h,weights=rows.lot_steps) if visits else 0,avg_process_h=np.average(rows.avg_process_h,weights=rows.lot_steps) if visits else 0,lot_steps=visits,average_batch_fill=rows.batch_lots.sum()/rows.batch_count.sum() if rows.batch_count.sum() else np.nan))
    save('sim_tool_group_q3_2026',pd.DataFrame(summary))
    return results


def reconcile_history():
    flow=pd.read_csv(TABLES/'weekly_program_flow.csv')
    flow=flow.groupby(['site','program','week_start'],as_index=False).wafer_starts.sum()
    weeks=sorted(flow.week_start.unique())
    mapping={w:['2025Q4','2026Q1','2026Q2','2026Q3'][i//13] for i,w in enumerate(weeks)}
    flow['quarter']=flow.week_start.map(mapping)
    actual=flow.groupby(['quarter','site','program']).wafer_starts.mean()
    history=pd.read_csv(TABLES/'wafer_start_history.csv')
    history['actual_wspw']=history.actual_wspw.astype(float)
    history['period_basis']='Synthetic reporting-quarter weekly rate'
    for index,row in history.iterrows():
        key=(row.quarter,row.site,row.program)
        if key in actual.index:
            value=float(actual.loc[key]);rng=rng_for(61,*key)
            history.loc[index,'actual_wspw']=value
            history.loc[index,'requested_wspw']=round(value/C.TRUE_REQUEST_BIAS[row.program]*rng.uniform(.97,1.03))
            history.loc[index,'period_basis']='Mean of 13 matching reporting weeks in weekly_program_flow.csv'
    save('wafer_start_history',history)


def study(seeds):
    bundles=[];summaries=[];comparisons=[];snapshots=[];batches=[]
    for seed in seeds:
        base=None
        for case in D.CASES:
            start=time.time();out=FabFlow('FAB_A',D.STUDY_START,D.STUDY_WEEKS,seed=seed,case=case).run().results()
            if base is None:base=out
            # Exactly the same pre-launch state, including queue/hold/remaining time.
            fields=[c for c in out['initial'].columns if c!='case']
            a=base['initial'][fields].sort_values('lot_id').reset_index(drop=True)
            b=out['initial'][fields].sort_values('lot_id').reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)
            cohort=base['initial'][['lot_id','program','priority','assumed_target']]
            left=cohort.merge(base['lots'][['lot_id','end_time']],on='lot_id',how='left').rename(columns={'end_time':'baseline_completion'})
            paired=left.merge(out['lots'][['lot_id','end_time']],on='lot_id',how='left').rename(columns={'end_time':'scenario_completion'})
            paired['seed']=seed;paired['case']=case
            paired['additional_days']=(pd.to_datetime(paired.scenario_completion)-pd.to_datetime(paired.baseline_completion)).dt.total_seconds()/86400
            paired['matched']=paired.scenario_completion.notna()&paired.baseline_completion.notna()
            comparisons.append(paired)
            post=out['weekly'][out['weekly'].week_start>=D.LAUNCH_DATE]
            completed=out['lots'][out['lots'].status=='COMPLETE']
            summaries.append(dict(seed=seed,case=case,initial_existing_lots=len(cohort),matched_existing_lots=int(paired.matched.sum()),unfinished_existing_lots=int(paired.scenario_completion.isna().sum()),
                post_launch_avg_wip=post.wip_wafers.mean(),post_launch_peak_wip=post.wip_wafers.max(),
                mean_existing_delay_days=paired.additional_days.mean(),p90_existing_delay_days=paired.additional_days.quantile(.9),
                pilot_completed_lots=int((completed.program==D.PILOT).sum()),pilot_cycle_days=completed.loc[completed.program==D.PILOT,'cycle_time_days'].mean()))
            for key in ['weekly','program_weekly','equipment']:
                bundles.append((key,out[key]))
            if case=='baseline':snapshots.append(out['initial'])
            # One reproducible trace replicate is sufficient for source inspection.
            if seed==seeds[0]:
                trace=out['lots'].copy();trace['seed']=seed;trace['case']=case
                save('launch_lots_'+case,trace)
                batches.append(out['batches'])
            print(f'Study seed {seed} {case}: {time.time()-start:.1f}s, matched {paired.matched.sum()}/{len(paired)}, WIP {post.wip_wafers.mean():.0f}',flush=True)
    for key in ['weekly','program_weekly','equipment']:
        save('launch_'+key,pd.concat([f for name,f in bundles if name==key],ignore_index=True))
    save('launch_summary',pd.DataFrame(summaries));save('launch_lot_comparison',pd.concat(comparisons,ignore_index=True));save('launch_initial_wip',pd.concat(snapshots,ignore_index=True));save('launch_batches',pd.concat(batches,ignore_index=True))


def notes():
    for p in DOCS.rglob('*.txt'):
        if 'v6_records' not in p.parts and not p.read_text().startswith('ARCHIVED PROTOTYPE NOTE'):
            p.write_text('ARCHIVED PROTOTYPE NOTE — earlier synthetic assumptions. Do not substitute its counts or loading for the v6 tables.\n\n'+p.read_text())
    target=DOCS/'v6_records';target.mkdir(exist_ok=True)
    (target/'v7-demand-comparison-policy.txt').write_text('''SYNTHETIC DEMAND COMPARISON — SYN-V7-1
The user supplied a 2026Q4 combined requested rate of 2,950 WSPW as a planning reference. It was not present in the v6 historical tables.
The assumed fab/program split is proportional to the baseline 2027Q1 request mix. This allocation is not measured program-level Q4 history. No 2026Q4 actuals are supplied.
The Q4 reference is stored separately in demand_bridge_2026q4.csv and excluded from the realization model's training and backtest data.
The 2027Q1 baseline remains 2,800 WSPW: a decline of 150 WSPW, or approximately 5.1%, from the Q4 planning reference. Later 2027 ramp growth is a separate comparison.
Demand-chart dropdowns and legends affect only their own plot. Sidebar filters apply to page totals and all views. WSPW is a weekly rate, not a quarterly wafer-volume total.
''')
    events=pd.read_csv(TABLES/'operating_events.csv')
    for r in events.to_dict('records'):
        text=f'''SYNTHETIC OPERATING RECORD {r['event_id']}
Date: {r['start_time']}
Fab: {r['site']}; equipment group: {r['tool_group']}
Issue/change: {r['reason']}
Requested capacity hold: {r['affected_tools']} tools for {r['duration_hours']} hours.
Actual non-preemptive hold: {r['actual_start']} through {r['actual_end']}.
The hold waits for in-flight processing; no mid-process wafer interruption is modeled.
Investigate the corresponding weekly flow and equipment queue table. The record documents the hold; it does not establish that all observed cycle-time variation was caused by this event.
Source: operating_events.csv and sim_tool_group_weekly.csv. Entirely synthetic.
'''
        (target/(r['event_id']+'.txt')).write_text(text)
    for month,rows in pd.read_csv(TABLES/'sim_fab_weekly.csv').assign(month=lambda f:f.week_start.str[:7]).groupby('month'):
        lines=[f'SYNTHETIC WEEKLY FLOW REVIEW — {month}', 'Scope: simulated history, not 2027 demand or real Micron operations.']
        for site,g in rows.groupby('site'):
            lines.append(f'{site}: {len(g)} reporting weeks; mean wafer starts {g.wafer_starts.mean():.1f}/week, completions {g.wafer_outs.mean():.1f}/week, mean WIP {g.wip_wafers.mean():.1f} input-wafer equivalents.')
        lines.append('Review program mix, outages and queues before attributing causes. Source: sim_fab_weekly.csv, weekly_program_flow.csv and dated operating records.')
        (target/(month+'-flow-review.txt')).write_text('\n'.join(lines))
    (target/'2027-05-17-pilot-introduction.txt').write_text('''SYNTHETIC PLANNING CHANGE — PILOT-2027-001
Planned release: 17 May 2027, seventh week of the 16-week study beginning 05 April 2027.
Program: DRAM_P, DRAM EUV pilot. Qualified only at FAB_A; FAB_B has no EUV resource.
Introduction: 100 WSPW for two weeks, then 200 WSPW. Slower response: 50 WSPW for four weeks, then 75 WSPW.
Both cases retain existing release plans and matched per-lot random inputs. Furnace recipes for the pilot are distinct compatible-batch families. Equipment recipes, ordered visits and sampling are in route_steps.csv.
No historical request-realization coefficient exists for this new program. Its requested ramp is a planning assumption, not a fitted demand forecast.
Compare launch_summary.csv, launch_lot_comparison.csv, launch_initial_wip.csv and future capacity calculations. Delays are relative to the same existing lots; targets are illustrative and no customer promise is asserted.
Changing HOT priority is not a capacity increase. Consolidating EUV-required routes into FAB_B is not qualified. A slower ramp may reduce impact without resolving every equipment constraint.
''')
    (target/'v6-model-and-mix-policy.txt').write_text('''SYNTHETIC MODEL POLICY — SYN-V6-1
The requested 2027 baseline and historical actual-start mix target DRAM technology 75%, NAND 15%, experiments 10%. DRAM technology includes HBM test vehicles; display family HBM may remain separate. The 75% wafer-start share is a demonstration assumption, not inferred from Micron revenue share.
History: 29 September 2025 through 27 September 2026. Future plan: calendar 2027.
Resources are shared tool-group pools with re-entrant routes, compatible furnace batching, sampled metrology, stochastic non-preemptive capacity holds and engineering holds. All device recipes and equipment counts are synthetic.
The capacity screen charges average full-route tool hours per wafer start. The DES follows each lot's remaining route and queues. Do not substitute capacity loading for a prediction of cycle time or order completion.
Model random-seed ranges are not confidence bounds on real fab accuracy. No real fab MES data was used to validate the prototype.
''')


def database():
    path=ROOT/'data/fab/td_fab.db'
    with sqlite3.connect(path) as con:
        for p in sorted(TABLES.glob('*.csv')):pd.read_csv(p).to_sql(p.stem,con,if_exists='replace',index=False)
        con.execute('VACUUM')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=5);parser.add_argument('--skip-history',action='store_true');args=parser.parse_args()
    if not 1<=args.seeds<=len(D.SEEDS):parser.error(f'--seeds must be between 1 and {len(D.SEEDS)}')
    seeds=D.SEEDS[:args.seeds]
    TABLES.mkdir(parents=True,exist_ok=True);plans(seeds)
    if not args.skip_history:history()
    reconcile_history();study(seeds);notes();database()
    print('Synthetic data, notes and SQLite refreshed.',flush=True)

if __name__=='__main__':main()
