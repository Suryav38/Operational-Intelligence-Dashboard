"""Re-entrant, lot-level synthetic fab DES with matched per-lot random streams.

Single-wafer-equivalent tool groups are pooled identical machines. Furnaces
form compatible batches of up to three lots (2 h maximum formation wait).
Resources are non-preemptive; a capacity hold starts after in-flight work.
No actual fab traveler, electrical yield, operator or chamber model is claimed.
"""
from collections import defaultdict
from datetime import timedelta
import hashlib
import math
import numpy as np
import pandas as pd
import simpy
import fab_config as C
import fab_demo as D


def rng_for(seed,*keys):
    payload=('|'.join(map(str,(seed,*keys)))).encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),'little'))


class FabFlow:
    def __init__(self,site,measure_start,weeks,seed=7,case='baseline',historical=False,warmup_weeks=16,incidents=None):
        self.site=site;self.seed=seed;self.case=case;self.historical=historical
        self.measure=pd.Timestamp(measure_start);self.origin=self.measure-pd.Timedelta(weeks=warmup_weeks)
        self.warm=warmup_weeks*168.;self.end=self.warm+weeks*168.;self.weeks=weeks
        self.env=simpy.Environment();self.active={};self.finished=[];self.initial=[]
        self.acc=defaultdict(lambda:defaultdict(float));self.toolacc=defaultdict(lambda:defaultdict(float))
        self.weekly=[];self.program_weekly=[];self.incident_log=[];self.batches=[]
        route=D.route_steps();self.routes={p:list(g.itertuples(index=False)) for p,g in route.groupby('program') if D.qualified(site,p)}
        self.arrivals=[]
        for s,g,q,n in C.TOOL_ARRIVALS:
            when=pd.Timestamp(int(q[:4]),3*(int(q[-1])-1)+1,1)
            if s==site and when<self.timestamp(self.end):self.arrivals.append((g,self.hours(when),n))
        self.capacity={g:n+sum(k for gg,t,k in self.arrivals if gg==g) for g,n in C.INSTALLED_TOOLS[site].items() if n>0}
        self.resources={g:simpy.PriorityResource(self.env,capacity=n) for g,n in self.capacity.items()}
        self.batch_queue=[];self.batch_signal=self.env.event()
        self.incidents=incidents or []

    def hours(self,date):return (pd.Timestamp(date)-self.origin).total_seconds()/3600
    def timestamp(self,h):return self.origin+pd.Timedelta(hours=float(h))
    def iso(self,h):return None if h is None else self.timestamp(h).isoformat(timespec='microseconds')
    def week(self,t):return int((t-self.warm)//168) if self.warm<=t<self.end else None

    def account(self,group,state,start,end):
        lo=max(start,self.warm);hi=min(end,self.end)
        while lo<hi-1e-8:
            week=min(self.weeks-1,int((lo-self.warm)//168));edge=min(hi,self.warm+(week+1)*168)
            self.toolacc[(week,group)][state]+=edge-lo;lo=edge

    def record_state(self,rec,state,group=None,until=None):
        rec['operation_state']=state;rec['current_tool_group']=group or ''
        rec['state_start_h']=self.env.now;rec['state_end_h']=until

    def future_tool(self,group,when):
        with self.resources[group].request(priority=-100) as token:
            yield token
            yield self.env.timeout(max(0,when-self.env.now))

    def availability(self,group,index,start=0):
        if start>0:yield self.env.timeout(start)
        rng=rng_for(self.seed,self.site,group,index,'availability');spec=C.TOOL_GROUPS[group]
        states=[('sched_down',spec['pm'],8.),('unsched_down',spec['down'],4.),('engineering',spec['eng'],6.)]
        frequencies=np.array([f/d for _,f,d in states]);gap=(1-sum(f for _,f,_ in states))/frequencies.sum()
        while True:
            yield self.env.timeout(rng.exponential(gap))
            state,_,mean=states[int(rng.choice(3,p=frequencies/frequencies.sum()))]
            with self.resources[group].request(priority=-10) as token:
                yield token;duration=float(rng.exponential(mean));start=self.env.now
                self.account(group,state,start,start+duration)
                yield self.env.timeout(duration)

    def incident(self,event):
        when=self.hours(event['start_time'])
        if when<0 or when>=self.end:return
        yield self.env.timeout(when)
        group=event['tool_group'];requests=[self.resources[group].request(priority=-20) for _ in range(int(event['affected_tools']))]
        yield self.env.all_of(requests);start=self.env.now;duration=float(event['duration_hours'])
        self.account(group,event['state'],start,start+duration)
        # account multiple unavailable machines separately
        for _ in requests[1:]:self.account(group,event['state'],start,start+duration)
        self.incident_log.append({**event,'actual_start':self.iso(start),'actual_end':self.iso(start+duration),'site':self.site,'case':self.case,'seed':self.seed})
        yield self.env.timeout(duration)
        for token in requests:self.resources[group].release(token)

    def batch_dispatch(self):
        while True:
            if not self.batch_queue:
                yield self.batch_signal
                self.batch_signal=self.env.event();continue
            self.batch_queue.sort(key=lambda j:(j['priority'],j['queued'],j['rec']['lot_id']))
            lead=self.batch_queue[0]
            compatible=[j for j in self.batch_queue if j['recipe']==lead['recipe']]
            remaining=lead['queued']+2.-self.env.now
            if len(compatible)<3 and remaining>1e-8:
                signal=self.batch_signal
                yield signal|self.env.timeout(remaining)
                if signal.triggered:self.batch_signal=self.env.event()
                continue
            batch=compatible[:3]
            for job in batch:self.batch_queue.remove(job)
            self.env.process(self.batch_run(batch))

    def batch_run(self,batch):
        with self.resources['FURN'].request(priority=min(j['priority'] for j in batch)) as token:
            yield token;start=self.env.now;duration=6.
            for job in batch:self.record_state(job['rec'],'processing','FURN',start+duration)
            self.account('FURN','productive',start,start+duration)
            week=self.week(start)
            if week is not None:
                self.toolacc[(week,'FURN')]['batch_count']+=1
                self.toolacc[(week,'FURN')]['batch_lots']+=len(batch)
            self.batches.append(dict(site=self.site,seed=self.seed,case=self.case,start_time=self.iso(start),recipe_id=batch[0]['recipe'],lots=len(batch),lot_ids='|'.join(j['rec']['lot_id'] for j in batch)))
            yield self.env.timeout(duration)
            for job in batch:job['event'].succeed((start,duration))

    def lot(self,program,lot_id):
        assert D.qualified(self.site,program),'Unqualified release'
        rng=rng_for(self.seed,self.site,lot_id);route=self.routes[program];n=len(route)
        hot=bool(rng.random()<.10);priority=0 if hot else 1
        move=rng.exponential(.45,n);mult=rng.gamma(16,1/16,n)
        hold=rng.random(n)<.02;hold_h=rng.exponential(24.,n);sample=rng.random(n)
        start=self.env.now;wafers=C.LOT_SIZE;survive=C.LINE_YIELD.get(program,.965)**(1/n)
        rec=dict(lot_id=lot_id,site=self.site,program=program,priority='HOT' if hot else 'NORMAL',start_h=start,
            wafers_in=C.LOT_SIZE,current_wafers=wafers,steps_total=n,steps_done=0,current_step=1,route_version='SYN-V6-1',
            process_h=0.,queue_h=0.,transport_h=0.,hold_h=0.,raw_process_h=0.,target_h=start+D.TARGET_DAYS[program]*24)
        self.active[lot_id]=rec;week=self.week(start)
        if week is not None:self.acc[(week,program,rec['priority'])]['starts']+=wafers
        for i,op in enumerate(route):
            rec['current_step']=i+1;rec['steps_done']=i
            if op.tool_group=='METRO' and sample[i]>op.sample_rate:continue
            self.record_state(rec,'transport',op.tool_group,self.env.now+move[i]);yield self.env.timeout(float(move[i]));rec['transport_h']+=move[i]
            queued=self.env.now;self.record_state(rec,'queued',op.tool_group)
            if op.tool_group=='FURN':
                event=self.env.event();self.batch_queue.append(dict(rec=rec,event=event,recipe=op.recipe_id,priority=priority,queued=queued))
                if not self.batch_signal.triggered:self.batch_signal.succeed()
                began,duration=yield event;nominal=6.
            else:
                with self.resources[op.tool_group].request(priority=priority) as token:
                    yield token;began=self.env.now;nominal=wafers*op.min_per_wafer/60;duration=nominal*mult[i]
                    self.record_state(rec,'processing',op.tool_group,began+duration)
                    self.account(op.tool_group,'productive',began,began+duration)
                    yield self.env.timeout(float(duration))
            wait=began-queued;rec['queue_h']+=wait;rec['process_h']+=duration;rec['raw_process_h']+=nominal
            week=self.week(self.env.now)
            if week is not None:
                stats=self.toolacc[(week,op.tool_group)];stats['queue_h']+=wait;stats['visits']+=1;stats['process_h']+=duration
                self.acc[(week,program,rec['priority'])]['moves']+=wafers
            remaining=int(rng.binomial(wafers,survive));scrap=wafers-remaining;wafers=remaining;rec['current_wafers']=wafers
            rec['steps_done']=i+1
            if week is not None:self.acc[(week,program,rec['priority'])]['scrap']+=scrap
            if not wafers:break
            if hold[i]:
                self.record_state(rec,'engineering_hold',op.tool_group,self.env.now+hold_h[i]);yield self.env.timeout(float(hold_h[i]));rec['hold_h']+=hold_h[i]
        end=self.env.now;self.active.pop(lot_id);rec.update(end_h=end,wafers_out=wafers,status='COMPLETE' if wafers else 'SCRAPPED',steps_done=n)
        self.finished.append(rec);week=self.week(end)
        if week is not None:self.acc[(week,program,rec['priority'])]['outs']+=wafers

    def baseline_rate(self,program,date):
        if not self.historical:
            q=f'{date.year}Q{(date.month-1)//3+1}'
            if q not in C.PLAN_QUARTERS:q=C.PLAN_QUARTERS[0]
            return C.PILOT_STUDY_RUN_RATE_2027[self.site][program][C.PLAN_QUARTERS.index(q)]
        # synthetic trend and variation; historical technology-family share is ~75%
        progress=min(1.,max(0.,(date-pd.Timestamp(D.HISTORY_START)).days/(52*7)))
        trend=.86+.14*progress
        raw=C.ACTUAL_WSPW_2026Q3[self.site][program]
        common=1+.025*math.sin(2*math.pi*progress*4)
        return raw*trend*common

    def releases(self,program):
        for week in range(int(self.end//168)):
            date=self.timestamp(week*168)
            rate=D.pilot_rate(date,self.case) if program==D.PILOT else self.baseline_rate(program,date)
            if program==D.PILOT and self.site!='FAB_A':continue
            rng=rng_for(self.seed,self.site,program,str(date.date()),'releases')
            exact=rate/C.LOT_SIZE;count=int(exact)+(rng.random()<exact-int(exact))
            for j in range(count):
                when=week*168+(j+.2+.6*rng.random())*168/max(count,1)
                yield self.env.timeout(max(0,when-self.env.now))
                lot_id=f'{self.site[-1]}-{program}-{date:%Y%m%d}-{j:03d}'
                self.env.process(self.lot(program,lot_id))
            if self.env.now<(week+1)*168:yield self.env.timeout((week+1)*168-self.env.now)

    def snapshot(self,cut):
        for rec in self.active.values():
            remainder=max(0.,(rec.get('state_end_h') or cut)-cut)
            self.initial.append(dict(seed=self.seed,case=self.case,lot_id=rec['lot_id'],site=self.site,program=rec['program'],priority=rec['priority'],
                snapshot_time=self.iso(cut),start_time=self.iso(rec['start_h']),current_step=rec['current_step'],steps_total=rec['steps_total'],
                steps_done=rec['steps_done'],remaining_route_steps=rec['steps_total']-rec['steps_done'],
                current_tool_group=rec['current_tool_group'],operation_state=rec['operation_state'],remaining_state_hours=remainder,
                current_wafers=rec['current_wafers'],wafers_in=rec['wafers_in'],route_version=rec['route_version'],
                assumed_target=self.iso(rec['target_h'])))

    def cutover(self):
        when=self.hours(D.LAUNCH_DATE)
        if when<0 or when>=self.end:return
        yield self.env.timeout(when);self.snapshot(when)

    def snapshots(self):
        yield self.env.timeout(self.warm)
        for week in range(self.weeks):
            opening=sum(r['current_wafers'] for r in self.active.values())
            yield self.env.timeout(168.)
            date=(self.measure+pd.Timedelta(weeks=week)).strftime('%Y-%m-%d')
            for program in self.routes:
                if program==D.PILOT and self.historical:continue
                for priority in ['HOT','NORMAL']:
                    lots=[r for r in self.active.values() if r['program']==program and r['priority']==priority]
                    stat=self.acc[(week,program,priority)]
                    self.program_weekly.append(dict(site=self.site,seed=self.seed,case=self.case,week_start=date,program=program,priority=priority,
                        wafer_starts=int(stat['starts']),wafer_outs=int(stat['outs']),scrap_wafers=int(stat['scrap']),
                        wip_wafers=sum(r['wafers_in'] for r in lots),wip_current_wafers=sum(r['current_wafers'] for r in lots),wip_lots=len(lots),moves_per_day=stat['moves']/7))
            stats=[r for r in self.program_weekly if r['week_start']==date]
            final=sum(r['current_wafers'] for r in self.active.values())
            starts=sum(r['wafer_starts'] for r in stats);outs=sum(r['wafer_outs'] for r in stats);scrap=sum(r['scrap_wafers'] for r in stats)
            self.weekly.append(dict(site=self.site,seed=self.seed,case=self.case,week_start=date,wafer_starts=starts,wafer_outs=outs,scrap_wafers=scrap,
                opening_current_wafers=opening,wip_current_wafers=final,wip_wafers=sum(r['wafers_in'] for r in self.active.values()),
                moves_per_day=sum(r['moves_per_day'] for r in stats),flow_balance_error=opening+starts-outs-scrap-final))

    def run(self):
        for group,n in C.INSTALLED_TOOLS[self.site].items():
            if n<=0:continue
            for index in range(n):self.env.process(self.availability(group,index))
        for group,when,n in self.arrivals:
            for index in range(n):
                if when>0:self.env.process(self.future_tool(group,when))
                self.env.process(self.availability(group,1000+index,max(0,when)))
        for event in self.incidents:
            if event['site']==self.site:self.env.process(self.incident(event))
        self.env.process(self.batch_dispatch())
        self.env.process(self.cutover())
        for program in C.QUALIFIED[self.site]:self.env.process(self.releases(program))
        if not self.historical and self.site=='FAB_A' and self.case!='baseline':self.env.process(self.releases(D.PILOT))
        self.env.process(self.snapshots())
        self.env.run(until=self.end+1e-6)
        return self

    def results(self):
        lots=[]
        for r in self.finished+list(self.active.values()):
            end=r.get('end_h')
            if end is not None and end<self.warm:continue
            completed=end is not None
            lots.append(dict(lot_id=r['lot_id'],site=self.site,program=r['program'],priority=r['priority'],start_time=self.iso(r['start_h']),end_time=self.iso(end),
                status=r.get('status','WIP'),wafers_in=r['wafers_in'],wafers_out=r.get('wafers_out'),steps_total=r['steps_total'],steps_done=r['steps_done'],
                raw_process_h=r['raw_process_h'] if completed else None,cycle_time_days=(end-r['start_h'])/24 if completed else None,
                **{k+'_days':r[k+'_h']/24 if completed else None for k in ['process','queue','transport','hold']},
                assumed_target=self.iso(r['target_h']),route_version=r['route_version'],current_step=r['current_step'],
                current_tool_group=r['current_tool_group'],operation_state=r['operation_state'],current_wafers=r['current_wafers']))
        lots=pd.DataFrame(lots);weekly=pd.DataFrame(self.weekly)
        if not lots.empty:
            completed=lots[lots.status=='COMPLETE'].copy();completed['week_start']=pd.to_datetime(completed.end_time).dt.to_period('W-SUN').dt.start_time.dt.strftime('%Y-%m-%d')
            for i,row in weekly.iterrows():
                cohort=completed[completed.week_start==row.week_start];full=cohort[cohort.program!='SHORT']
                weekly.loc[i,'avg_cycle_time_days']=full.cycle_time_days.mean()
                weekly.loc[i,'line_yield_pct']=100*cohort.wafers_out.sum()/cohort.wafers_in.sum() if len(cohort) else float('nan')
        equipment=[]
        for week in range(self.weeks):
            lo=self.warm+week*168;hi=lo+168;date=self.iso(lo)[:10]
            for group,initial in C.INSTALLED_TOOLS[self.site].items():
                if initial==0:continue
                capacity=initial*168+sum(max(0.,hi-max(lo,when))*n for g,when,n in self.arrivals if g==group)
                s=self.toolacc[(week,group)];nonprod=s['engineering']+s['sched_down']+s['unsched_down']
                productive=s['productive'];standby=max(0.,capacity-nonprod-productive)
                equipment.append(dict(site=self.site,seed=self.seed,case=self.case,week_start=date,tool_group=group,tools=capacity/168,
                    calendar_h=capacity,productive_h=productive,standby_h=standby,engineering_h=s['engineering'],sched_down_h=s['sched_down'],unsched_down_h=s['unsched_down'],
                    productive_pct=100*productive/capacity,standby_pct=100*standby/capacity,engineering_pct=100*s['engineering']/capacity,
                    sched_down_pct=100*s['sched_down']/capacity,unsched_down_pct=100*s['unsched_down']/capacity,
                    loading_pct=100*productive/(capacity-nonprod) if capacity>nonprod else float('nan'),
                    avg_queue_h=s['queue_h']/s['visits'] if s['visits'] else 0.,avg_process_h=s['process_h']/s['visits'] if s['visits'] else 0.,
                    lot_steps=int(s['visits']),moves=0,batch_count=s['batch_count'],batch_lots=s['batch_lots']))
        return dict(lots=lots,weekly=weekly,program_weekly=pd.DataFrame(self.program_weekly),equipment=pd.DataFrame(equipment),
            initial=pd.DataFrame(self.initial),batches=pd.DataFrame(self.batches),incidents=pd.DataFrame(self.incident_log))
