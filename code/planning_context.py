"""Synthetic planning context: previous plan version, program priorities and milestones.

These are small, explicit planning-metadata tables. They do not change the simulated
history or the 2027 requested plan; they add the context a capacity planner reads
alongside it. Run `python code/planning_context.py` to rewrite them.
"""
import sqlite3

import pandas as pd

import fab_config as C
import fab_data as fd

CURRENT_VERSION = '2026-10 plan (current)'
PREVIOUS_VERSION = '2026-07 plan'
VERSIONS = [PREVIOUS_VERSION, CURRENT_VERSION]
VERSION_SHORT = {PREVIOUS_VERSION: 'July plan', CURRENT_VERSION: 'October plan'}

PROGRAMS = [
    dict(program='DRAM_B', priority=1, priority_label='Ramp', milestone='Node B yield-learning gate',
         milestone_date='2027-06-30', protected=False, reason='Primary DRAM node ramp; requests above plan are reviewed against EUV.'),
    dict(program='HBM_TV', priority=1, priority_label='Qualification', milestone='HBM test-vehicle qualification lots out',
         milestone_date='2027-03-26', protected=True, reason='Customer-facing qualification date; starts and dispatch are protected.'),
    dict(program='DRAM_A', priority=2, priority_label='Sustaining', milestone='Node A process-transfer readiness at Fab B',
         milestone_date='2027-09-30', protected=False, reason='Dual-qualified; can move between fabs.'),
    dict(program='NAND_C', priority=2, priority_label='Module learning', milestone='Gen C stack-etch module freeze',
         milestone_date='2027-08-13', protected=False, reason='Dual-qualified; long etch and deposition loops.'),
    dict(program='SHORT', priority=3, priority_label='Experiments', milestone='None (rolling short loops)',
         milestone_date='', protected=False, reason='Deferrable experiments; first to give way.'),
]


def plan_versions(data):
    current = data['plan_2027'].assign(version=CURRENT_VERSION)
    rows = [dict(site=s, program=p, quarter=q, requested_wspw=float(v[i]), version=PREVIOUS_VERSION)
            for s, programs in C.JULY_REQUESTED_WSPW_2027.items() for p, v in programs.items() for i, q in enumerate(C.PLAN_QUARTERS)]
    return pd.concat([pd.DataFrame(rows), current], ignore_index=True)[['version', 'site', 'program', 'quarter', 'requested_wspw']]


def plan_2027():
    return pd.DataFrame([dict(site=s, program=p, quarter=q, requested_wspw=v[i])
                         for s, programs in C.REQUESTED_WSPW_2027.items() for p, v in programs.items()
                         for i, q in enumerate(C.PLAN_QUARTERS)])


def write():
    import fab_demo as D
    data = fd.load()
    data['plan_2027'] = plan_2027()
    tables = {'plan_2027': data['plan_2027'], 'demand_bridge_2026q4': D.demand_bridge(data['wafer_start_history']),
              'plan_versions_2027': plan_versions(data), 'program_planning': pd.DataFrame(PROGRAMS)}
    with sqlite3.connect(fd.DB_PATH) as db:
        for name, frame in tables.items():
            frame.to_csv(fd.TABLES / f'{name}.csv', index=False)
            frame.to_sql(name, db, if_exists='replace', index=False)
    return tables


if __name__ == '__main__':
    for name, frame in write().items():
        print(name, len(frame), 'rows')


def versions_frame(data):
    """2027 plan versions (July, October) with requested and expected starts."""
    import fab_demo as D
    import forecast_model as fm
    v = data['plan_versions_2027'].copy()
    rate = fm.fit(data['wafer_start_history']).set_index('program').realization_rate
    v['realization_rate'] = v.program.map(rate)
    v['expected_wspw'] = v.requested_wspw * v.realization_rate
    v['family'] = v.program.map(D.FAMILIES)
    v['technology_family'] = v.program.map(D.TECHNOLOGY)
    return v
