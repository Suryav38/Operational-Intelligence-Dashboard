"""Simulate 2026Q4 to date (28 Sep - 25 Oct 2026) as a continuation of the historical fab model.

The historical simulation (seed 42, both fabs, from 29 Sep 2025) is rerun for 56 instead of 52
weeks. Lot releases, process times and tool availability use per-lot / per-tool random streams,
so the first 52 weeks reproduce the stored history exactly (checked below); only the four new
weeks are written, to separate *_q4 tables. History-based tables are not touched.

    python code/simulate_q4_to_date.py
"""
import inspect
import sqlite3

import pandas as pd

import fab_config as C
import fab_data as fd
import fab_demo as D
import generate_fab_demo as G
from fab_flow import FabFlow

Q4_START = '2026-09-28'
WEEKS = D.HISTORY_WEEKS + 4


def incidents():
    src = inspect.getsource(G.history)
    return eval(src[src.index('incidents=[') + len('incidents='):src.index(']\n') + 1])


def main():
    data = fd.load()
    weekly, equipment, flow, lots = [], [], [], []
    for site in C.SITES:
        out = FabFlow(site, D.HISTORY_START, WEEKS, seed=42, historical=True, incidents=incidents()).run().results()
        old = data['sim_fab_weekly'].query('site == @site').set_index('week_start')[['wafer_starts', 'wafer_outs', 'wip_wafers']]
        new = out['weekly'].set_index('week_start').loc[old.index, ['wafer_starts', 'wafer_outs', 'wip_wafers']]
        if not new.equals(old.astype(new.dtypes.to_dict())):
            raise SystemExit(f'{site}: extended run does not reproduce stored history; not writing Q4 tables.')
        weekly.append(out['weekly'][out['weekly'].week_start >= Q4_START])
        equipment.append(out['equipment'][out['equipment'].week_start >= Q4_START])
        flow.append(out['program_weekly'][out['program_weekly'].week_start >= Q4_START])
        l = out['lots']
        lots.append(l[(pd.to_datetime(l.end_time, errors='coerce') >= Q4_START) | l.status.eq('ACTIVE') | l.end_time.isna()])
        print(site, 'history reproduced; Q4 weeks:', sorted(weekly[-1].week_start.unique()))
    tables = {'sim_fab_weekly_q4': pd.concat(weekly), 'sim_tool_group_weekly_q4': pd.concat(equipment),
              'weekly_program_flow_q4': pd.concat(flow), 'sim_lots_q4': pd.concat(lots)}
    with sqlite3.connect(fd.DB_PATH) as db:
        for name, frame in tables.items():
            frame.to_csv(fd.TABLES / f'{name}.csv', index=False)
            frame.to_sql(name, db, if_exists='replace', index=False)
            print(name, len(frame))


if __name__ == '__main__':
    main()
