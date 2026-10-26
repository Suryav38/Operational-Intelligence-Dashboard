"""Write the 2026Q4-to-date tool logbook (synthetic) from the simulated tool-group hours.

Two outputs, both consistent with sim_tool_group_weekly_q4 to the hour:
  * tables/tool_events_q4.csv (+ SQLite): one row per unscheduled-down event, tool ID, cause and hours.
    For every fab, week and tool group the event hours sum to the simulated unscheduled-down hours.
  * documents/tool_logs/2026Q4/<FAB>_<week>.txt: a weekly equipment review per fab, the notes
    the Fab performance assistant retrieves and cites.

Causes are drawn from a fixed catalog per tool group (seed 42). Run after simulate_q4_to_date.py:
    python code/make_tool_logs.py
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import fab_config as C
import fab_data as fd
from review_model import GROUP_LABELS, SITE_LABELS

SEED = 42
OUT = fd.DOCS / 'tool_logs' / '2026Q4'
TYPICAL_EVENT_H = dict(EUV=9, DUV=7, ETCH=14, DEP=14, PVD=8, CMP=8, WET=8, FURN=10, IMPL=8, METRO=5)
MAX_EVENTS = 14
MAX_EVENT_H = 60      # one tool cannot be down longer than this inside one week

CAUSES = {
    'EUV': [('Source power drop: droplet generator instability', 'Droplet generator replaced; dose re-baselined'),
            ('Collector mirror contamination', 'Collector swapped; imaging requalified'),
            ('Reticle stage position error', 'Stage recalibrated; overlay check passed'),
            ('Dose uniformity drift', 'Illuminator recalibrated')],
    'DUV': [('Lens heating drift beyond spec', 'Lens model re-baselined'),
            ('Wafer stage interferometer fault', 'Interferometer board replaced'),
            ('Track coater nozzle clog', 'Nozzle cleaned; coat thickness verified'),
            ('Reticle handler jam', 'Handler gripper realigned')],
    'ETCH': [('Chamber particle excursion', 'Wet clean and seasoning wafers run'),
             ('RF generator fault', 'RF generator swapped'),
             ('Electrostatic chuck helium leak', 'Chuck O-rings replaced; leak check passed'),
             ('Endpoint detector failure', 'Optical endpoint sensor replaced')],
    'DEP': [('Showerhead clogging: PM pulled in', 'Showerhead replaced; film uniformity requalified'),
            ('ALD precursor valve fault', 'Valve replaced; delivery line purged'),
            ('Particle excursion after chamber clean', 'Additional seasoning; particle monitor passed'),
            ('Heater zone failure', 'Heater replaced; temperature profile requalified')],
    'PVD': [('Target reached end of life early', 'Target swapped; burn-in completed'),
            ('Magnetron rotation fault', 'Magnetron drive repaired'),
            ('Shield flaking particles', 'Shield kit replaced')],
    'CMP': [('Pad conditioner disk wear', 'Disk replaced; removal rate requalified'),
            ('Slurry delivery flow alarm', 'Slurry pump serviced'),
            ('Wafer breakage on platen', 'Platen cleaned; head inspected')],
    'WET': [('Bath chemistry out of spec', 'Bath dumped and refilled; chemistry verified'),
            ('Robot handoff error', 'Robot taught; handoff verified'),
            ('Megasonic transducer fault', 'Transducer replaced')],
    'FURN': [('Thermocouple drift', 'Thermocouple replaced; profile requalified'),
             ('Boat elevator fault', 'Elevator drive serviced'),
             ('Mass-flow controller failure', 'MFC replaced; leak check passed')],
    'IMPL': [('Ion source filament failure', 'Source rebuilt; beam tuned'),
             ('Beamline arcing', 'Beamline cleaned; arc count normal'),
             ('Wafer handler vacuum fault', 'Vacuum pump serviced')],
    'METRO': [('Stage calibration out of tolerance', 'Stage recalibrated'),
              ('Light source end of life', 'Light source replaced')],
}


def events(tools: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for r in tools.sort_values(['site', 'week_start', 'tool_group']).itertuples():
        hours = float(r.unsched_down_h)
        if hours < .5:
            continue
        n = int(min(MAX_EVENTS, max(1, round(hours / TYPICAL_EVENT_H[r.tool_group]), int(np.ceil(hours / MAX_EVENT_H)))))
        for _ in range(200):
            split = rng.dirichlet(np.full(n, 3.0)) * hours
            if split.max() <= MAX_EVENT_H:
                break
        else:
            split = np.full(n, hours / n)
        split[-1] = hours - split[:-1].sum()                       # hours sum exactly to the simulated total
        fleet = max(1, int(round(r.tools)))
        ids = rng.choice(np.arange(1, fleet + 1), size=n, replace=n > fleet)
        offsets = [rng.uniform(0, max(.01, 7 - h / 24)) for h in split]   # each event ends inside the week
        for k in range(n):
            cause, action = CAUSES[r.tool_group][rng.integers(len(CAUSES[r.tool_group]))]
            start = pd.Timestamp(r.week_start) + pd.Timedelta(days=float(offsets[k]))
            rows.append(dict(event_id=f'EV-{r.site[-1]}-{len(rows)+1:04d}', site=r.site, week_start=r.week_start,
                             tool_group=r.tool_group, tool_id=f'{r.tool_group}-{r.site[-1]}{int(ids[k]):02d}',
                             start_time=start.strftime('%Y-%m-%d %H:%M'), down_hours=round(float(split[k]), 2),
                             cause=cause, action=action, state='unsched_down'))
    return pd.DataFrame(rows)


def week_note(site, week, fab, tools, ev):
    t = tools[(tools.site == site) & (tools.week_start == week)].copy()
    t['avail_pct'] = 100 * (t.calendar_h - t.engineering_h - t.sched_down_h - t.unsched_down_h) / t.calendar_h
    t['assumed_pct'] = t.tool_group.map(lambda g: 100 * C.production_availability(g))
    t['unsched_pct'] = 100 * t.unsched_down_h / t.calendar_h
    f = fab[(fab.site == site) & (fab.week_start == week)].iloc[0]
    end = (pd.Timestamp(week) + pd.Timedelta(days=6)).strftime('%d %b %Y')
    lines = [f'Weekly equipment review · {SITE_LABELS[site]} · Q4 2026 · week of {pd.Timestamp(week):%d %b} – {end} · SYNTHETIC',
             f'Source: simulated tool states (seed {SEED}); event detail in table tool_events_q4.', '',
             f'Line: {int(f.wafer_starts):,} wafers started, {int(f.wafer_outs):,} completed, WIP {int(f.wip_wafers):,} wafers, '
             f'average cycle time {f.avg_cycle_time_days:.1f} days.', '',
             'Unscheduled downtime by tool group this week (largest first):']
    for r in t[t.unsched_down_h >= .5].sort_values('unsched_down_h', ascending=False).itertuples():
        e = ev[(ev.site == site) & (ev.week_start == week) & (ev.tool_group == r.tool_group)].sort_values('down_hours', ascending=False)
        lines.append(f'- {GROUP_LABELS[r.tool_group]} ({r.tool_group}, {int(round(r.tools))} tools): {r.unsched_down_h:.1f} h unscheduled down '
                     f'({r.unsched_pct:.1f}% of calendar), {len(e)} event{"s" if len(e) != 1 else ""}. Availability {r.avail_pct:.1f}% vs {r.assumed_pct:.1f}% assumed; '
                     f'loading {r.loading_pct:.1f}%; queue {r.avg_queue_h:.2f} h per lot step.')
        show = 3
        for x in e.head(show).itertuples():
            lines.append(f'    {x.tool_id} {x.start_time}: {x.cause}, {x.down_hours:.1f} h. Action: {x.action}.')
        if len(e) > show:
            lines.append(f'    Plus {len(e) - show} smaller event{"s" if len(e) - show != 1 else ""}, {e.down_hours.iloc[show:].sum():.1f} h in total.')
        top = e.groupby('cause').down_hours.sum().sort_values(ascending=False)
        lines.append(f'    Main cause by hours: {top.index[0]} ({top.iloc[0]:.1f} h).')
    below = t[t.avail_pct < t.assumed_pct - 2].sort_values('avail_pct')
    lines += ['', 'Availability below the planning assumption by more than 2 points: ' +
              ('; '.join(f'{GROUP_LABELS[r.tool_group]} {r.avail_pct:.1f}% vs {r.assumed_pct:.1f}%' for r in below.itertuples()) if len(below) else 'none.')]
    hot = t[t.loading_pct >= 85].sort_values('loading_pct', ascending=False)
    lines += ['Loading at or above the 85% planning limit: ' +
              ('; '.join(f'{GROUP_LABELS[r.tool_group]} {r.loading_pct:.1f}% (queue {r.avg_queue_h:.2f} h)' for r in hot.itertuples()) if len(hot) else 'none.')]
    return '\n'.join(lines) + '\n'


def main():
    data = fd.load()
    tools, fab = data['sim_tool_group_weekly_q4'], data['sim_fab_weekly_q4']
    ev = events(tools)
    check = ev.groupby(['site', 'week_start', 'tool_group']).down_hours.sum()
    sim = tools.set_index(['site', 'week_start', 'tool_group']).unsched_down_h
    assert (check - sim.reindex(check.index)).abs().max() < .05, 'event hours must match the simulation'
    ev.to_csv(fd.TABLES / 'tool_events_q4.csv', index=False)
    with sqlite3.connect(fd.DB_PATH) as db:
        ev.to_sql('tool_events_q4', db, if_exists='replace', index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob('*.txt'):
        old.unlink()
    for site in C.SITES:
        for week in sorted(tools.week_start.unique()):
            (OUT / f'{site}_{week}.txt').write_text(week_note(site, week, fab, tools, ev))
    print(f'{len(ev)} events, {ev.down_hours.sum():,.0f} h; {len(list(OUT.glob("*.txt")))} weekly notes written to {OUT}')


if __name__ == '__main__':
    main()
