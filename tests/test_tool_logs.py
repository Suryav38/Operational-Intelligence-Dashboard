"""The Q4 tool logbook must agree with the simulated tool-group hours it was written from."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import fab_data as fd
import planning_copilot as rag


def test_event_hours_match_simulated_downtime():
    data = fd.load()
    ev, sim = data['tool_events_q4'], data['sim_tool_group_weekly_q4']
    got = ev.groupby(['site', 'week_start', 'tool_group']).down_hours.sum()
    want = sim.set_index(['site', 'week_start', 'tool_group']).unsched_down_h.reindex(got.index)
    assert (got - want).abs().max() < .05
    assert ev.down_hours.max() <= 60
    per_tool_week = ev.groupby(['site', 'week_start', 'tool_id']).down_hours.sum()
    assert per_tool_week.max() <= 168


def test_weekly_reviews_exist_and_are_retrieved():
    notes = sorted((fd.DOCS / 'tool_logs' / '2026Q4').glob('*.txt'))
    assert len(notes) == 2 * fd.load()['sim_fab_weekly_q4'].week_start.nunique()
    hits = rag.DocumentIndex(fd.DOCS).search('What problems did the deposition tools at Fab A have in Q4 so far?', 4)
    assert hits[0]['source'].startswith('tool_logs/2026Q4/FAB_A')
