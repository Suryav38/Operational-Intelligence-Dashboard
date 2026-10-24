"""Machine/period OEE from explicit single-recipe time and first-pass counts.

One non-overlapping machine-period record has one ideal cycle standard. No
machine OEE is inferred from fab yield or aggregate equipment time states.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COLUMNS=['machine_id','period','recipe_id','planned_minutes','run_minutes','ideal_cycle_seconds','total_count','good_count']
NUMERIC=COLUMNS[3:]

def example():
    return pd.DataFrame([dict(machine_id='EXAMPLE-01',period='Illustrative shift',recipe_id='Example single recipe',
        planned_minutes=480,run_minutes=360,ideal_cycle_seconds=60,total_count=300,good_count=294)])

def calculate(records):
    missing=[c for c in COLUMNS if c not in records]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    if records.empty:raise ValueError('Supply at least one machine-period record.')
    if len(records)>10000:raise ValueError('Limit the file to 10,000 machine-period records.')
    frame=records[COLUMNS].copy()
    for c in COLUMNS[:3]:
        if frame[c].isna().any() or frame[c].astype(str).str.strip().eq('').any():raise ValueError(c+' must be populated.')
    if frame.duplicated(['machine_id','period']).any():
        raise ValueError('Use one single-recipe record per machine and period; split mixed recipes into separate non-overlapping periods.')
    for c in NUMERIC:
        frame[c]=pd.to_numeric(frame[c],errors='coerce')
        if not np.isfinite(frame[c]).all():raise ValueError(c+' must contain finite numeric values.')
    if (frame[NUMERIC]<0).any().any():raise ValueError('Times and counts cannot be negative.')
    if (frame.planned_minutes<=0).any() or (frame.ideal_cycle_seconds<=0).any():
        raise ValueError('Planned time and ideal cycle time must be positive.')
    if (frame.run_minutes>frame.planned_minutes).any():raise ValueError('Run time cannot exceed planned production time.')
    if (frame.good_count>frame.total_count).any():raise ValueError('Good count cannot exceed total count.')
    if ((frame[['total_count','good_count']]%1)!=0).any().any():raise ValueError('Unit counts must be whole numbers.')
    ideal_minutes=frame.ideal_cycle_seconds*frame.total_count/60
    if (ideal_minutes>frame.run_minutes+1e-8).any():
        raise ValueError('Performance exceeds 100%. Recheck the ideal cycle standard, counts and run time.')
    frame['availability_pct']=100*frame.run_minutes/frame.planned_minutes
    frame['performance_pct']=100*ideal_minutes/frame.run_minutes.replace(0,np.nan)
    frame['quality_pct']=100*frame.good_count/frame.total_count.replace(0,np.nan)
    frame['oee_pct']=100*(frame.good_count*frame.ideal_cycle_seconds/60)/frame.planned_minutes
    return frame
