"""Explicit assumptions for the synthetic fab-flow study; no real fab calibration."""
from copy import deepcopy
import fab_config as C

PILOT='DRAM_P'
PILOT_NAME='DRAM EUV pilot'
PROGRAMS=deepcopy(C.PROGRAMS)
PROGRAMS[PILOT]={'name':PILOT_NAME,'mask_layers':44,'time_mult':{'ETCH':1.1},
    'passes':{'EUV':8,'DUV':36,'ETCH':58,'DEP':66,'PVD':10,'CMP':18,'WET':96,'FURN':14,'IMPL':20,'METRO':76}}
NAMES={p:v['name'] for p,v in PROGRAMS.items()}
TECHNOLOGY={p:('DRAM' if p.startswith('DRAM') or p=='HBM_TV' else 'NAND' if p=='NAND_C' else 'Experiments') for p in PROGRAMS}
FAMILIES={p:('HBM' if p=='HBM_TV' else TECHNOLOGY[p]) for p in PROGRAMS}
CASES={'baseline':'Pre-pilot baseline','introduction':'Pilot introduction','slower':'Slower pilot ramp'}
HISTORY_START='2025-09-29'
HISTORY_WEEKS=52
STUDY_START='2027-04-05'
STUDY_WEEKS=16
LAUNCH_DATE='2027-05-17'  # seventh reporting week, after the same simulated prefix
SEEDS=[7,17,27,37,47]
TARGET_DAYS={'DRAM_A':38,'DRAM_B':45,'NAND_C':42,'HBM_TV':48,'SHORT':12,PILOT:48}

def pilot_rate(date,case):
    import pandas as pd
    elapsed=(pd.Timestamp(date)-pd.Timestamp(LAUNCH_DATE)).days
    if case=='baseline' or elapsed<0:return 0.
    if case=='introduction':return 100. if elapsed<14 else 200.
    if case=='slower':return 50. if elapsed<28 else 75.
    raise ValueError('Unknown pilot case.')

def qualified(site,program):return site=='FAB_A' if program==PILOT else program in C.QUALIFIED[site]

def route_steps():
    """Declared synthetic layer modules, with exact configured visit counts.

    This represents re-entry and workload, not a validated device process recipe.
    Lithography is one visit per mask layer. Other visits are spread across the
    layers in a declared clean/deposit/anneal/litho/inspect/etch/finish module.
    """
    rows=[]
    for program,definition in PROGRAMS.items():
        layers=definition['mask_layers'];passes=definition['passes'];step=0
        for layer in range(layers):
            counts={g:((layer+1)*n//layers-layer*n//layers) for g,n in passes.items() if g not in ('EUV','DUV')}
            litho='EUV' if ((layer+1)*passes['EUV']//layers-layer*passes['EUV']//layers) else 'DUV'
            # One pre-clean, optional post-etch cleans; all visit totals retained.
            operations=['WET']*min(1,counts['WET'])+['DEP']*counts['DEP']+['FURN']*counts['FURN']+[litho]
            operations+=['METRO']*counts['METRO']+['ETCH']*counts['ETCH']+['WET']*max(0,counts['WET']-1)
            operations+=['IMPL']*counts['IMPL']+['CMP']*counts['CMP']+['PVD']*counts['PVD']
            for group in operations:
                step+=1;tool=C.TOOL_GROUPS[group]
                # Compatible furnace recipe groups are explicit. Pilot needs its own recipe.
                recipe=(f'PILOT_ANNEAL_{layer%2}' if program==PILOT else f'QUAL_ANNEAL_{layer%2}') if group=='FURN' else f'{program}_{group}'
                rows.append(dict(program=program,program_name=definition['name'],route_version='SYN-V6-1',step_no=step,
                    layer=layer+1,tool_group=group,recipe_id=recipe,
                    min_per_wafer=tool['min_per_wafer']*definition['time_mult'].get(group,1.),
                    batch_size_lots=3 if group=='FURN' else 1,batch_run_hours=6. if group=='FURN' else 0.,
                    sample_rate=tool.get('sample_rate',1.),qualified_fabs='FAB_A' if program==PILOT else '|'.join(s for s in C.SITES if qualified(s,program))))
    import pandas as pd
    return pd.DataFrame(rows)

def hours_per_wafer(program,group):
    spec=PROGRAMS[program];tool=C.TOOL_GROUPS[group]
    return spec['passes'][group]*tool['min_per_wafer']*spec['time_mult'].get(group,1.)*tool.get('sample_rate',1.)/60
