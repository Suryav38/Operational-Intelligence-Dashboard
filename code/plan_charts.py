"""Charts for the capital-planning views. One palette, one type scale, no chart junk."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import capital_model as K
import executive_charts as ec
import fab_config as C
import fab_demo as D
import review_model as rm

INK, MUTED, GRID = ec.INK, ec.MUTED, ec.GRID
ACCENT, LIGHT, WARN, BAD = ec.BLUE, '#9db4c9', ec.AMBER, ec.RED
HISTORY = '#8a9199'
PROGRAM_COLOR = dict(zip(C.PROGRAM_ORDER + [D.PILOT], ec.PROGRAM_COLORS))


def _limit_line(fig, y, text, color=WARN, dash='dash'):
    fig.add_hline(y=y, line=dict(color=color, width=1.2, dash=dash),
                  annotation=dict(text=text, font=dict(size=10, color=color), xanchor='right', x=1, yanchor='bottom'))


# ------------------------------------------------------------------ capital plan
def drivers(state, site, group, ceiling):
    """Required hours by program (stacked) against available hours and the planning limit."""
    i_site = K.SITES.index(site)
    fig = go.Figure()
    for p in K.PROGRAMS:
        hours = [state['wspw'][site].get(p, [0.] * 4)[q] * D.hours_per_wafer(p, group) for q in range(4)]
        if not any(hours):
            continue
        fig.add_bar(x=K.Q, y=hours, name=D.NAMES[p], marker_color=PROGRAM_COLOR[p],
                    hovertemplate='%{x} · %{y:,.0f} h/wk<extra>' + D.NAMES[p] + '</extra>')
    tools = K.timed_tools()[i_site, K.GROUPS.index(group)]
    per = np.array([168 * state['avail'][site][group][q] for q in range(4)])
    available = tools * per
    # One short horizontal tick per quarter: capacity steps when tools arrive.
    fig.add_scatter(x=K.Q, y=available, name='Available hours', mode='markers',
                    marker=dict(symbol='line-ew', size=70, line=dict(color=INK, width=2.5)),
                    customdata=tools, hovertemplate='%{x} · %{y:,.0f} h/wk from %{customdata:.0f} tools<extra></extra>')
    fig.add_scatter(x=K.Q, y=available * ceiling, name=f'{ceiling*100:.0f}% planning limit', mode='markers',
                    marker=dict(symbol='line-ew', size=70, line=dict(color=WARN, width=2)),
                    hovertemplate='%{x} · limit %{y:,.0f} h/wk<extra></extra>')
    ec.style(fig, 330)
    fig.update_layout(barmode='stack')
    fig.update_yaxes(title='Tool hours / week', rangemode='tozero')
    fig.update_xaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ demand
def demand_trend(totals, history, q4=None):
    """Actual and requested history, the 2026Q4 trend forecast, then the 2027 request and expected starts."""
    fig = go.Figure()
    h = history
    hq = list(h.index)
    req_x, req_y = hq + (['2026Q4'] if q4 else []), list(h.requested_wspw) + ([q4['requested_wspw']] if q4 else [])
    fig.add_scatter(x=req_x, y=req_y, name='Requested (history + Q4)', mode='lines',
                    line=dict(color=HISTORY, width=1.5, dash='dot'), hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
    fig.add_scatter(x=hq, y=h.actual_wspw, name='Actual starts', mode='lines+markers',
                    line=dict(color=INK, width=2), marker=dict(size=5), hovertemplate='%{x} · %{y:,.0f} actual<extra></extra>')
    if q4:
        fig.add_scatter(x=[hq[-1], '2026Q4'], y=[h.actual_wspw.iloc[-1], q4['forecast_wspw']], name='Q4 forecast (no actuals yet)',
                        mode='lines+markers', line=dict(color=INK, width=1.5, dash='dash'), marker=dict(size=7, symbol='circle-open'),
                        hovertemplate='%{x} · %{y:,.0f} forecast<extra></extra>')
    t = totals
    fig.add_scatter(x=t.index, y=t.requested_wspw, name='2027 requested', mode='lines+markers',
                    line=dict(color=ACCENT, width=2.5), marker=dict(size=6), hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
    fig.add_scatter(x=t.index, y=t.expected_wspw, name='2027 expected', mode='lines+markers',
                    line=dict(color=ACCENT, width=2, dash='dash'), marker=dict(size=6, symbol='circle-open'),
                    hovertemplate='%{x} · %{y:,.0f} expected<extra></extra>')
    order = hq + (['2026Q4'] if q4 else []) + list(t.index)
    first_plan = order.index(t.index[0])
    fig.add_vrect(x0=first_plan - .5, x1=len(order) - .5, fillcolor='#f3f5f8', line_width=0, layer='below')
    fig.add_annotation(x=first_plan - .45, y=1, yref='paper', text='2027 plan', showarrow=False, xanchor='left',
                       font=dict(size=10, color=MUTED))
    ec.style(fig, 340)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=order, showgrid=False)
    fig.update_yaxes(title='Wafer starts / week', rangemode='tozero')
    return fig


def demand_mix(frame, split, measure):
    fig = go.Figure()
    if split == 'Program':
        for p in C.PROGRAM_ORDER + [D.PILOT]:
            rows = frame[frame.program == p].groupby('quarter')[measure].sum(min_count=1).reindex(C.PLAN_QUARTERS)
            if rows.fillna(0).sum() == 0:
                continue
            fig.add_bar(x=rows.index, y=rows, name=D.NAMES[p], marker_color=PROGRAM_COLOR[p],
                        hovertemplate='%{x} · %{y:,.0f} WSPW<extra>' + D.NAMES[p] + '</extra>')
    else:
        for s in C.SITES:
            rows = frame[frame.site == s].groupby('quarter')[measure].sum(min_count=1).reindex(C.PLAN_QUARTERS)
            fig.add_bar(x=rows.index, y=rows, name=rm.SITE_LABELS[s], marker_color=ec.SITE_COLORS[s],
                        hovertemplate='%{x} · %{y:,.0f} WSPW<extra>' + rm.SITE_LABELS[s] + '</extra>')
    ec.style(fig, 330)
    fig.update_layout(barmode='stack')
    fig.update_yaxes(title='Wafer starts / week')
    fig.update_xaxes(showgrid=False)
    return fig


def realization(model):
    m = model.sort_values('realization_rate')
    colors = [ACCENT if r < 1 else LIGHT for r in m.realization_rate]
    fig = go.Figure(go.Bar(y=m.program_name, x=100 * m.realization_rate, orientation='h', marker_color=colors,
                           text=[f'{100*r:.0f}%' for r in m.realization_rate], textposition='outside',
                           textfont=dict(size=11, color=INK), hovertemplate='%{y}: runs %{x:.0f}% of request<extra></extra>'))
    fig.add_vline(x=100, line=dict(color=MUTED, width=1, dash='dot'))
    ec.style(fig, 260)
    fig.update_xaxes(title='Actual starts as % of request', range=[0, 130], showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ levers
def waterfall(steps, final='Tools to buy'):
    values = steps.net_new.tolist()
    measure = ['absolute'] + ['relative'] * (len(values) - 1) + ['total']
    y = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))] + [values[-1]]
    x = [str(v).replace(' ', '<br>') for v in steps.step] + [final.replace(' ', '<br>')]
    text = [str(values[0])] + [f'{v:+d}' for v in y[1:-1]] + [str(values[-1])]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=measure, text=text, textposition='outside',
                                 textfont=dict(size=13, color=INK),
                                 connector=dict(line=dict(color='#cfd4d9', width=1)),
                                 increasing=dict(marker=dict(color=BAD)), decreasing=dict(marker=dict(color='#7fa1c3')),
                                 totals=dict(marker=dict(color=ACCENT)), hovertemplate='%{x}: %{text}<extra></extra>'))
    fig.data[0].update(base=None)
    ec.style(fig, 330)
    fig.update_layout(showlegend=False, margin=dict(t=30))
    fig.update_yaxes(title='Net-new tools (count)', rangemode='tozero', range=[0, max(values) * 1.18])
    fig.update_xaxes(showgrid=False, tickangle=0)
    return fig


def operating_curve(curve, ceiling, higher):
    u, qh = curve['u_pct'], curve['queue_h']
    fig = go.Figure()
    fig.add_scatter(x=u, y=qh, mode='lines', line=dict(color=ACCENT, width=2.5), name='Queue time per visit',
                    hovertemplate='%{x:.0f}% loaded · %{y:.2f} h queue<extra></extra>')
    k, pt = curve['k'], curve['process_h']
    for level, color in ((ceiling, WARN), (higher, BAD)):
        y = k * level / (1 - level) * pt
        fig.add_scatter(x=[100 * level], y=[y], mode='markers+text', marker=dict(color=color, size=10),
                        text=[f'{100*level:.0f}% → {y:.1f} h'], textposition='middle left', textfont=dict(size=11, color=color),
                        showlegend=False, hoverinfo='skip')
    x0, y0 = curve['sim_point']
    fig.add_scatter(x=[x0], y=[y0], mode='markers', marker=dict(color=INK, size=8, symbol='diamond'),
                    name='2026 simulated point', hovertemplate='2026: %{x:.1f}% loaded · %{y:.2f} h<extra></extra>')
    ec.style(fig, 300)
    fig.update_xaxes(title='Tool loading (%)', range=[30, 100])
    fig.update_yaxes(title='Queue hours per visit', rangemode='tozero', range=[0, float(np.nanmax(qh[u <= 97])) * 1.05])
    return fig


# ------------------------------------------------------------------ baseline
def model_check(frame):
    f = frame.copy()
    f['label'] = f.site.map(rm.SITE_LABELS) + ' · ' + f.tool_group.map(rm.GROUP_LABELS)
    f = f.sort_values('simulated_pct')
    fig = go.Figure()
    for r in f.itertuples():
        fig.add_shape(type='line', x0=r.model_pct, x1=r.simulated_pct, y0=r.label, y1=r.label,
                      line=dict(color='#cfd4d9', width=3))
    fig.add_scatter(x=f.model_pct, y=f.label, mode='markers', name='Capacity model', marker=dict(color=ACCENT, size=9),
                    hovertemplate='%{y}<br>Model %{x:.1f}%<extra></extra>')
    fig.add_scatter(x=f.simulated_pct, y=f.label, mode='markers', name='Lot simulation',
                    marker=dict(color=INK, size=9, symbol='diamond-open', line=dict(width=2)),
                    hovertemplate='%{y}<br>Simulated %{x:.1f}%<extra></extra>')
    ec.style(fig, max(280, 26 * len(f) + 70))
    fig.update_xaxes(title='Production loading, 2026Q3 (%)', range=[40, 100])
    fig.update_yaxes(showgrid=False)
    return fig


def attainment(frame):
    fig = go.Figure()
    fig.add_bar(x=frame.quarter, y=frame.requested_wspw, name='Requested', marker_color='#d9dde2',
                hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
    fig.add_bar(x=frame.quarter, y=frame.actual_wspw, name='Actual', marker_color=INK,
                hovertemplate='%{x} · %{y:,.0f} actual<extra></extra>')
    ec.style(fig, 300)
    fig.update_layout(barmode='overlay', bargap=.35)
    fig.data[1].update(width=.45)
    fig.update_yaxes(title='Wafer starts / week')
    fig.update_xaxes(showgrid=False)
    return fig


def cycle_trend(weekly, site):
    fig = go.Figure()
    sites = list(C.SITES) if site == 'Both' else [site]
    for s in sites:
        rows = weekly[weekly.site == s].sort_values('week_start')
        fig.add_scatter(x=rows.week_start, y=rows.avg_cycle_time_days, name=rm.SITE_LABELS[s], mode='lines',
                        line=dict(color=ec.SITE_COLORS[s], width=2), hovertemplate='%{x|%d %b %Y} · %{y:.1f} days<extra>' + rm.SITE_LABELS[s] + '</extra>')
    ec.style(fig, 260)
    fig.update_xaxes(type='date', tickformat='%b %Y', showgrid=False)
    fig.update_yaxes(title='Cycle time (days)', rangemode='tozero')
    return fig


# ------------------------------------------------------------------ demand context
def version_change(versions, previous, current):
    v = versions.groupby(['version', 'program']).requested_wspw.sum().unstack(0).fillna(0) / 4
    v['delta'] = v[current] - v[previous]
    v = v.sort_values('delta')
    colors = [ACCENT if d > 0 else (WARN if d < 0 else LIGHT) for d in v.delta]
    fig = go.Figure(go.Bar(y=[D.NAMES[p] for p in v.index], x=v.delta, orientation='h', marker_color=colors,
                           text=[f'{d:+.0f}' for d in v.delta], textposition='outside', textfont=dict(size=11, color=INK),
                           customdata=np.c_[v[previous], v[current]],
                           hovertemplate='%{y}<br>%{customdata[0]:,.0f} → %{customdata[1]:,.0f} WSPW (2027 avg)<extra></extra>'))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    ec.style(fig, 260)
    span = max(10, float(v.delta.abs().max()) * 1.35)
    fig.update_xaxes(title='Change in 2027 average starts / week', range=[-span, span], showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ fab performance (one month)
def month_flow(weekly, plan_wspw):
    labels = [pd.Timestamp(w).strftime('%d %b') for w in weekly.week_start]
    fig = go.Figure()
    fig.add_bar(x=labels, y=weekly.wafer_starts, name='Starts', marker_color=INK,
                hovertemplate='Week of %{x} · %{y:,.0f} started<extra></extra>')
    fig.add_bar(x=labels, y=weekly.wafer_outs, name='Completions', marker_color=LIGHT,
                hovertemplate='Week of %{x} · %{y:,.0f} completed<extra></extra>')
    fig.add_hline(y=plan_wspw, line=dict(color=WARN, width=1.5, dash='dash'),
                  annotation=dict(text=f'Plan {plan_wspw:,.0f}', font=dict(size=11, color=WARN), xanchor='right', x=1, yanchor='bottom'))
    ec.style(fig, 300)
    fig.update_layout(barmode='group', bargap=.35)
    fig.update_yaxes(title='Wafers / week', rangemode='tozero', range=[0, max(plan_wspw, weekly[['wafer_starts', 'wafer_outs']].max().max()) * 1.15])
    fig.update_xaxes(title='Week starting', showgrid=False)
    return fig


def month_wip_ct(weekly):
    from plotly.subplots import make_subplots
    labels = [pd.Timestamp(w).strftime('%d %b') for w in weekly.week_start]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=('WIP at week end (wafers)', 'Cycle time of lots completed (days)'))
    fig.add_scatter(x=labels, y=weekly.wip_wafers, mode='lines+markers+text', line=dict(color=INK, width=2), row=1, col=1,
                    text=[f'{v:,.0f}' for v in weekly.wip_wafers], textposition='top center', textfont=dict(size=11),
                    hovertemplate='%{x} · %{y:,.0f} wafers<extra></extra>', showlegend=False)
    fig.add_scatter(x=labels, y=weekly.cycle_time_days, mode='lines+markers+text', line=dict(color=ACCENT, width=2), row=2, col=1,
                    text=[f'{v:.1f}' for v in weekly.cycle_time_days], textposition='top center', textfont=dict(size=11),
                    hovertemplate='%{x} · %{y:.1f} days<extra></extra>', showlegend=False)
    ec.style(fig, 320)
    fig.update_layout(margin=dict(t=34))
    fig.update_annotations(font=dict(size=11, color=MUTED), x=0, xanchor='left')
    w, c = weekly.wip_wafers, weekly.cycle_time_days
    fig.update_yaxes(range=[w.min() * .9, w.max() * 1.08], row=1, col=1)
    fig.update_yaxes(range=[c.min() * .85, c.max() * 1.12], row=2, col=1)
    fig.update_xaxes(showgrid=False)
    return fig


def availability(tools, both):
    f = tools.copy()
    f['label'] = (f.site.map(rm.SITE_LABELS) + ' · ' if both else '') + f.tool_group.map(rm.GROUP_LABELS)
    f = f.sort_values('avail_gap_pts')
    fig = go.Figure()
    for r in f.itertuples():
        fig.add_shape(type='line', x0=r.assumed_avail_pct, x1=r.observed_avail_pct, y0=r.label, y1=r.label,
                      line=dict(color=BAD if r.avail_gap_pts < -2 else '#cfd4d9', width=3))
    fig.add_scatter(x=f.assumed_avail_pct, y=f.label, mode='markers', name='Planning assumption', marker=dict(color=LIGHT, size=9),
                    hovertemplate='%{y}<br>Assumed %{x:.1f}%<extra></extra>')
    fig.add_scatter(x=f.observed_avail_pct, y=f.label, mode='markers', name='Measured, Q4 to date',
                    marker=dict(color=INK, size=9, symbol='diamond'), hovertemplate='%{y}<br>Measured %{x:.1f}%<extra></extra>')
    ec.style(fig, max(280, 25 * len(f) + 80))
    fig.update_xaxes(title='Production availability (% of calendar time)', range=[60, 95])
    fig.update_yaxes(showgrid=False)
    return fig


def loading_queue(tools, limit, both):
    f = tools.copy()
    f['label'] = (f.site.map(rm.SITE_LABELS) + ' · ' if both else '') + f.tool_group.map(rm.GROUP_LABELS)
    f = f.sort_values('loading_pct')
    colors = [BAD if v > 100 else WARN if v >= limit * 100 else LIGHT for v in f.loading_pct]
    fig = go.Figure(go.Bar(y=f.label, x=f.loading_pct, orientation='h', marker_color=colors,
                           text=[f'{l:.0f}% · {q:.1f} h queue' for l, q in zip(f.loading_pct, f.queue_h)], textposition='outside',
                           textfont=dict(size=10, color=MUTED), hovertemplate='%{y}<br>%{x:.1f}% loaded<extra></extra>'))
    fig.add_vline(x=limit * 100, line=dict(color=WARN, width=1.2, dash='dash'))
    ec.style(fig, max(280, 25 * len(f) + 80))
    fig.update_xaxes(title='Production loading (%)', range=[0, 125], showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ wafer loading analyst
def analyst_weeks(result):
    req = result['request']
    site, group = req['site'], result['bottleneck']
    fig = go.Figure()

    def series(frame, name, color, dash, width=2):
        rows = frame[(frame.site == site) & (frame.tool_group == group)].sort_values('week_start')
        fig.add_scatter(x=pd.to_datetime(rows.week_start), y=rows.loading_pct, name=name, mode='lines',
                        line=dict(color=color, width=width, dash=dash, shape='hv'),
                        hovertemplate='Week of %{x|%d %b} · %{y:.1f}%<extra>' + name + '</extra>')
    series(result['base_weekly'], 'Baseline plan', HISTORY, 'solid')
    series(result['request_weekly'], 'Full request', BAD, 'solid')
    if result['recommended'] is not None:
        series(result['recommended']['weekly'], 'Recommended', ACCENT, 'solid', 3)
    w0 = pd.Timestamp(result['weeks'][0])
    w1 = pd.Timestamp(result['weeks'][-1]) + pd.Timedelta(days=7)
    fig.add_vrect(x0=w0, x1=w1, fillcolor='#f3f5f8', line_width=0, layer='below',
                  annotation=dict(text='Request window', font=dict(size=10, color=MUTED), yanchor='top', y=1))
    _limit_line(fig, result['limit'] * 100, f'{result["limit"]*100:.0f}% planning limit')
    _limit_line(fig, result['ceiling'] * 100, f'{result["ceiling"]*100:.0f}% acceptance ceiling', BAD, 'dot')
    ec.style(fig, 300)
    fig.update_xaxes(type='date', tickformat='%b', showgrid=False, range=['2027-01-01', '2027-12-31'])
    fig.update_yaxes(title=f'{rm.SITE_LABELS[site]} · {rm.GROUP_LABELS[group]} loading (%)', range=[40, 130])
    return fig


def freed_added(frame, both):
    f = frame.copy()
    f['label'] = (f.site.map(rm.SITE_LABELS) + ' · ' if both else '') + f.tool_group.map(rm.GROUP_LABELS)
    f = f.sort_values('added_h_wk')
    fig = go.Figure()
    fig.add_bar(y=f.label, x=f.added_h_wk, orientation='h', name='Added by growing programs', marker_color=BAD,
                hovertemplate='%{y}<br>+%{x:,.0f} h/wk<extra></extra>')
    fig.add_bar(y=f.label, x=-f.freed_h_wk, orientation='h', name='Freed by declining programs', marker_color='#7fa1c3',
                customdata=f.offset_pct, hovertemplate='%{y}<br>%{x:,.0f} h/wk · offsets %{customdata:.0f}% of the increase<extra></extra>')
    ec.style(fig, max(300, 24 * len(f) + 90))
    fig.update_layout(barmode='relative')
    fig.update_xaxes(title='Tool-hours / week (freed ← → added)', zeroline=True, zerolinecolor=MUTED)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ demand outlook (versions)
import planning_context as PCX
VERSION_COLOR = {PCX.PREVIOUS_VERSION: '#9aa1a8', PCX.CURRENT_VERSION: ACCENT}
VERSION_DASH = {PCX.PREVIOUS_VERSION: 'solid', PCX.CURRENT_VERSION: 'solid'}


def version_trend(history, q4, vf, col, shown):
    """All lines solid: actual runs straight into the Q4 forecast point, plan versions continue from it."""
    fig = go.Figure()
    hq = list(history.index)
    fig.add_scatter(x=hq + ['2026Q4'], y=list(history.requested_wspw) + [q4['requested_wspw']], name='Requested',
                    mode='lines', line=dict(color='#c3c8ce', width=2), hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
    fig.add_scatter(x=hq + ['2026Q4'], y=list(history.actual_wspw) + [q4['forecast_wspw']], name='Actual starts',
                    mode='lines+markers', line=dict(color=INK, width=2.2), marker=dict(size=4),
                    hovertemplate='%{x} · %{y:,.0f} starts<extra></extra>')
    fig.add_scatter(x=['2026Q4'], y=[q4['forecast_wspw']], mode='markers+text', marker=dict(size=9, color=INK),
                    text=['Q4 fcst'], textposition='bottom center', textfont=dict(size=10, color=MUTED), showlegend=False,
                    hovertemplate='2026Q4 forecast · %{y:,.0f}<extra></extra>')
    anchor = q4['requested_wspw'] if col == 'requested_wspw' else q4['forecast_wspw']
    label = 'request' if col == 'requested_wspw' else 'expected'
    for v in PCX.VERSIONS:
        if v not in shown or (col == 'expected_wspw' and v == PCX.CURRENT_VERSION):
            continue
        t = vf[vf.version == v].groupby('quarter')[col].sum().reindex(C.PLAN_QUARTERS)
        fig.add_scatter(x=['2026Q4'] + list(t.index), y=[anchor] + list(t), name=f'{PCX.VERSION_SHORT[v]} {label}', mode='lines+markers',
                        line=dict(color=VERSION_COLOR[v], width=2.6 if v == PCX.CURRENT_VERSION else 2),
                        marker=dict(size=6), hovertemplate='%{x} · %{y:,.0f}<extra>' + PCX.VERSION_SHORT[v] + '</extra>')
    # Starts forecast: actual -> 2026Q4 trend forecast -> 2027 expected starts of the current plan.
    e = vf[vf.version == PCX.CURRENT_VERSION].groupby('quarter').expected_wspw.sum().reindex(C.PLAN_QUARTERS)
    fig.add_scatter(x=['2026Q4'] + C.PLAN_QUARTERS, y=[q4['forecast_wspw']] + list(e), name='Realization-adjusted starts (before capacity limits)',
                    mode='lines+markers', line=dict(color=INK, width=2.2), marker=dict(size=6, symbol='circle-open', line=dict(width=2)),
                    hovertemplate='%{x} · %{y:,.0f} forecast starts<extra></extra>')
    order = hq + ['2026Q4'] + C.PLAN_QUARTERS
    fig.add_vrect(x0=order.index('2027Q1') - .5, x1=len(order) - .5, fillcolor='#f3f5f8', line_width=0, layer='below',
                  annotation=dict(text='2027 plan', font=dict(size=10, color=MUTED), yanchor='top', y=1, xanchor='left'))
    ec.style(fig, 400)
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=order, showgrid=False, tickangle=-45)
    fig.update_yaxes(title='Wafer starts / week', rangemode='tozero')
    return fig


def version_bridge(vf, base, target, col):
    b = vf[vf.version == base].groupby('program')[col].sum() / 4
    t = vf[vf.version == target].groupby('program')[col].sum() / 4
    d = (t - b.reindex(t.index).fillna(0))
    d = d.reindex([p for p in C.PROGRAM_ORDER if p in d.index])
    SHORTN = {'DRAM_A': 'Node A', 'DRAM_B': 'Node B', 'NAND_C': 'NAND', 'HBM_TV': 'HBM', 'SHORT': 'Short<br>loops'}
    x = [PCX.VERSION_SHORT[base].replace(' ', '<br>')] + [SHORTN[p] for p in d.index] + [PCX.VERSION_SHORT[target].replace(' ', '<br>')]
    y = [b.sum()] + list(d) + [t.sum()]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=['absolute'] + ['relative'] * len(d) + ['total'],
                                 text=[f'{b.sum():,.0f}'] + [f'{v:+,.0f}' for v in d] + [f'{t.sum():,.0f}'], textposition='outside',
                                 textfont=dict(size=11, color=INK), connector=dict(line=dict(color='#cfd4d9', width=1)),
                                 increasing=dict(marker=dict(color='#7fa1c3')), decreasing=dict(marker=dict(color=BAD)),
                                 totals=dict(marker=dict(color=ACCENT)), hovertemplate='%{x}: %{text}<extra></extra>'))
    lo = min(b.sum(), t.sum()) * .8
    ec.style(fig, 360)
    fig.update_layout(showlegend=False, margin=dict(t=30))
    fig.update_yaxes(title='Starts / week (2027 avg)', range=[lo, max(b.sum(), t.sum(), b.sum() + d.clip(lower=0).sum()) * 1.06])
    fig.update_xaxes(showgrid=False, tickangle=0, tickfont=dict(size=10))
    return fig


def program_multiples(hp, bridge, vf, col, shown):
    from plotly.subplots import make_subplots
    progs = [p for p in C.PROGRAM_ORDER if p in set(vf.program)]
    fig = make_subplots(rows=1, cols=len(progs), subplot_titles=[D.NAMES[p].replace(' (EUV)', '') for p in progs], horizontal_spacing=.035)
    order = sorted(hp.quarter.unique()) + ['2026Q4'] + C.PLAN_QUARTERS
    ticks = ['2024Q1', '2025Q1', '2026Q1', '2027Q1']
    for i, p in enumerate(progs, start=1):
        h = hp[hp.program == p].groupby('quarter')[['requested_wspw', 'actual_wspw']].sum()
        q4 = bridge[bridge.program == p]
        q4r, q4f = float(q4.requested_wspw.sum()), float(q4.forecast_wspw.sum())
        fig.add_scatter(x=list(h.index) + ['2026Q4'], y=list(h.requested_wspw) + [q4r], mode='lines', line=dict(color='#c3c8ce', width=1.6),
                        showlegend=False, row=1, col=i, hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
        e = vf[(vf.version == PCX.CURRENT_VERSION) & (vf.program == p)].groupby('quarter').expected_wspw.sum().reindex(C.PLAN_QUARTERS)
        fig.add_scatter(x=list(h.index) + ['2026Q4'] + C.PLAN_QUARTERS, y=list(h.actual_wspw) + [q4f] + list(e), mode='lines',
                        line=dict(color=INK, width=1.6), showlegend=False, row=1, col=i,
                        hovertemplate='%{x} · %{y:,.0f} starts (actual → forecast)<extra></extra>')
        anchor = q4r if col == 'requested_wspw' else q4f
        for v in PCX.VERSIONS:
            if v not in shown:
                continue
            t = vf[(vf.version == v) & (vf.program == p)].groupby('quarter')[col].sum().reindex(C.PLAN_QUARTERS)
            fig.add_scatter(x=['2026Q4'] + C.PLAN_QUARTERS, y=[anchor] + list(t), mode='lines', showlegend=False, row=1, col=i,
                            line=dict(color=VERSION_COLOR[v], width=2, dash=VERSION_DASH[v]),
                            hovertemplate='%{x} · %{y:,.0f}<extra>' + PCX.VERSION_SHORT[v] + '</extra>')
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=order, tickvals=ticks, ticktext=['24', '25', '26', '27'],
                         showgrid=False, row=1, col=i)
        fig.update_yaxes(rangemode='tozero', row=1, col=i)
    ec.style(fig, 250)
    fig.update_annotations(font=dict(size=12, color=INK))
    fig.update_layout(margin=dict(t=40, b=30))
    return fig


def version_mix(frame, col, group):
    key = {'Program': 'program', 'Family': 'technology_family', 'Fab': 'site'}[group]
    fig = go.Figure()
    values = list(dict.fromkeys(frame[key]))
    if key == 'program':
        values = [p for p in C.PROGRAM_ORDER if p in values]
    palette = ['#2f5d8a', '#7fa1c3', '#6f8f7a', '#b08d4a', '#c9ced4']
    for i, v in enumerate(values):
        rows = frame[frame[key] == v].groupby('quarter')[col].sum().reindex(C.PLAN_QUARTERS)
        name = D.NAMES[v] if key == 'program' else rm.SITE_LABELS.get(v, v)
        color = PROGRAM_COLOR[v] if key == 'program' else (ec.SITE_COLORS.get(v) or palette[i % len(palette)])
        fig.add_bar(x=rows.index, y=rows, name=name, marker_color=color, hovertemplate='%{x} · %{y:,.0f}<extra>' + name + '</extra>')
    ec.style(fig, 320)
    fig.update_layout(barmode='stack')
    fig.update_yaxes(title='Wafer starts / week')
    fig.update_xaxes(showgrid=False)
    return fig


def program_vs_plan(programs):
    f = programs.sort_values('plan_wspw', ascending=True)
    names = [D.NAMES[p] for p in f.program]
    fig = go.Figure()
    fig.add_bar(y=names, x=f.plan_wspw, orientation='h', name='2026Q4 request', marker_color='#d9dde2',
                hovertemplate='%{y}<br>Plan %{x:,.0f}<extra></extra>')
    fig.add_bar(y=names, x=f.actual_wspw, orientation='h', name='Actual', marker_color=INK, width=.4,
                text=[f'{a:.0f}%' for a in f.attainment_pct], textposition='outside', textfont=dict(size=11, color=MUTED),
                hovertemplate='%{y}<br>Actual %{x:,.0f}<extra></extra>')
    ec.style(fig, 280)
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(title='Wafer starts / week', showgrid=False, range=[0, float(f.plan_wspw.max()) * 1.18])
    fig.update_yaxes(showgrid=False)
    return fig


def scenario_delta(frame, ceiling, both):
    """Baseline vs. scenario loading for the tool groups a scenario changes."""
    frame = frame.sort_values('loading_pct_scenario', ascending=False, na_position='first')
    names = frame.tool_group.map(rm.GROUP_LABELS)
    labels = (frame.site.map(rm.SITE_LABELS) + ' · ' + names) if both else names
    fig = go.Figure()
    for col, name, color in [('loading_pct_base', 'Baseline', '#bac9dd'), ('loading_pct_scenario', 'Scenario', ACCENT)]:
        fig.add_bar(y=labels, x=frame[col].fillna(0), orientation='h', name=name, marker_color=color,
                    text=frame[col].map(lambda x: 'No capacity' if pd.isna(x) else f'{x:.0f}%'), textposition='outside', cliponaxis=False)
    ec.style(fig, max(260, len(frame) * 44 + 80))
    fig.update_layout(barmode='group', bargap=.28)
    fig.add_vline(x=ceiling * 100, line_color=WARN, line_width=1.5)
    top = frame[['loading_pct_base', 'loading_pct_scenario']].max().max()
    fig.update_xaxes(title='Loading (%)', range=[0, max(110, (0 if pd.isna(top) else top) * 1.15)])
    fig.update_yaxes(autorange='reversed')
    return fig


TIME_STATES = [('productive_pct', 'Productive', '#2f5d8a'), ('standby_pct', 'Standby (idle, available)', '#c9d6e3'),
               ('engineering_pct', 'Engineering', '#8a9199'), ('sched_down_pct', 'Scheduled down (PM)', '#e3c48f'),
               ('unsched_down_pct', 'Unscheduled down', '#b8412f')]


def time_split(fleet, both):
    """Where each tool group's calendar time went (E10-style states), sorted by productive share."""
    f = fleet.sort_values('productive_pct')
    labels = (f.site.map(rm.SITE_LABELS) + ' · ' if both else '') + f.tool_group.map(rm.GROUP_LABELS)
    fig = go.Figure()
    for col, name, color in TIME_STATES:
        fig.add_bar(y=labels, x=f[col], orientation='h', name=name, marker_color=color,
                    text=f[col].map(lambda v: f'{v:.0f}' if v >= 6 else ''), textposition='inside', insidetextanchor='middle',
                    hovertemplate='%{y}<br>' + name + ' %{x:.1f}%<extra></extra>')
    ec.style(fig, max(300, 26 * len(f) + 90))
    fig.update_layout(barmode='stack', bargap=.3)
    fig.update_xaxes(title='Share of calendar time, 2026Q4 to date (%)', range=[0, 100])
    fig.update_yaxes(showgrid=False)
    return fig
