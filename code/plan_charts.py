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
def demand_trend(totals, history):
    """Actual and requested history, then the 2027 request and expected starts."""
    fig = go.Figure()
    h = history
    fig.add_scatter(x=h.index, y=h.requested_wspw, name='Requested (history)', mode='lines',
                    line=dict(color=HISTORY, width=1.5, dash='dot'), hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
    fig.add_scatter(x=h.index, y=h.actual_wspw, name='Actual starts', mode='lines+markers',
                    line=dict(color=INK, width=2), marker=dict(size=5), hovertemplate='%{x} · %{y:,.0f} actual<extra></extra>')
    t = totals
    fig.add_scatter(x=t.index, y=t.requested_wspw, name='2027 requested', mode='lines+markers',
                    line=dict(color=ACCENT, width=2.5), marker=dict(size=6), hovertemplate='%{x} · %{y:,.0f} requested<extra></extra>')
    fig.add_scatter(x=t.index, y=t.expected_wspw, name='2027 expected', mode='lines+markers',
                    line=dict(color=ACCENT, width=2, dash='dash'), marker=dict(size=6, symbol='circle-open'),
                    hovertemplate='%{x} · %{y:,.0f} expected<extra></extra>')
    fig.add_vrect(x0=len(h) - .5, x1=len(h) + len(t) - .5, fillcolor='#f3f5f8', line_width=0, layer='below')
    fig.add_annotation(x=len(h) - .45, y=1, yref='paper', text='Plan', showarrow=False, xanchor='left',
                       font=dict(size=10, color=MUTED))
    ec.style(fig, 330)
    order = list(h.index) + list(t.index)
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
def waterfall(steps):
    values = steps.net_new.tolist()
    measure = ['absolute'] + ['relative'] * (len(values) - 1) + ['total']
    y = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))] + [values[-1]]
    x = [str(v).replace(' ', '<br>', 1) for v in steps.step] + ['Tools<br>to buy']
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
    fig.update_xaxes(title='Production loading, last 13 weeks (%)', range=[40, 100])
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
