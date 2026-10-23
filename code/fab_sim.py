"""
Discrete-event simulation (SimPy) of one development fab for Q3 2026.

What it models:
  * Lots of 25 wafers released at each program's actual wafer-start rate.
  * Re-entrant routes: a lot visits the same tool groups again and again, layer
    after layer, in the order built by `build_route`.
  * Tool groups are pools of identical tools. Hot lots jump the queue.
  * Furnaces are batch tools (a run holds up to 3 lots for 6 hours).
  * Metrology measures only a sample of lots at each metrology step.
  * Tools leave production for preventive maintenance, breakdowns and engineering
    experiments (SEMI E10 states), at the rates set in fab_config.
  * Between steps a lot is transported and dispatched (time it spends neither
    queued at a tool nor processing), and lots are sometimes put on engineering
    hold for review or split decisions, which is common in development fabs.
  * Wafers are occasionally scrapped along the route (line yield).

It runs a 12-week warm-up, then records 13 weeks of results. Outputs feed the
"Fab performance" strip and give an independent check on the capacity model:
the model's predicted loading should match the simulated utilization.
"""
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import simpy

import fab_config as C

SIM_START = datetime(2026, 4, 6)      # Monday; window starts after warm-up
WARMUP_WEEKS = 12
MEASURE_WEEKS = 13
PRIORITY = {"hold": 0, "hot": 1, "normal": 2}
TRANSPORT_H = {"hot": 0.30, "normal": 0.60}       # mean transport/dispatch time per step
LOT_HOLD_PROB = 0.025                             # chance a lot goes on engineering hold after a step
LOT_HOLD_H = {"hot": 12.0, "normal": 30.0}        # mean engineering-hold duration
LAYER_ORDER = {"WET": 0.00, "DEP": 0.10, "FURN": 0.20, "EUV": 0.35, "DUV": 0.35, "METRO": 0.50,
               "ETCH": 0.60, "IMPL": 0.70, "CMP": 0.80, "PVD": 0.90}


def build_route(program: str) -> list:
    """Interleave each tool group's passes evenly across the route (re-entrant flow)."""
    steps = []
    for tg, n in C.PROGRAMS[program]["passes"].items():
        for k in range(n):
            steps.append(((k + LAYER_ORDER[tg]) / n, tg))
    return [tg for _, tg in sorted(steps)]


class FabSim:
    def __init__(self, site: str, seed: int = 7):
        self.site = site
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(seed)
        self.t_warm = WARMUP_WEEKS * C.HOURS_PER_WEEK
        self.t_end = self.t_warm + MEASURE_WEEKS * C.HOURS_PER_WEEK
        self.tools = {tg: n for tg, n in C.INSTALLED_TOOLS[site].items() if n > 0}
        self.slots = {tg: C.TOOL_GROUPS[tg].get("batch_lots", 1) for tg in self.tools}
        self.res = {tg: simpy.PriorityResource(self.env, capacity=n * self.slots[tg]) for tg, n in self.tools.items()}
        self.routes = {p: build_route(p) for p in C.QUALIFIED[site]}
        # accumulators (measurement window only)
        self.state_h = defaultdict(lambda: defaultdict(float))     # tg -> state -> slot-hours
        self.queue = defaultdict(lambda: [0.0, 0])                 # tg -> [queue hours, steps]
        self.proc = defaultdict(lambda: [0.0, 0])
        self.weekly_tg = defaultdict(lambda: defaultdict(float))   # (week, tg) -> metrics
        self.weekly = defaultdict(lambda: defaultdict(float))      # week -> metrics
        self.lots = []
        self.active = {}
        self.lot_seq = 0

    # -------------------------------------------------------------- helpers
    def _window_overlap(self, t0, t1):
        return max(0.0, min(t1, self.t_end) - max(t0, self.t_warm))

    def _week(self, t):
        return int((t - self.t_warm) // C.HOURS_PER_WEEK) if self.t_warm <= t < self.t_end else None

    def _proc_time(self, program, tg, wafers):
        t = C.TOOL_GROUPS[tg]
        if "run_hours" in t:
            return t["run_hours"] * self.rng.gamma(25, 1 / 25)
        mean = wafers * t["min_per_wafer"] * C.PROGRAMS[program]["time_mult"].get(tg, 1.0) / 60
        return mean * self.rng.gamma(16, 1 / 16)          # CV 0.25

    def _nominal_time(self, program, tg, wafers):
        t = C.TOOL_GROUPS[tg]
        if "run_hours" in t:
            return t["run_hours"]
        return wafers * t["min_per_wafer"] * C.PROGRAMS[program]["time_mult"].get(tg, 1.0) / 60

    # -------------------------------------------------------------- tool holds (E10)
    def hold_process(self, tg, state, duration_mean, frac, exponential):
        """One tool's PM / breakdown / engineering cycle. Holds a whole tool (all batch slots)."""
        if frac <= 0:
            return
        gap_mean = duration_mean * (1 - frac) / frac
        yield self.env.timeout(self.rng.uniform(0, gap_mean))
        while True:
            reqs = [self.res[tg].request(priority=PRIORITY["hold"]) for _ in range(self.slots[tg])]
            yield simpy.AllOf(self.env, reqs)
            dur = self.rng.exponential(duration_mean) if exponential else duration_mean * self.rng.uniform(0.8, 1.2)
            t0 = self.env.now
            yield self.env.timeout(dur)
            self.state_h[tg][state] += self._window_overlap(t0, self.env.now) * self.slots[tg]
            for r in reqs:
                self.res[tg].release(r)
            gap = self.rng.exponential(gap_mean) if exponential else gap_mean * self.rng.uniform(0.8, 1.2)
            yield self.env.timeout(gap)

    # -------------------------------------------------------------- lots
    def lot(self, program, hot):
        self.lot_seq += 1
        lot_id = f"{self.site[-1]}{self.lot_seq:05d}"
        route = self.routes[program]
        wafers = C.LOT_SIZE
        survive = C.LINE_YIELD[program] ** (1 / len(route))
        start = self.env.now
        rec = {"lot_id": lot_id, "site": self.site, "program": program, "priority": "HOT" if hot else "NORMAL",
               "start_h": start, "wafers_in": wafers, "steps_total": len(route)}
        self.active[lot_id] = rec
        wk = self._week(start)
        if wk is not None:
            self.weekly[wk]["starts"] += wafers
        rpt = 0.0
        parts = {"process_h": 0.0, "queue_h": 0.0, "transport_h": 0.0, "hold_h": 0.0}
        kind = "hot" if hot else "normal"
        sample = C.TOOL_GROUPS["METRO"]["sample_rate"]
        for i, tg in enumerate(route):
            rec["steps_done"] = i
            if tg == "METRO" and self.rng.random() > sample:
                continue
            move = self.rng.exponential(TRANSPORT_H[kind])
            yield self.env.timeout(move)
            parts["transport_h"] += move
            t_q = self.env.now
            with self.res[tg].request(priority=PRIORITY["hot" if hot else "normal"]) as req:
                yield req
                t_s = self.env.now
                pt = self._proc_time(program, tg, wafers)
                yield self.env.timeout(pt)
            t_e = self.env.now
            parts["queue_h"] += t_s - t_q
            parts["process_h"] += pt
            rpt += self._nominal_time(program, tg, wafers)
            wafers = int(self.rng.binomial(wafers, survive))
            if self.t_warm <= t_s < self.t_end:
                q = self.queue[tg]; q[0] += t_s - t_q; q[1] += 1
                p = self.proc[tg]; p[0] += pt; p[1] += 1
            self.state_h[tg]["productive"] += self._window_overlap(t_s, t_e)
            wk = self._week(t_e)
            if wk is not None:
                self.weekly[wk]["moves"] += wafers
                self.weekly_tg[(wk, tg)]["moves"] += wafers
                self.weekly_tg[(wk, tg)]["queue_h"] += t_s - t_q
                self.weekly_tg[(wk, tg)]["steps"] += 1
            if wafers == 0:
                break
            if self.rng.random() < LOT_HOLD_PROB:
                hold = self.rng.exponential(LOT_HOLD_H[kind])
                yield self.env.timeout(hold)
                parts["hold_h"] += hold
        end = self.env.now
        del self.active[lot_id]
        rec.update({"end_h": end, "wafers_out": wafers, "steps_done": len(route), "rpt_h": rpt,
                    "ct_h": end - start, **parts})
        self.lots.append(rec)
        wk = self._week(end)
        if wk is not None:
            self.weekly[wk]["outs"] += wafers

    def release(self, program, wspw):
        lots_per_week = wspw / C.LOT_SIZE
        gap = C.HOURS_PER_WEEK / lots_per_week
        yield self.env.timeout(self.rng.uniform(0, gap))
        while self.env.now < self.t_end:
            self.env.process(self.lot(program, hot=self.rng.random() < C.HOT_LOT_SHARE))
            yield self.env.timeout(gap * self.rng.uniform(0.7, 1.3))

    def wip_snapshots(self):
        yield self.env.timeout(self.t_warm)
        for wk in range(MEASURE_WEEKS):
            yield self.env.timeout(C.HOURS_PER_WEEK)
            self.weekly[wk]["wip_wafers"] = sum(r["wafers_in"] for r in self.active.values())
            self.weekly[wk]["wip_lots"] = len(self.active)

    # -------------------------------------------------------------- run
    def run(self):
        for tg, n in self.tools.items():
            t = C.TOOL_GROUPS[tg]
            for _ in range(n):
                self.env.process(self.hold_process(tg, "sched_down", 8.0, t["pm"], False))
                self.env.process(self.hold_process(tg, "unsched_down", 4.0, t["down"], True))
                self.env.process(self.hold_process(tg, "engineering", 6.0, t["eng"], True))
        for program, wspw in C.ACTUAL_WSPW_2026Q3[self.site].items():
            self.env.process(self.release(program, wspw))
        self.env.process(self.wip_snapshots())
        self.env.run(until=self.t_end + 1e-6)
        for rec in self.active.values():   # lots still in the fab at the end
            self.lots.append({**rec, "end_h": None, "wafers_out": None, "rpt_h": None, "ct_h": None,
                              "process_h": None, "queue_h": None, "transport_h": None, "hold_h": None})
        return self

    # -------------------------------------------------------------- results
    def ts(self, h):
        return None if h is None else SIM_START + timedelta(hours=h)

    def results(self):
        window_h = MEASURE_WEEKS * C.HOURS_PER_WEEK
        lots = []
        for r in self.lots:
            if r["end_h"] is not None and r["end_h"] < self.t_warm:
                continue
            lots.append({
                "lot_id": r["lot_id"], "site": r["site"], "program": r["program"], "priority": r["priority"],
                "start_time": self.ts(r["start_h"]).isoformat(timespec="minutes"),
                "end_time": self.ts(r["end_h"]).isoformat(timespec="minutes") if r["end_h"] is not None else None,
                "status": "COMPLETE" if r["end_h"] is not None else "WIP",
                "wafers_in": r["wafers_in"], "wafers_out": r["wafers_out"],
                "steps_total": r["steps_total"], "steps_done": r["steps_done"],
                "raw_process_h": round(r["rpt_h"], 2) if r["rpt_h"] else None,
                "cycle_time_days": round(r["ct_h"] / 24, 2) if r["ct_h"] else None,
                **{k.replace("_h", "_days"): (round(r[k] / 24, 2) if r[k] is not None else None) for k in ("process_h", "queue_h", "transport_h", "hold_h")},
            })
        tg_rows = []
        for tg, n in self.tools.items():
            cap = n * self.slots[tg] * window_h
            s = self.state_h[tg]
            nonprod = s["sched_down"] + s["unsched_down"] + s["engineering"]
            standby = max(0.0, cap - nonprod - s["productive"])
            q, p = self.queue[tg], self.proc[tg]
            tg_rows.append({
                "site": self.site, "tool_group": tg, "tools": n,
                "productive_pct": round(s["productive"] / cap * 100, 1),
                "standby_pct": round(standby / cap * 100, 1),
                "engineering_pct": round(s["engineering"] / cap * 100, 1),
                "sched_down_pct": round(s["sched_down"] / cap * 100, 1),
                "unsched_down_pct": round(s["unsched_down"] / cap * 100, 1),
                "loading_pct": round(s["productive"] / max(cap - nonprod, 1e-9) * 100, 1),
                "avg_queue_h": round(q[0] / q[1], 2) if q[1] else 0.0,
                "avg_process_h": round(p[0] / p[1], 2) if p[1] else 0.0,
                "lot_steps": q[1],
            })
        week_rows, wtg_rows = [], []
        done = [l for l in lots if l["status"] == "COMPLETE"]
        for wk in range(MEASURE_WEEKS):
            w0 = SIM_START + timedelta(hours=self.t_warm + wk * C.HOURS_PER_WEEK)
            w1 = w0 + timedelta(weeks=1)
            outs = [l for l in done if w0.isoformat() <= l["end_time"] < w1.isoformat()]
            full = [l for l in outs if l["program"] != "SHORT"]
            hot = [l for l in full if l["priority"] == "HOT"]
            norm = [l for l in full if l["priority"] == "NORMAL"]
            m = self.weekly[wk]
            week_rows.append({
                "site": self.site, "week_start": w0.date().isoformat(),
                "wafer_starts": int(m["starts"]), "wafer_outs": int(m["outs"]),
                "moves_per_day": round(m["moves"] / 7), "wip_wafers": int(m["wip_wafers"]),
                "avg_cycle_time_days": round(np.mean([l["cycle_time_days"] for l in full]), 1) if full else None,
                "days_per_mask_layer": round(np.mean([l["cycle_time_days"] / C.PROGRAMS[l["program"]]["mask_layers"] for l in full]), 2) if full else None,
                "x_factor": round(np.mean([l["cycle_time_days"] * 24 / l["raw_process_h"] for l in full]), 2) if full else None,
                "hot_cycle_time_days": round(np.mean([l["cycle_time_days"] for l in hot]), 1) if hot else None,
                "normal_cycle_time_days": round(np.mean([l["cycle_time_days"] for l in norm]), 1) if norm else None,
                "line_yield_pct": round(sum(l["wafers_out"] for l in outs) / sum(l["wafers_in"] for l in outs) * 100, 2) if outs else None,
            })
            for tg in self.tools:
                d = self.weekly_tg[(wk, tg)]
                wtg_rows.append({"site": self.site, "week_start": w0.date().isoformat(), "tool_group": tg,
                                 "moves": int(d["moves"]),
                                 "avg_queue_h": round(d["queue_h"] / d["steps"], 2) if d["steps"] else 0.0})
        return lots, tg_rows, week_rows, wtg_rows


def simulate_all(seed=7):
    out = {"lots": [], "tool_groups": [], "weekly": [], "weekly_tg": []}
    for i, site in enumerate(C.SITES):
        lots, tg, wk, wtg = FabSim(site, seed + i).run().results()
        out["lots"] += lots; out["tool_groups"] += tg; out["weekly"] += wk; out["weekly_tg"] += wtg
    return out


if __name__ == "__main__":
    import time
    import pandas as pd
    t = time.time()
    res = simulate_all()
    print(f"simulated in {time.time() - t:.0f}s")
    print(pd.DataFrame(res["tool_groups"]).to_string())
    print(pd.DataFrame(res["weekly"]).groupby("site").mean(numeric_only=True).round(2).T)
