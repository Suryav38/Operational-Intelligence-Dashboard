"""
Forecast accuracy: how closely do program teams' wafer-start requests match what
they actually run?

Model (one per program, pooled across sites):
    actual_wspw = b x requested_wspw          (least squares through the origin)
b is the share of a request that historically gets run (b < 1: over-request).

Backtest: fit on 2024Q1-2025Q4, test on 2026Q1-2026Q3, and compare the error of
using the raw request as the forecast with the error of the corrected forecast
(MAPE = mean absolute percentage error).
"""
import numpy as np
import pandas as pd

import fab_config as C

TRAIN_END = "2025Q4"


def _slope(df):
    x, y = df["requested_wspw"].to_numpy(float), df["actual_wspw"].to_numpy(float)
    return float((x * y).sum() / (x * x).sum()) if len(df) else 1.0


def _mape(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.mean(np.abs(forecast - actual) / actual) * 100) if len(actual) else float("nan")


def fit(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in C.PROGRAM_ORDER:
        h = history[history["program"] == p]
        train, test = h[h["quarter"] <= TRAIN_END], h[h["quarter"] > TRAIN_END]
        b_train = _slope(train)
        b_all = _slope(h)
        rows.append({
            "program": p, "program_name": C.PROGRAMS[p]["name"], "quarters": int(h["quarter"].nunique()),
            "realization_rate": round(b_all, 3),
            "avg_bias_pct": round((h["requested_wspw"].sum() / h["actual_wspw"].sum() - 1) * 100, 1),
            "backtest_mape_raw_pct": round(_mape(test["actual_wspw"], test["requested_wspw"]), 1),
            "backtest_mape_corrected_pct": round(_mape(test["actual_wspw"], b_train * test["requested_wspw"]), 1),
        })
    return pd.DataFrame(rows)


def corrected_plan(plan: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    rate = model.set_index("program")["realization_rate"]
    out = plan.copy()
    out["corrected_wspw"] = (out["requested_wspw"] * out["program"].map(rate)).round(0)
    return out


def summary(history, plan) -> dict:
    m = fit(history)
    test_raw = _mape_all(history, m, corrected=False)
    test_cor = _mape_all(history, m, corrected=True)
    cp = corrected_plan(plan, m)
    return {"model": m.to_dict(orient="records"),
            "backtest_quarters": sorted(history[history["quarter"] > TRAIN_END]["quarter"].unique().tolist()),
            "overall_backtest_mape_raw_pct": round(test_raw, 1),
            "overall_backtest_mape_corrected_pct": round(test_cor, 1),
            "plan_2027_totals": cp.groupby("quarter")[["requested_wspw", "corrected_wspw"]].sum().astype(int)
                                  .reset_index().to_dict(orient="records")}


def _mape_all(history, model, corrected):
    rows = []
    for p in C.PROGRAM_ORDER:
        h = history[history["program"] == p]
        train, test = h[h["quarter"] <= TRAIN_END], h[h["quarter"] > TRAIN_END]
        b = _slope(train) if corrected else 1.0
        rows += list(np.abs(b * test["requested_wspw"] - test["actual_wspw"]) / test["actual_wspw"])
    return float(np.mean(rows) * 100)
