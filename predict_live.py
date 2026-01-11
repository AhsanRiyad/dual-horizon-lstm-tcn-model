#!/usr/bin/env python3
"""
predict_live.py
- Loads trained model + artifacts
- Watches XAU 5m CSV (updated over time)
- For each NEW bar detected, generates a signal using ONLY past data (up to previous bar)
- Maintains online rolling-quantile threshold state across restarts

Run (continuous):
  python predict_live.py \
    --xau_csv "/path/XAU_5m_data.csv" --xau_sep ";" --xau_date_format "%Y.%m.%d %H:%M" \
    --dxy_csv "/path/DXY_merged_2013_2020_2021.csv" \
    --spx_csv "/path/SPX_2013_2021_merged.csv" \
    --model "./out/dual_horizon_lstm_tcn.keras" \
    --artifacts "./out/artifacts.joblib" \
    --state "./out/live_state.joblib" \
    --poll_seconds 15

Run (one-shot: process whatever is new and exit):
  python predict_live.py ... --once
"""

import time
import argparse
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib


EPS = 1e-12


# -----------------------------
# Helpers (same as train)
# -----------------------------
def _clean_cols(df_: pd.DataFrame) -> pd.DataFrame:
    df_.columns = [c.strip() for c in df_.columns]
    return df_

def _parse_dt(s: pd.Series, fmt: Optional[str]) -> pd.Series:
    return pd.to_datetime(s, format=fmt, errors="coerce") if fmt else pd.to_datetime(s, errors="coerce")

def rolling_std(x, w):
    return pd.Series(x).rolling(w, min_periods=1).std().fillna(0.0).to_numpy()

def rolling_mean(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().fillna(0.0).to_numpy()

def load_daily_series(path, sep, date_col, date_format, value_col, series_name):
    d = pd.read_csv(path, sep=sep, engine="python", thousands=",")
    d = _clean_cols(d)

    if date_col not in d.columns:
        raise ValueError(f"{series_name}: date column '{date_col}' not found. Found: {d.columns.tolist()}")
    if value_col not in d.columns:
        raise ValueError(f"{series_name}: value column '{value_col}' not found. Found: {d.columns.tolist()}")

    d[date_col] = _parse_dt(d[date_col], date_format)
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).reset_index(drop=True)

    lvl = d[value_col].to_numpy(dtype=float)
    log_lvl = np.log(np.maximum(lvl, EPS))
    ret1d = np.diff(log_lvl, prepend=log_lvl[0])

    out = pd.DataFrame({
        "date_only": d[date_col].dt.floor("D"),
        f"{series_name}_ret1d": ret1d.astype(float),
    }).groupby("date_only", as_index=False).last()

    return out

def build_features(g: pd.DataFrame) -> np.ndarray:
    open_  = g["Open"].to_numpy(dtype=float)
    high   = g["High"].to_numpy(dtype=float)
    low    = g["Low"].to_numpy(dtype=float)
    close  = g["Close"].to_numpy(dtype=float)
    volume = g["Volume"].to_numpy(dtype=float)

    dxy_ret = g["DXY_ret1d"].to_numpy(dtype=float)
    spx_ret = g["SPX_ret1d"].to_numpy(dtype=float)

    log_close = np.log(np.maximum(close, EPS))
    log_open  = np.log(np.maximum(open_, EPS))
    log_vol   = np.log(np.maximum(volume, EPS))

    ret_close = np.diff(log_close, prepend=log_close[0])
    ret_open  = np.diff(log_open,  prepend=log_open[0])

    hl_range  = (high - low) / (close + EPS)
    oc_change = (close - open_) / (open_ + EPS)

    vol_20  = rolling_std(ret_close, 20)
    vol_96  = rolling_std(ret_close, 96)
    vol_288 = rolling_std(ret_close, 288)

    mom_20 = rolling_mean(ret_close, 20)
    mom_96 = rolling_mean(ret_close, 96)

    X_feat = np.column_stack([
        ret_close, ret_open,
        hl_range, oc_change,
        log_close, log_vol,
        vol_20, vol_96, vol_288,
        mom_20, mom_96,
        dxy_ret, spx_ret
    ]).astype(np.float32)

    return X_feat

class OnlineThreshold:
    """
    Online rolling-quantile threshold controller.
    thr(t) computed from buffer (past probs), then we update buffer with p(t).
    """
    def __init__(self, alert_rate: float, window: int, init_buffer=None):
        self.alert_rate = float(np.clip(alert_rate, 1e-4, 0.9999))
        self.window = int(window)
        self.buf = deque(init_buffer or [], maxlen=self.window)

    def current_threshold(self) -> float:
        if len(self.buf) < 10:
            return 1.0
        arr = np.asarray(self.buf, dtype=float)
        return float(np.quantile(arr, 1.0 - self.alert_rate))

    def step(self, p: float):
        thr = self.current_threshold()
        alert = (p >= thr)
        self.buf.append(float(p))
        return thr, alert

    def dump(self):
        return list(self.buf)


def load_and_merge(xau_csv: str, xau_sep: str, xau_date_col: str, xau_date_format: str,
                   dxy_df: pd.DataFrame, spx_df: pd.DataFrame,
                   macro_shift_days: int):
    df = pd.read_csv(xau_csv, sep=xau_sep, engine="python")
    df = _clean_cols(df)

    required = ["Open", "High", "Low", "Close", "Volume", xau_date_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"XAU: Missing columns {missing}. Found: {df.columns.tolist()}")

    df[xau_date_col] = _parse_dt(df[xau_date_col], xau_date_format)
    df = df.dropna(subset=[xau_date_col]).sort_values(xau_date_col).reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

    dxy = dxy_df.copy()
    spx = spx_df.copy()
    if macro_shift_days != 0:
        dxy["date_only"] = dxy["date_only"] + pd.Timedelta(days=macro_shift_days)
        spx["date_only"] = spx["date_only"] + pd.Timedelta(days=macro_shift_days)

    df["date_only"] = df[xau_date_col].dt.floor("D")
    df = df.merge(dxy, on="date_only", how="left").merge(spx, on="date_only", how="left")
    df[["DXY_ret1d", "SPX_ret1d"]] = df[["DXY_ret1d", "SPX_ret1d"]].ffill().fillna(0.0)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xau_csv", required=True)
    ap.add_argument("--dxy_csv", required=True)
    ap.add_argument("--spx_csv", required=True)

    ap.add_argument("--model", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--state", required=True)

    ap.add_argument("--xau_sep", default=";")
    ap.add_argument("--xau_date_col", default="Date")
    ap.add_argument("--xau_date_format", default="%Y.%m.%d %H:%M")

    ap.add_argument("--poll_seconds", type=int, default=15)
    ap.add_argument("--once", action="store_true")

    ap.add_argument("--macro_shift_days", type=int, default=0,
                    help="0=notebook behavior; 1=safer intraday (use previous day daily returns)")

    args = ap.parse_args()

    # Load artifacts + model
    art: Dict[str, Any] = joblib.load(args.artifacts)
    scaler = art["scaler"]
    yP_mean = float(art["yP_mean"])
    yP_std = float(art["yP_std"])
    cfg = art["cfg"]
    lookback = int(cfg["lookback"])

    model = tf.keras.models.load_model(args.model, compile=False)

    # Load daily once
    dxy = load_daily_series(args.dxy_csv, ",", "Date", None, "Close", "DXY")
    spx = load_daily_series(args.spx_csv, ",", "Date", None, "Close", "SPX")

    # Load/Init state
    state = joblib.load(args.state) if os.path.exists(args.state) else {
        "prob_buffer": [],
        "last_processed_dt": None,
        "rolling_thr_window": int(cfg.get("rolling_thr_window", 500)),
        "target_alert_rate": float(cfg.get("target_alert_rate", 0.15)),
    }

    thr_ctl = OnlineThreshold(
        alert_rate=float(state.get("target_alert_rate", 0.15)),
        window=int(state.get("rolling_thr_window", 500)),
        init_buffer=state.get("prob_buffer", [])
    )

    last_processed_dt = state.get("last_processed_dt", None)
    if last_processed_dt is not None:
        last_processed_dt = pd.to_datetime(last_processed_dt)

    def scale_X_one(X_seq: np.ndarray) -> np.ndarray:
        # X_seq shape: (1, lookback, n_features)
        Xs = scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape).astype(np.float32)
        return Xs

    def process_new_rows():
        nonlocal last_processed_dt

        df = load_and_merge(
            xau_csv=args.xau_csv,
            xau_sep=args.xau_sep,
            xau_date_col=args.xau_date_col,
            xau_date_format=args.xau_date_format,
            dxy_df=dxy,
            spx_df=spx,
            macro_shift_days=args.macro_shift_days
        )

        if len(df) < lookback + 2:
            print(f"[WAIT] Not enough rows yet. Have={len(df)}, need≈{lookback+2}")
            return

        # Determine new bars
        dts = df[args.xau_date_col]
        if last_processed_dt is None:
            # Start from the last available bar (but we need at least lookback history)
            start_idx = lookback
        else:
            new_mask = dts > last_processed_dt
            if not new_mask.any():
                return
            start_idx = int(np.argmax(new_mask.to_numpy()))

        # Precompute features once
        X_feat = build_features(df)

        # For each newly observed bar at index i:
        # Generate a signal for time i using ONLY history up to i-1:
        # window = X_feat[i-lookback : i]  (ends at i-1)
        for i in range(start_idx, len(df)):
            bar_time = df[args.xau_date_col].iloc[i]
            if last_processed_dt is not None and bar_time <= last_processed_dt:
                continue

            if i < lookback:
                continue

            X_win = X_feat[i - lookback:i]  # shape (lookback, n_features)
            X_in = X_win[None, :, :]        # (1, lookback, n_features)
            X_in_s = scale_X_one(X_in)

            pred = model.predict(X_in_s, verbose=0)
            p_crash = float(pred["crash"].reshape(-1)[0])

            # price head (standardized) -> unstandardize to log-return
            r_pred = float(pred["price"].reshape(-1)[0] * yP_std + yP_mean)

            thr, alert = thr_ctl.step(p_crash)

            # Print signal (you can redirect to file / DB)
            tag = "ALERT" if alert else "-----"
            print(f"[{bar_time}] crash_prob={p_crash:.4f} thr={thr:.4f} {tag} | pred_ret={r_pred:+.6f}")

            last_processed_dt = bar_time

        # Save state
        state_out = {
            "prob_buffer": thr_ctl.dump(),
            "last_processed_dt": None if last_processed_dt is None else pd.Timestamp(last_processed_dt).isoformat(),
            "rolling_thr_window": thr_ctl.window,
            "target_alert_rate": thr_ctl.alert_rate,
        }
        joblib.dump(state_out, args.state)

    # Main loop
    while True:
        try:
            process_new_rows()
        except Exception as e:
            print("[ERROR]", repr(e))

        if args.once:
            break
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    import os
    main()
