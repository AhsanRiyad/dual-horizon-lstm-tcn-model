#!/usr/bin/env python3
"""
train.py
- Train Dual-Head TCN+LSTM on XAU 5m + daily DXY/SPX returns
- Saves:
  - model: dual_horizon_lstm_tcn.keras
  - artifacts: artifacts.joblib  (scaler, yP_mean, yP_std, cfg, feature_order)
  - init live state: live_state.joblib (rolling prob buffer, last_processed_dt)

Run:
  python train.py \
    --xau_csv "/path/XAU_5m_data.csv" --xau_sep ";" --xau_date_format "%Y.%m.%d %H:%M" \
    --dxy_csv "/path/DXY_merged_2013_2020_2021.csv" \
    --spx_csv "/path/SPX_2013_2021_merged.csv" \
    --out_dir "./out"
"""

import os
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib


EPS = 1e-12


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # Paths (overridable by CLI)
    xau_csv_path: str = ""
    xau_sep: str = ";"
    xau_date_col: str = "Date"
    xau_date_format: Optional[str] = "%Y.%m.%d %H:%M"

    dxy_csv_path: str = ""
    dxy_sep: str = ","
    dxy_date_col: str = "Date"
    dxy_date_format: Optional[str] = None
    dxy_value_col: str = "Close"

    spx_csv_path: str = ""
    spx_sep: str = ","
    spx_date_col: str = "Date"
    spx_date_format: Optional[str] = None
    spx_value_col: str = "Close"

    # Feature / model horizons (bars)
    lookback: int = 96
    horizon_price: int = 1

    crash_k: int = 288
    crash_theta: float = 0.025

    # Training
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    seed: int = 42
    train_val_ratio: float = 0.80

    # Online thresholding (used later in live)
    target_alert_rate: float = 0.15
    rolling_thr_window: int = 500

    # Macro handling
    # NOTE: To avoid look-ahead bias with daily close returns during the day, set macro_shift_days=1
    macro_shift_days: int = 0  # 0 = notebook behavior; 1 = safer for real scenario


# -----------------------------
# Helpers
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
    """
    Reads daily series; handles thousands commas (needed for SPX sometimes).
    Returns: date_only, <series>_ret1d
    """
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

def make_crash_labels_within_window_fast(log_close_arr: np.ndarray, k: int, theta: float) -> np.ndarray:
    """
    crash=1 if within next k bars, price drops by >= theta at any point.
    """
    s = pd.Series(log_close_arr, dtype="float64")
    future_min = (
        s.shift(-1)
         .iloc[::-1]
         .rolling(window=k, min_periods=1)
         .min()
         .iloc[::-1]
         .to_numpy()
    )
    thr = np.log(1.0 - theta)  # negative
    drop = future_min - log_close_arr
    y = (drop <= thr).astype(int)
    y[np.isnan(drop)] = 0
    return y.astype(np.float32)

def build_features(g: pd.DataFrame) -> np.ndarray:
    """
    Build features in the SAME order as notebook.
    Output: X_feat shape (n, n_features)
    """
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

def build_sequences_from_df(df_all: pd.DataFrame, cfg: Config):
    """
    Build sequences across the WHOLE dataframe (time-ordered).
    This is more realistic than training only on event windows.

    For crash labels, we need future within crash_k, so we stop earlier.
    """
    df_all = df_all.sort_values(cfg.xau_date_col).reset_index(drop=True)

    close = df_all["Close"].to_numpy(dtype=float)
    log_close = np.log(np.maximum(close, EPS))
    ret_close = np.diff(log_close, prepend=log_close[0]).astype(np.float32)

    X_feat = build_features(df_all)
    y_crash = make_crash_labels_within_window_fast(log_close, cfg.crash_k, cfg.crash_theta)

    Xs, yP, yC = [], [], []
    n = len(df_all)
    last_t = n - max(cfg.horizon_price, cfg.crash_k) - 1
    for t in range(cfg.lookback, last_t + 1):
        Xs.append(X_feat[t - cfg.lookback:t])
        yP.append(ret_close[t + cfg.horizon_price])
        yC.append(y_crash[t])

    return np.array(Xs, np.float32), np.array(yP, np.float32), np.array(yC, np.float32)

def tcn_block(x, filters, kernel_size, dilation_rate, dropout=0.20):
    conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                  dilation_rate=dilation_rate, activation="relu")(x)
    conv = tf.keras.layers.Dropout(dropout)(conv)
    conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                  dilation_rate=dilation_rate, activation="relu")(conv)
    res = x if x.shape[-1] == filters else tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
    out = tf.keras.layers.Add()([res, conv])
    return tf.keras.layers.Activation("relu")(out)

def build_model(lookback: int, n_features: int):
    inp = tf.keras.Input(shape=(lookback, n_features))
    x = inp
    for d in [1, 2, 4, 8]:
        x = tcn_block(x, filters=32, kernel_size=3, dilation_rate=d, dropout=0.20)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    price = tf.keras.layers.Dense(32, activation="relu")(x)
    price = tf.keras.layers.Dense(1, activation="linear", name="price")(price)

    crash = tf.keras.layers.Dense(32, activation="relu")(x)
    crash = tf.keras.layers.Dense(1, activation="sigmoid", name="crash")(crash)

    return tf.keras.Model(inp, {"price": price, "crash": crash})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xau_csv", required=True)
    ap.add_argument("--dxy_csv", required=True)
    ap.add_argument("--spx_csv", required=True)
    ap.add_argument("--out_dir", default="./out")

    ap.add_argument("--xau_sep", default=";")
    ap.add_argument("--xau_date_col", default="Date")
    ap.add_argument("--xau_date_format", default="%Y.%m.%d %H:%M")

    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--crash_k", type=int, default=288)
    ap.add_argument("--crash_theta", type=float, default=0.025)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_val_ratio", type=float, default=0.80)

    ap.add_argument("--target_alert_rate", type=float, default=0.15)
    ap.add_argument("--rolling_thr_window", type=int, default=500)

    ap.add_argument("--macro_shift_days", type=int, default=0,
                    help="0=notebook behavior; 1=safer (use previous day daily returns intraday)")

    args = ap.parse_args()

    cfg = Config(
        xau_csv_path=args.xau_csv,
        dxy_csv_path=args.dxy_csv,
        spx_csv_path=args.spx_csv,
        xau_sep=args.xau_sep,
        xau_date_col=args.xau_date_col,
        xau_date_format=args.xau_date_format,
        lookback=args.lookback,
        crash_k=args.crash_k,
        crash_theta=args.crash_theta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        train_val_ratio=args.train_val_ratio,
        target_alert_rate=args.target_alert_rate,
        rolling_thr_window=args.rolling_thr_window,
        macro_shift_days=args.macro_shift_days
    )

    os.makedirs(args.out_dir, exist_ok=True)

    tf.keras.utils.set_random_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- Load XAU 5m
    df = pd.read_csv(cfg.xau_csv_path, sep=cfg.xau_sep, engine="python")
    df = _clean_cols(df)

    required = ["Open", "High", "Low", "Close", "Volume", cfg.xau_date_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"XAU: Missing columns {missing}. Found: {df.columns.tolist()}")

    df[cfg.xau_date_col] = _parse_dt(df[cfg.xau_date_col], cfg.xau_date_format)
    df = df.dropna(subset=[cfg.xau_date_col]).sort_values(cfg.xau_date_col).reset_index(drop=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

    # ---- Load daily DXY/SPX and merge
    dxy = load_daily_series(cfg.dxy_csv_path, cfg.dxy_sep, cfg.dxy_date_col, cfg.dxy_date_format, cfg.dxy_value_col, "DXY")
    spx = load_daily_series(cfg.spx_csv_path, cfg.spx_sep, cfg.spx_date_col, cfg.spx_date_format, cfg.spx_value_col, "SPX")

    # macro shift to reduce look-ahead
    if cfg.macro_shift_days != 0:
        dxy["date_only"] = dxy["date_only"] + pd.Timedelta(days=cfg.macro_shift_days)
        spx["date_only"] = spx["date_only"] + pd.Timedelta(days=cfg.macro_shift_days)

    df["date_only"] = df[cfg.xau_date_col].dt.floor("D")
    df = df.merge(dxy, on="date_only", how="left").merge(spx, on="date_only", how="left")
    df[["DXY_ret1d", "SPX_ret1d"]] = df[["DXY_ret1d", "SPX_ret1d"]].ffill().fillna(0.0)

    # ---- Build sequences
    X_all, yP_all, yC_all = build_sequences_from_df(df, cfg)
    if len(X_all) == 0:
        raise RuntimeError("No sequences built. Need more rows or reduce lookback/crash_k.")

    N = len(X_all)
    cut = int(N * cfg.train_val_ratio)
    X_train, X_val = X_all[:cut], X_all[cut:]
    yP_train, yP_val = yP_all[:cut], yP_all[cut:]
    yC_train, yC_val = yC_all[:cut], yC_all[cut:]

    print(f"[data] sequences total={N} | train={len(X_train)} val={len(X_val)}")
    print(f"[crash rate] train={yC_train.mean()*100:.2f}% val={yC_val.mean()*100:.2f}%")

    # ---- Scaling
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

    def scale_X(X):
        return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s   = scale_X(X_val)

    yP_mean = float(yP_train.mean())
    yP_std  = float(yP_train.std() + 1e-12)
    yP_train_s = ((yP_train - yP_mean) / yP_std).astype(np.float32)
    yP_val_s   = ((yP_val   - yP_mean) / yP_std).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_s, {"price": yP_train_s, "crash": yC_train})) \
        .shuffle(4096, seed=cfg.seed, reshuffle_each_iteration=True) \
        .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val_s, {"price": yP_val_s, "crash": yC_val})) \
        .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    # ---- Build model
    model = build_model(cfg.lookback, X_train_s.shape[-1])

    pos = float(np.sum(yC_train == 1))
    neg = float(np.sum(yC_train == 0))
    pos_weight = neg / (pos + 1e-12)
    print(f"[imbalance] pos_weight={pos_weight:.2f}")

    def weighted_bce(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        w = 1.0 + y_true * (pos_weight - 1.0)
        return bce * w

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss={"price": tf.keras.losses.Huber(delta=1.0), "crash": weighted_bce},
        loss_weights={"price": 1.0, "crash": 3.0},
        metrics={
            "price": [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
            "crash": [tf.keras.metrics.AUC(name="auc"),
                      tf.keras.metrics.AUC(name="prauc", curve="PR")]
        }
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_crash_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_crash_auc", mode="max", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=1)

    # Quick sanity AUC print
    pred_val = model.predict(val_ds, verbose=0)["crash"].reshape(-1)
    y_val = yC_val.astype(int).reshape(-1)
    if len(np.unique(y_val)) == 2:
        print("[val] ROC-AUC:", roc_auc_score(y_val, pred_val))

    # ---- Save outputs
    model_path = os.path.join(args.out_dir, "dual_horizon_lstm_tcn.keras")
    model.save(model_path)

    feature_order = [
        "ret_close","ret_open","hl_range","oc_change","log_close","log_vol",
        "vol_20","vol_96","vol_288","mom_20","mom_96","DXY_ret1d","SPX_ret1d"
    ]

    artifacts = {
        "scaler": scaler,
        "yP_mean": yP_mean,
        "yP_std": yP_std,
        "cfg": asdict(cfg),
        "feature_order": feature_order,
    }
    artifacts_path = os.path.join(args.out_dir, "artifacts.joblib")
    joblib.dump(artifacts, artifacts_path)

    # Init live state (empty rolling prob buffer + last_processed_dt)
    live_state = {
        "prob_buffer": [],
        "last_processed_dt": None,  # ISO string
        "rolling_thr_window": cfg.rolling_thr_window,
        "target_alert_rate": cfg.target_alert_rate,
    }
    live_state_path = os.path.join(args.out_dir, "live_state.joblib")
    joblib.dump(live_state, live_state_path)

    print("\n[SAVED]")
    print(" model     :", model_path)
    print(" artifacts :", artifacts_path)
    print(" live_state:", live_state_path)


if __name__ == "__main__":
    main()
