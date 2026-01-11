# pip install pandas numpy scikit-learn tensorflow matplotlib

import numpy as np
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt

# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class Config:
    # XAU 5m (semicolon)
    xau_csv_path: str = "/content/drive/MyDrive/Colab Notebooks/XAU_5m_data.csv"  # <-- CHANGE
    xau_sep: str = ";"
    xau_date_col: str = "Date"
    xau_date_format: Optional[str] = "%Y.%m.%d %H:%M"  # e.g., 2004.06.11 07:15

    # DXY daily (comma)
    dxy_csv_path: str = "/content/drive/MyDrive/Colab Notebooks/DXY_merged_2013_2020_2021.csv"
    dxy_sep: str = ","
    dxy_date_col: str = "Date"
    dxy_date_format: Optional[str] = None
    dxy_value_col: str = "Close"

    # SPX daily (comma + thousands commas in numbers)
    spx_csv_path: str = "/content/drive/MyDrive/Colab Notebooks/SPX_2013_2021_merged.csv"
    spx_sep: str = ","
    spx_date_col: str = "Date"
    spx_date_format: Optional[str] = None
    spx_value_col: str = "Close"

    # Model horizons (bars)
    lookback: int = 96
    horizon_price: int = 1

    crash_k: int = 288          # 288 five-min bars ~= 1 day
    crash_theta: float = 0.025  # 2.5% drop

    # Training
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    seed: int = 42

    # Validation split from TRAIN (time-based)
    train_val_ratio: float = 0.80  # 80% train, 20% val from combined train events

    # Online thresholding
    # before it was 10
    target_alert_rate: float = 0.15
    rolling_thr_window: int = 500

    # Turn off flip by default (keep True if you want the old hack)
    allow_flip_if_val_auc_below_half: bool = False

cfg = Config()

tf.keras.utils.set_random_seed(cfg.seed)
np.random.seed(cfg.seed)

# -----------------------------
# 2) Crash windows (edit if needed)
# -----------------------------
EVENTS: List[Dict[str, str]] = [
    {"id": "2013_apr_crash", "start": "2013-04-11 00:00", "end": "2013-04-18 23:55"},
    {"id": "2020_mar_liquidation", "start": "2020-03-06 00:00", "end": "2020-03-24 23:55"},
    {"id": "2021_aug_flash", "start": "2021-08-06 00:00", "end": "2021-08-12 23:55"},
]

# FIXED split strategy:
TRAIN_EVENT_IDS = ["2013_apr_crash", "2020_mar_liquidation"]
TEST_EVENT_ID   = "2021_aug_flash"

# -----------------------------
# 3) Helpers
# -----------------------------
EPS = 1e-12

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
    Reads daily series; handles thousands commas (needed for your SPX).
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

def rolling_quantile_threshold(scores: np.ndarray, alert_rate: float, window: int) -> np.ndarray:
    """
    Online threshold: thr[t] uses only scores from (t-window .. t-1)
    """
    scores = np.asarray(scores, dtype=float)
    alert_rate = float(np.clip(alert_rate, 1e-4, 0.9999))
    thr = np.empty_like(scores, dtype=float)
    for t in range(len(scores)):
        start = max(0, t - window)
        past = scores[start:t]
        thr[t] = 1.0 if past.size < 10 else float(np.quantile(past, 1.0 - alert_rate))
    return thr

# -----------------------------
# 4) Load XAU 5m
# -----------------------------
df = pd.read_csv(cfg.xau_csv_path, sep=cfg.xau_sep, engine="python")
df = _clean_cols(df)

required = ["Open", "High", "Low", "Close", "Volume", cfg.xau_date_col]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"XAU: Missing columns {missing}. Found: {df.columns.tolist()}")

df[cfg.xau_date_col] = _parse_dt(df[cfg.xau_date_col], cfg.xau_date_format)
df = df.dropna(subset=[cfg.xau_date_col]).sort_values(cfg.xau_date_col)

for c in ["Open", "High", "Low", "Close", "Volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

# -----------------------------
# 5) Load DXY & SPX daily and merge into XAU 5m
# -----------------------------
dxy = load_daily_series(cfg.dxy_csv_path, cfg.dxy_sep, cfg.dxy_date_col, cfg.dxy_date_format, cfg.dxy_value_col, "DXY")
spx = load_daily_series(cfg.spx_csv_path, cfg.spx_sep, cfg.spx_date_col, cfg.spx_date_format, cfg.spx_value_col, "SPX")

df["date_only"] = df[cfg.xau_date_col].dt.floor("D")
df = df.merge(dxy, on="date_only", how="left").merge(spx, on="date_only", how="left")

df[["DXY_ret1d", "SPX_ret1d"]] = df[["DXY_ret1d", "SPX_ret1d"]].ffill().fillna(0.0)

# -----------------------------
# 6) Extract event windows
# -----------------------------
pieces = []
for ev in EVENTS:
    start = pd.to_datetime(ev["start"])
    end = pd.to_datetime(ev["end"])
    w = df[(df[cfg.xau_date_col] >= start) & (df[cfg.xau_date_col] <= end)].copy()
    if len(w) == 0:
        print(f"[WARN] No XAU rows for {ev['id']} in {ev['start']}..{ev['end']}")
        continue
    w["event_id"] = ev["id"]
    pieces.append(w)

if not pieces:
    raise RuntimeError("No event windows found. Check EVENTS date ranges vs your XAU timezone.")

df_ev = pd.concat(pieces, ignore_index=True).sort_values(["event_id", cfg.xau_date_col]).reset_index(drop=True)
print("[event rows]", df_ev.groupby("event_id").size().to_dict())

# -----------------------------
# 7) Build sequences per event
# -----------------------------
def build_sequences_for_event(g: pd.DataFrame):
    g = g.sort_values(cfg.xau_date_col).reset_index(drop=True)

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

    y_crash = make_crash_labels_within_window_fast(log_close, cfg.crash_k, cfg.crash_theta)

    Xs, yP, yC = [], [], []
    n = len(ret_close)
    last_t = n - max(cfg.horizon_price, cfg.crash_k) - 1
    for t in range(cfg.lookback, last_t + 1):
        Xs.append(X_feat[t - cfg.lookback:t])
        yP.append(ret_close[t + cfg.horizon_price])
        yC.append(y_crash[t])

    return np.array(Xs, np.float32), np.array(yP, np.float32), np.array(yC, np.float32)

seq_by_event = {}
for ev_id, g in df_ev.groupby("event_id", sort=False):
    Xs, yP, yC = build_sequences_for_event(g)
    if len(Xs) == 0:
        print(f"[WARN] {ev_id}: too short for lookback/crash_k. Expand the event window.")
        continue
    seq_by_event[ev_id] = (Xs, yP, yC)

# Ensure needed events exist
for need in TRAIN_EVENT_IDS + [TEST_EVENT_ID]:
    if need not in seq_by_event:
        raise RuntimeError(f"Missing sequences for event '{need}'. Check XAU coverage / event window.")

# Combine train events
X_train_all = np.concatenate([seq_by_event[e][0] for e in TRAIN_EVENT_IDS], axis=0)
yP_train_all = np.concatenate([seq_by_event[e][1] for e in TRAIN_EVENT_IDS], axis=0)
yC_train_all = np.concatenate([seq_by_event[e][2] for e in TRAIN_EVENT_IDS], axis=0)

# Test event
X_test = seq_by_event[TEST_EVENT_ID][0]
yP_test = seq_by_event[TEST_EVENT_ID][1]
yC_test = seq_by_event[TEST_EVENT_ID][2]

print(f"[sequences] train_total={len(X_train_all)} | train_crash_pos={int(yC_train_all.sum())} ({yC_train_all.mean()*100:.2f}%)")
print(f"[sequences] test_total={len(X_test)} | test_crash_pos={int(yC_test.sum())} ({yC_test.mean()*100:.2f}%)")

# -----------------------------
# 8) Time-based split TRAIN -> TRAIN/VAL (end chunk as VAL)
# -----------------------------
N = len(X_train_all)
cut = int(N * cfg.train_val_ratio)

X_train, X_val = X_train_all[:cut], X_train_all[cut:]
yP_train, yP_val = yP_train_all[:cut], yP_train_all[cut:]
yC_train, yC_val = yC_train_all[:cut], yC_train_all[cut:]

print("[split] train/val/test:", len(X_train), len(X_val), len(X_test))
print("[split positives] train/val/test:",
      int(yC_train.sum()), int(yC_val.sum()), int(yC_test.sum()))

# -----------------------------
# 9) Scaling
# -----------------------------
scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

def scale_X(X):
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

X_train_s, X_val_s, X_test_s = scale_X(X_train), scale_X(X_val), scale_X(X_test)

yP_mean = float(yP_train.mean())
yP_std  = float(yP_train.std() + 1e-12)

yP_train_s = ((yP_train - yP_mean) / yP_std).astype(np.float32)
yP_val_s   = ((yP_val   - yP_mean) / yP_std).astype(np.float32)
yP_test_s  = ((yP_test  - yP_mean) / yP_std).astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((X_train_s, {"price": yP_train_s, "crash": yC_train}))\
    .shuffle(4096, seed=cfg.seed, reshuffle_each_iteration=True)\
    .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val_s, {"price": yP_val_s, "crash": yC_val}))\
    .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test_s, {"price": yP_test_s, "crash": yC_test}))\
    .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 10) Model (slightly smaller = less overfit on small event dataset)
# -----------------------------
def tcn_block(x, filters, kernel_size, dilation_rate, dropout=0.20):
    conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                  dilation_rate=dilation_rate, activation="relu")(x)
    conv = tf.keras.layers.Dropout(dropout)(conv)
    conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                  dilation_rate=dilation_rate, activation="relu")(conv)
    res = x if x.shape[-1] == filters else tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
    out = tf.keras.layers.Add()([res, conv])
    return tf.keras.layers.Activation("relu")(out)

def build_model(n_features):
    inp = tf.keras.Input(shape=(cfg.lookback, n_features))
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

model = build_model(X_train_s.shape[-1])

# Weighted BCE
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

# -----------------------------
# 11) Predict
# -----------------------------
pred_val = model.predict(val_ds, verbose=0)
p_val = pred_val["crash"].reshape(-1)
y_val = yC_val.astype(int).reshape(-1)

pred_test = model.predict(test_ds, verbose=0)
p_test = pred_test["crash"].reshape(-1)
y_test = yC_test.astype(int).reshape(-1)

# Optional flip (OFF by default)
if cfg.allow_flip_if_val_auc_below_half and len(np.unique(y_val)) == 2:
    auc_val = roc_auc_score(y_val, p_val)
    print(f"[val] ROC-AUC before flip: {auc_val:.4f}")
    if auc_val < 0.5:
        print("[flip] VAL AUC < 0.5 -> flipping probabilities (p := 1-p)")
        p_val = 1.0 - p_val
        p_test = 1.0 - p_test

# Regression (unstandardize)
yP_pred_test = pred_test["price"].reshape(-1) * yP_std + yP_mean
yP_true_test = yP_test.reshape(-1)

# -----------------------------
# 12) ONLINE thresholding (VAL stats + TEST evaluation)
# -----------------------------
thr_val_series = rolling_quantile_threshold(p_val, cfg.target_alert_rate, cfg.rolling_thr_window)
y_hat_val = (p_val >= thr_val_series).astype(int)
prec_v, rec_v, f1_v, _ = precision_recall_fscore_support(y_val, y_hat_val, average="binary", zero_division=0)

print("\n[online thr] VAL:",
      {"rolling_window": cfg.rolling_thr_window,
       "target_alert_rate": cfg.target_alert_rate,
       "val_alert_rate": float(y_hat_val.mean()),
       "val_prec": float(prec_v),
       "val_rec": float(rec_v),
       "val_f1": float(f1_v)})

thr_test_series = rolling_quantile_threshold(p_test, cfg.target_alert_rate, cfg.rolling_thr_window)
y_hat_test = (p_test >= thr_test_series).astype(int)

# -----------------------------
# 13) Metrics (TEST)
# -----------------------------
rmse_ret = np.sqrt(mean_squared_error(yP_true_test, yP_pred_test))
mae_ret  = mean_absolute_error(yP_true_test, yP_pred_test)

print("\n--- Price Forecast (Test) [log-return] ---")
print(f"RMSE: {rmse_ret:.6f}")
print(f"MAE : {mae_ret:.6f}")

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_hat_test, average="binary", zero_division=0)
cm = confusion_matrix(y_test, y_hat_test)

rocauc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) == 2 else np.nan
prauc  = average_precision_score(y_test, p_test) if len(np.unique(y_test)) == 2 else np.nan
mcc    = matthews_corrcoef(y_test, y_hat_test) if len(np.unique(y_test)) == 2 else np.nan

print("\n--- Crash Warning (Test) ---")
print(f"Definition: >= {cfg.crash_theta*100:.1f}% drop within next {cfg.crash_k} bars")
print(f"Test positives: {int(y_test.sum())}/{len(y_test)} ({y_test.mean()*100:.2f}%)")
print(f"Online threshold: rolling_window={cfg.rolling_thr_window}, target_alert_rate={cfg.target_alert_rate} | test_alert_rate: {y_hat_test.mean():.3f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {rocauc:.4f}")
print(f"PR-AUC   : {prauc:.4f}")
print(f"MCC      : {mcc:.4f}")
print("Confusion Matrix:\n", cm)

# -----------------------------
# 14) Plots
# -----------------------------
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training Curve (Total Loss)")
plt.show()

plt.figure()
plt.plot(p_test, label="Crash prob (test)")
plt.plot(thr_test_series, label="Online thr (test)")
plt.plot(y_test, label="True crash (test)")
plt.legend(); plt.title("Crash prob vs online threshold (test)")
plt.show()