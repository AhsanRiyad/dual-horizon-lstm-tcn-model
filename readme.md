# Dual-Horizon LSTM–TCN (XAUUSD 5m) — Train + Live CSV Inference

This project trains and runs a **dual-head** deep model on **5-minute XAUUSD bars** plus daily macro proxies (DXY, SPX):

- **Price head (regression):** predicts next-bar **log-return** (*t+1*).
- **Crash head (classification):** predicts probability of a **downside crash** within the next **k** bars.
- **Online alerting:** uses a rolling quantile threshold to target a stable alert rate (instead of fixed 0.5).

## Files

- `train.py`  
  Trains the model and saves all artifacts needed for production inference.
- `predict_live.py`  
  Reads an **updating 5m CSV**, detects new bars, outputs signals, and persists online-threshold state across restarts.

## Requirements

Python 3.9+ recommended.

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib joblib
```

## Data formats

### 1) XAU 5-minute CSV (updated over time)
Required columns:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

Common settings:
- separator: `;`
- date format: `%Y.%m.%d %H:%M` (example: `2004.06.11 07:15`)

### 2) DXY daily CSV
Required columns: `Date`, `Close`

### 3) SPX daily CSV
Required columns: `Date`, `Close`  
(Thousands separators are handled automatically.)

## Features used (fixed order)

The model expects features in this exact order:

1. `ret_close` (log-return close)  
2. `ret_open` (log-return open)  
3. `hl_range` = (High − Low) / Close  
4. `oc_change` = (Close − Open) / Open  
5. `log_close`  
6. `log_vol`  
7. `vol_20`  (rolling std of `ret_close`)  
8. `vol_96`  
9. `vol_288`  
10. `mom_20` (rolling mean of `ret_close`)  
11. `mom_96`  
12. `DXY_ret1d` (daily log-return merged to intraday)  
13. `SPX_ret1d`

## Crash definition (label)

Crash at time *t* is **1** if within the next `crash_k` 5-minute bars, price drops by at least `crash_theta` at any point.

Defaults:
- `crash_k = 288` (≈ 1 day)
- `crash_theta = 0.025` (2.5%)

## Online thresholding (alerts)

Instead of a fixed threshold, the live script maintains a rolling buffer of past crash probabilities and sets:

- `threshold(t) = quantile(buffer, 1 - target_alert_rate)`

Defaults:
- `target_alert_rate = 0.15`
- `rolling_thr_window = 500`

Warm-up behavior:
- if the buffer has < 10 values → threshold = 1.0 (no alerts)

## Important: daily macro look-ahead (recommended)

Daily close returns for “today” are not truly known intraday. To reduce look-ahead bias, run both training and live inference with:

- `--macro_shift_days 1`

This shifts daily returns so intraday bars effectively use **yesterday’s** daily returns.

## How to run

### 1) Train

```bash
python train.py \
  --xau_csv "/path/XAU_5m_data.csv" --xau_sep ";" --xau_date_format "%Y.%m.%d %H:%M" \
  --dxy_csv "/path/DXY_daily.csv" \
  --spx_csv "/path/SPX_daily.csv" \
  --out_dir "./out" \
  --macro_shift_days 1
```

Outputs in `./out/`:
- `dual_horizon_lstm_tcn.keras` (model)
- `artifacts.joblib` (scaler, yP_mean/yP_std, config snapshot, feature_order)
- `live_state.joblib` (online threshold state)

### 2) Live inference (continuous polling)

```bash
python predict_live.py \
  --xau_csv "/path/XAU_5m_data.csv" --xau_sep ";" --xau_date_format "%Y.%m.%d %H:%M" \
  --dxy_csv "/path/DXY_daily.csv" \
  --spx_csv "/path/SPX_daily.csv" \
  --model "./out/dual_horizon_lstm_tcn.keras" \
  --artifacts "./out/artifacts.joblib" \
  --state "./out/live_state.joblib" \
  --poll_seconds 15 \
  --macro_shift_days 1
```

### 3) Live inference (one-shot)

```bash
python predict_live.py ... --once
```

## No-leakage timing (real scenario)

For a newly arrived bar at index **i**, `predict_live.py` builds the input window from bars:

- `[i - lookback, ..., i - 1]`

So the signal is generated **without using bar i as input**.

## Output meaning

Example log line:

```
[2026-01-11 10:35:00] crash_prob=0.2310 thr=0.1985 ALERT | pred_ret=+0.000412
```

- `crash_prob`: model probability of a crash event within the next `crash_k` bars  
- `thr`: online threshold at that time  
- `ALERT`: `crash_prob >= thr`  
- `pred_ret`: predicted next-bar log-return

Approx. price projection:
- `Close_pred ≈ Close_now * exp(pred_ret)`

## Key knobs (what you’ll tune)

- `lookback` (default 96): history length (96×5m = 8 hours)
- `crash_k`, `crash_theta`: definition of crash
- `target_alert_rate`: how often alerts trigger on average
- `rolling_thr_window`: smoothness of online threshold
- `macro_shift_days`: set to 1 for more realistic macro usage

## Troubleshooting

- **Missing columns**: XAU must include `Date Open High Low Close Volume`.
- **Not enough rows**: need at least `lookback + 2` rows.
- **No new rows detected**: your CSV may be rewritten without new timestamps or has duplicates.
- **Macro values all 0/NaN**: check timezone/date alignment between XAU `Date` and daily `Date`.

## Disclaimer

This code is for research/engineering. Trading is risky; validate thoroughly and apply risk controls.
