# Dual-Horizon LSTM–TCN (XAU 5m) — Train + Live CSV Inference

This repo contains **two clean scripts** to train and run a **dual-head deep model** for XAUUSD (gold) on **5-minute bars**:

- **Price head (regression):** predicts next-bar **log-return** (`t+1`)
- **Crash head (classification):** predicts **crash risk** at time `t`, defined as:
  > “Price drops by ≥ `theta` within the next `k` 5-minute bars”

It also includes **online thresholding** to keep alert frequency stable by targeting a chosen alert rate.

---

## 1) Files

- `train.py`
  - Loads XAU 5m + daily DXY/SPX
  - Builds features + sequences
  - Trains the TCN+LSTM multi-task model
  - Saves model + scaler + normalization stats + initial live state

- `predict_live.py`
  - Loads trained model + artifacts
  - Polls your **5-minute CSV** for new bars
  - Produces a signal for each new bar using only *past* data (no leakage)
  - Maintains and persists the online threshold state across restarts

---

## 2) Requirements

Python 3.9+ recommended.

Install:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib joblib
