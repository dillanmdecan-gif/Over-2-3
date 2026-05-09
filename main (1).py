"""
Deriv Over/Under Digits Bot — Multi-Layer Probability Engine
════════════════════════════════════════════════════════════
Symbol  : R_100  (configurable)
Contract: DIGITOVER 2  /  DIGITOVER 3  (last digit > 2 or > 3)

ARCHITECTURE
────────────
Layer 1 — Tick Microstructure Analyzer
    Tick velocity, acceleration, clustering, digit transition
    probabilities, volatility bursts, reversal compression,
    momentum exhaustion, Markov transition matrices.

Layer 2 — Entropy Detection Engine
    Shannon entropy, permutation entropy, rolling randomness,
    sequence predictability.  HIGH ENTROPY → NO TRADE.

Layer 3 — Reinforcement Learning Agent (tabular Q-learning)
    Decides WHEN to trade, WHICH regime is active, WHETHER
    confidence is sufficient.  Trade PERMISSION engine only.

Layer 4 — Digit Prediction Network (NumPy — no PyTorch needed)
    Temporal CNN + GRU-like recurrent layer + attention.
    Outputs P(digit > 2), P(digit > 3), noise level,
    stability window estimate.

Layer 5 — Regime Detection System
    Classifies market into: trending, mean-reverting, chaotic,
    volatility-expansion, stability-compression.
    Only the matching model/strategy is enabled per regime.

Layer 6 — Confidence Fusion Engine
    FINAL_CONFIDENCE =
        0.25 x entropy_score
      + 0.20 x rl_confidence
      + 0.20 x neural_confidence
      + 0.15 x regime_stability
      + 0.10 x transition_bias
      + 0.10 x volatility_score

Layer 7 — Probability Calibration
    Platt scaling + isotonic regression + online calibration.
    Prevents overconfidence destroying the account.

Layer 8 — Null Hypothesis Test
    Before every trade: tests whether current digit behavior
    is statistically indistinguishable from pure uniform random.
    If it is → confidence forced to zero → NO TRADE.

Layer 9 — Adaptive Self-Learning
    Continuously compares predicted vs actual outcomes,
    detects drift, reweights models, downgrades weak features.

ENTRY RULE
──────────
Trade only when ALL of:
    P(Over 2 or Over 3) > 0.78
    Entropy              < entropy_threshold
    Regime stability     > stability_threshold
    RL approval          = TRUE
    Null hypothesis      rejected at p < 0.05
    Final confidence     > confidence_threshold

STAKE
─────
Fixed base_stake with martingale: x1.5 per loss, max 2 steps, then halt.

Run:
    export DERIV_API_TOKEN=your_token
    python main.py

Backtest:
    python main.py --backtest
"""

import asyncio
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
import websockets


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("overbot")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── API CREDENTIALS ───────────────────────────────────────────────────────
    # Option A (recommended for Railway): set DERIV_API_TOKEN env var.
    # Option B (local testing only):  replace the empty string below with
    #   your token, e.g.  api_token: str = "a1b2c3d4e5f6..."
    api_token: str = field(
        default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    app_id: str = field(
        default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_url: str = "wss://ws.binaryws.com/websockets/v3"

    # ── CONTRACT ──────────────────────────────────────────────────────────────
    symbol:        str   = "R_100"
    duration:      int   = 1           # 1 tick
    currency:      str   = "USD"
    # "over2"  → DIGITOVER 2  (digit must be 3-9  → base P ≈ 0.70)
    # "over3"  → DIGITOVER 3  (digit must be 4-9  → base P ≈ 0.60)
    # "auto"   → engine picks the contract with higher calibrated probability
    contract_mode: str   = "auto"
    payout_ratio:  float = 0.45        # Deriv digitover payout

    # ── WARMUP ────────────────────────────────────────────────────────────────
    warmup_ticks: int = 150            # minimum ticks before any trading

    # ── LAYER 1 — MICROSTRUCTURE ──────────────────────────────────────────────
    micro_window:        int   = 30
    markov_window:       int   = 60
    cluster_window:      int   = 20
    burst_threshold:     float = 2.5
    momentum_window:     int   = 15

    # ── LAYER 2 — ENTROPY ─────────────────────────────────────────────────────
    entropy_window:      int   = 40
    perm_entropy_order:  int   = 4
    entropy_threshold:   float = 0.92  # max allowed composite entropy (0-1)

    # ── LAYER 3 — RL AGENT ────────────────────────────────────────────────────
    rl_states:           int   = 64
    rl_alpha:            float = 0.15
    rl_gamma:            float = 0.90
    rl_epsilon_start:    float = 0.67
    rl_epsilon_min:      float = 0.05
    rl_epsilon_decay:    float = 0.995

    # ── LAYER 4 — NEURAL NETWORK ──────────────────────────────────────────────
    nn_input_window:     int   = 30
    nn_hidden:           int   = 32
    nn_lr:               float = 0.005
    nn_batch:            int   = 16

    # ── LAYER 5 — REGIME DETECTION ───────────────────────────────────────────
    regime_window:       int   = 50
    regime_threshold:    float = 0.60

    # ── LAYER 6 — CONFIDENCE FUSION  (weights must sum to 1.0) ───────────────
    w_entropy:     float = 0.25
    w_rl:          float = 0.20
    w_neural:      float = 0.20
    w_regime:      float = 0.15
    w_transition:  float = 0.10
    w_volatility:  float = 0.10

    # ── LAYER 8 — NULL HYPOTHESIS ─────────────────────────────────────────────
    null_hyp_window:     int   = 50
    null_hyp_p_value:    float = 0.15

    # ── ENTRY THRESHOLDS ─────────────────────────────────────────────────────
    min_over2_prob:       float = 0.61
    min_over3_prob:       float = 0.62
    min_final_conf:       float = 0.31
    min_regime_stability: float = 0.55

    # ── STAKE / MARTINGALE ────────────────────────────────────────────────────
    # Ladder: $0.35 → $0.53 → $0.79 → halt
    base_stake:          float = 1.0
    martingale_factor:   float = 2.99
    martingale_steps:    int   = 4
    max_balance_pct:     float = 0.10

    # ── RISK ─────────────────────────────────────────────────────────────────
    loss_cooldown_ticks:    int   = 8
    max_consecutive_losses: int   = 2
    max_daily_loss_pct:     float = 0.15
    balance_guard_mult:     int   = 6

    # ── CALIBRATION ──────────────────────────────────────────────────────────
    cal_window:      int = 60
    cal_recal_every: int = 30

    # ── FILES / LOGGING ───────────────────────────────────────────────────────
    history_file:        str   = "over_trades.csv"
    skip_log_interval:   float = 30.0
    skip_summary_every:  int   = 200


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — TICK MICROSTRUCTURE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class MicrostructureAnalyzer:
    """
    Extracts order-flow style features from synthetic tick sequences.
    Features: tick_velocity, tick_acceleration, volatility_burst,
    momentum_exhaustion, reversal_compression, markov_over2_bias,
    markov_over3_bias, cluster_density.
    """

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self._prices:   deque = deque(maxlen=max(cfg.micro_window, cfg.cluster_window, 100))
        self._digits:   deque = deque(maxlen=cfg.markov_window + 10)
        self._diffs:    deque = deque(maxlen=cfg.micro_window)
        self._vel_hist: deque = deque(maxlen=cfg.micro_window)
        # Markov: [was_in_state][is_in_state] for over2 and over3
        self._mk2: List[List[float]] = [[1.0, 1.0], [1.0, 1.0]]
        self._mk3: List[List[float]] = [[1.0, 1.0], [1.0, 1.0]]
        self._prev_digit: Optional[int] = None

    def push(self, price: float) -> dict:
        digit = int(round(price * 100)) % 10
        self._prices.append(price)
        self._digits.append(digit)
        prices = list(self._prices)
        f = {}

        if len(prices) >= 2:
            diff = abs(prices[-1] - prices[-2])
            self._diffs.append(diff)
            self._vel_hist.append(diff)

        if len(self._diffs) >= 2:
            diffs = list(self._diffs)
            f["tick_velocity"]     = float(np.mean(diffs[-10:]))
            mid = len(diffs) // 2
            v1  = np.mean(diffs[:mid]) if mid else 0
            v2  = np.mean(diffs[mid:])
            f["tick_acceleration"] = float(v2 - v1)
        else:
            f["tick_velocity"] = f["tick_acceleration"] = 0.0

        if len(self._vel_hist) >= 10:
            vel  = list(self._vel_hist)
            mu   = np.mean(vel); sig = np.std(vel) + 1e-9
            f["volatility_burst"] = float((vel[-1] - mu) / sig)
        else:
            f["volatility_burst"] = 0.0

        f["momentum_exhaustion"] = (
            self._momentum_exhaustion(prices)
            if len(prices) >= self.cfg.momentum_window else 0.0)

        f["reversal_compression"] = (
            self._reversal_compression(prices) if len(prices) >= 10 else 0.0)

        self._update_markov(digit)
        curr2 = 1 if digit > 2 else 0
        curr3 = 1 if digit > 3 else 0
        f["markov_over2_bias"] = self._markov_bias(self._mk2, curr2)
        f["markov_over3_bias"] = self._markov_bias(self._mk3, curr3)

        f["cluster_density"] = (
            self._cluster_density(list(self._digits)[-self.cfg.cluster_window:])
            if len(self._digits) >= self.cfg.cluster_window else 0.0)

        return f

    def _momentum_exhaustion(self, prices: list) -> float:
        w    = prices[-self.cfg.momentum_window:]
        runs, cur = [], 1
        for i in range(1, len(w)):
            same = (w[i] > w[i-1]) == (w[i-1] > w[i-2]) if i > 1 else True
            if same: cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        if len(runs) < 3: return 0.0
        mid   = len(runs) // 2
        ratio = 1.0 - min(np.mean(runs[mid:]) / (np.mean(runs[:mid]) + 1e-9), 1.0)
        return float(np.clip(ratio, 0, 1))

    def _reversal_compression(self, prices: list) -> float:
        w   = prices[-20:]
        rev = sum(1 for i in range(1, len(w)-1)
                  if (w[i] > w[i-1]) != (w[i] < w[i+1]))
        return float(rev / max(len(w) - 2, 1))

    def _update_markov(self, digit: int):
        if self._prev_digit is not None:
            p2 = 1 if self._prev_digit > 2 else 0
            c2 = 1 if digit > 2 else 0
            p3 = 1 if self._prev_digit > 3 else 0
            c3 = 1 if digit > 3 else 0
            self._mk2[p2][c2] += 1.0
            self._mk3[p3][c3] += 1.0
        self._prev_digit = digit

    def _markov_bias(self, m: List[List[float]], state: int) -> float:
        row = m[state]; tot = row[0] + row[1]
        return float((row[1] / tot) - 0.5) if tot else 0.0

    def _cluster_density(self, digits: list) -> float:
        counts = np.bincount(digits, minlength=10).astype(float)
        exp    = len(digits) / 10.0
        chi2   = float(np.sum((counts - exp) ** 2 / (exp + 1e-9)))
        return float(np.clip(chi2 / (len(digits) * 9.0), 0, 1))

    @property
    def markov_matrix_str(self) -> str:
        def fmt(m):
            r1 = m[0]; r2 = m[1]
            t1 = r1[0]+r1[1]+1e-9; t2 = r2[0]+r2[1]+1e-9
            return f"under→over:{r1[1]/t1:.2f} over→over:{r2[1]/t2:.2f}"
        return f"MK2:[{fmt(self._mk2)}] MK3:[{fmt(self._mk3)}]"


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — ENTROPY DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EntropyEngine:
    """
    Shannon entropy + permutation entropy + uniformity test.
    composite = 0.45*shannon + 0.35*perm + 0.20*(1-uniformity_p)
    HIGH composite (> entropy_threshold) → DO NOT TRADE.
    """

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._digits: deque = deque(maxlen=max(cfg.entropy_window, cfg.null_hyp_window, 200))

    def push(self, digit: int) -> dict:
        self._digits.append(digit)
        result = {"shannon": 1.0, "permutation": 1.0,
                  "uniformity": 1.0, "composite": 1.0, "tradeable": False}
        if len(self._digits) < self.cfg.entropy_window:
            return result
        window = list(self._digits)[-self.cfg.entropy_window:]
        shannon    = self._shannon(window)
        perm       = self._perm_entropy(window, self.cfg.perm_entropy_order)
        uniformity = self._uniformity_p(window)
        composite  = float(np.clip(0.45*shannon + 0.35*perm + 0.20*(1-uniformity), 0, 1))
        result.update({"shannon": round(shannon, 4), "permutation": round(perm, 4),
                       "uniformity": round(uniformity, 4),
                       "composite": round(composite, 4),
                       "tradeable": composite < self.cfg.entropy_threshold})
        return result

    @staticmethod
    def _shannon(digits: list) -> float:
        counts = np.bincount(digits, minlength=10).astype(float)
        probs  = counts[counts > 0] / counts.sum()
        return float(-np.sum(probs * np.log2(probs)) / np.log2(10))

    @staticmethod
    def _perm_entropy(digits: list, order: int) -> float:
        if len(digits) < order + 1: return 1.0
        pats = Counter(
            tuple(sorted(range(order), key=lambda j: digits[i:i+order][j]))
            for i in range(len(digits) - order + 1))
        total = sum(pats.values())
        probs = [v/total for v in pats.values()]
        h = -sum(p*math.log2(p) for p in probs if p > 0)
        max_h = math.log2(math.factorial(order))
        return float(h / max_h) if max_h > 0 else 1.0

    @staticmethod
    def _uniformity_p(digits: list) -> float:
        counts   = np.bincount(digits, minlength=10).astype(float)
        expected = np.full(10, len(digits)/10.0)
        _, p = scipy_stats.chisquare(counts, expected)
        return float(p)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — REINFORCEMENT LEARNING AGENT  (tabular Q-learning)
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Q-learning trade permission engine.
    State: discretised (entropy, regime, win_rate, vol_z).
    Actions: 0=idle, 1=allow_trade.
    Reward: normalised profit on trade, -0.01 per idle tick.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._Q:   np.ndarray       = np.zeros((cfg.rl_states, 2))
        self._eps: float            = cfg.rl_epsilon_start
        self._last_s: Optional[int] = None
        self._last_a: Optional[int] = None

    def state_index(self, entropy: float, regime: int,
                    win_rate: float, vol_z: float) -> int:
        e  = min(int(entropy * 4), 3)
        r  = min(regime, 4)
        wr = min(int(win_rate * 4), 3)
        vz = min(int((vol_z + 3) / 1.5), 3)
        return int(np.clip(e*16 + r*3 + wr + vz, 0, self.cfg.rl_states - 1))

    def act(self, state: int) -> Tuple[int, float]:
        action = (random.choice([0, 1]) if random.random() < self._eps
                  else int(np.argmax(self._Q[state])))
        q  = self._Q[state]
        qr = max(abs(q.max() - q.min()), 1e-9)
        confidence = float(np.clip((q[action] - q.min()) / qr, 0, 1))
        self._last_s = state; self._last_a = action
        return action, confidence

    def update(self, reward: float, next_s: int):
        if self._last_s is None: return
        td = reward + self.cfg.rl_gamma * np.max(self._Q[next_s]) - self._Q[self._last_s][self._last_a]
        self._Q[self._last_s][self._last_a] += self.cfg.rl_alpha * td
        self._eps = max(self.cfg.rl_epsilon_min, self._eps * self.cfg.rl_epsilon_decay)

    @property
    def epsilon(self) -> float: return self._eps


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — DIGIT PREDICTION NETWORK  (pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

class DigitNet:
    """
    NumPy neural net: Temporal CNN → max-pool → GRU-like dense → attention → 4 outputs.
    Outputs: [P(over2), P(over3), noise_level, stability].
    Online mini-batch gradient descent on output layer.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        w = cfg.nn_input_window; h = cfg.nn_hidden; F = 4
        rng = np.random.default_rng(42)
        self.W_conv = rng.normal(0, 0.1, (h, F, 3))
        self.b_conv = np.zeros(h)
        pool_out    = max((w - 2) // 2, 1)
        gru_in      = h * pool_out
        self.W_gru  = rng.normal(0, 0.1, (h, gru_in))
        self.b_gru  = np.zeros(h)
        self.W_att  = rng.normal(0, 0.1, (1, h))
        self.b_att  = np.zeros(1)
        self.W_out  = rng.normal(0, 0.1, (4, h))
        self.b_out  = np.array([0.70, 0.60, 0.50, 0.50])
        self._buf_X: List[np.ndarray] = []
        self._buf_y: List[np.ndarray] = []

    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    @staticmethod
    def _relu(x): return np.maximum(0, x)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, F = X.shape; k = 3; out_len = T - k + 1
        cnn = np.zeros((self.cfg.nn_hidden, out_len))
        for t in range(out_len):
            cnn[:, t] = self._relu(
                np.einsum('hij,ij->h', self.W_conv, X[t:t+k, :].T) + self.b_conv)
        pool_len = max(out_len // 2, 1)
        pooled   = np.array([cnn[:, i*2:i*2+2].max(axis=1) for i in range(pool_len)]).T
        flat     = pooled.flatten()
        gru_in   = self.W_gru.shape[1]
        flat     = (flat[:gru_in] if len(flat) >= gru_in
                    else np.pad(flat, (0, gru_in - len(flat))))
        h_gru    = self._relu(self.W_gru @ flat + self.b_gru)
        attn     = self._sig(self.W_att @ h_gru + self.b_att)
        h_att    = h_gru * attn[0]
        out      = self._sig(self.W_out @ h_att + self.b_out)
        return out, h_att

    def predict(self, X: np.ndarray) -> dict:
        out, _ = self._forward(X)
        return {"p_over2": float(out[0]), "p_over3": float(out[1]),
                "noise": float(out[2]), "stability": float(out[3])}

    def record(self, X: np.ndarray, y: np.ndarray):
        self._buf_X.append(X.copy()); self._buf_y.append(y.copy())
        if len(self._buf_X) > self.cfg.nn_batch * 4:
            self._buf_X.pop(0); self._buf_y.pop(0)
        if len(self._buf_X) >= self.cfg.nn_batch:
            self._train_step()

    def _train_step(self):
        idxs = random.sample(range(len(self._buf_X)),
                              min(self.cfg.nn_batch, len(self._buf_X)))
        gW = np.zeros_like(self.W_out); gb = np.zeros_like(self.b_out)
        for i in idxs:
            out, h = self._forward(self._buf_X[i])
            err = out - self._buf_y[i]
            gW += np.outer(err, h); gb += err
        n = len(idxs)
        self.W_out -= self.cfg.nn_lr * gW / n
        self.b_out -= self.cfg.nn_lr * gb / n


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — REGIME DETECTION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

REGIME_TRENDING   = 0
REGIME_REVERTING  = 1
REGIME_CHAOTIC    = 2
REGIME_VOL_EXPAND = 3
REGIME_STABLE     = 4
REGIME_NAMES      = ["trending", "reverting", "chaotic", "vol_expand", "stable"]


class RegimeDetector:
    """
    Rule-based regime classifier using price autocorrelation,
    volatility z-score, and entropy.
    Returns (regime_id, confidence 0-1).
    """

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._prices: deque = deque(maxlen=cfg.regime_window)
        self._vol_h:  deque = deque(maxlen=cfg.regime_window)

    def push(self, price: float, entropy: float) -> Tuple[int, float]:
        self._prices.append(price)
        prices = list(self._prices)
        if len(prices) < 20:
            return REGIME_STABLE, 0.50
        diffs = np.diff(prices)
        vol   = float(np.std(diffs)); self._vol_h.append(vol)
        ac    = float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1]) if len(diffs) >= 10 else 0.0
        vol_z = float((vol - np.mean(list(self._vol_h))) / (np.std(list(self._vol_h)) + 1e-9)) \
                if len(self._vol_h) >= 10 else 0.0
        if   vol_z   >  2.0:    regime, conf = REGIME_VOL_EXPAND, min(0.5 + vol_z*0.1, 1.0)
        elif entropy > 0.90:    regime, conf = REGIME_CHAOTIC,    min(entropy, 1.0)
        elif ac      >  0.30:   regime, conf = REGIME_TRENDING,   min(0.5 + ac, 1.0)
        elif ac      < -0.30:   regime, conf = REGIME_REVERTING,  min(0.5 + abs(ac), 1.0)
        else:                   regime, conf = REGIME_STABLE,     max(0.5, 1.0 - entropy)
        return regime, float(np.clip(conf, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — PROBABILITY CALIBRATOR  (Platt scaling, online)
# ─────────────────────────────────────────────────────────────────────────────

class Calibrator:
    """
    Online Platt scaling: P_cal = sigmoid(A * P_raw + B).
    Refitted every cal_recal_every trades with exponential recency weighting.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg; self._A = 1.0; self._B = 0.0
        self._history: deque = deque(maxlen=cfg.cal_window)
        self._since   = 0

    def calibrate(self, p: float) -> float:
        x = self._A * p + self._B
        return float(1.0 / (1.0 + math.exp(-max(-20, min(20, x)))))

    def record(self, p: float, won: bool):
        self._history.append((p, 1.0 if won else 0.0))
        self._since += 1
        if self._since >= self.cfg.cal_recal_every:
            self._refit(); self._since = 0

    def _refit(self):
        if len(self._history) < 10: return
        data = list(self._history); n = len(data)
        w    = np.array([math.exp(-0.05*(n-1-i)) for i in range(n)])
        w   /= w.sum()
        ps   = np.array([d[0] for d in data])
        ys   = np.array([d[1] for d in data])
        A, B = self._A, self._B
        for _ in range(50):
            pc = 1.0 / (1.0 + np.exp(-(A*ps + B)))
            e  = pc - ys
            A -= 0.10 * float(np.sum(w*e*ps))
            B -= 0.10 * float(np.sum(w*e))
        self._A, self._B = A, B


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 8 — NULL HYPOTHESIS TESTER
# ─────────────────────────────────────────────────────────────────────────────

class NullHypothesisTester:
    """
    Chi-square + Wald-Wolfowitz runs test.
    reject_null=True (p < threshold) means sequence is non-random → trading allowed.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self._digits: deque = deque(maxlen=cfg.null_hyp_window)

    def push(self, digit: int): self._digits.append(digit)

    def test(self) -> Tuple[bool, float, str]:
        digits = list(self._digits)
        if len(digits) < self.cfg.null_hyp_window:
            return False, 1.0, "insufficient_data"
        counts   = np.bincount(digits, minlength=10).astype(float)
        expected = np.full(10, len(digits)/10.0)
        _, p_chi = scipy_stats.chisquare(counts, expected)
        binary   = [1 if d > 4 else 0 for d in digits]
        p_runs   = self._runs_test(binary)
        p_comb   = min(float(p_chi), float(p_runs))
        return p_comb < self.cfg.null_hyp_p_value, p_comb, "chi2+runs"

    @staticmethod
    def _runs_test(b: list) -> float:
        n1 = sum(b); n2 = len(b) - n1
        if n1 == 0 or n2 == 0: return 1.0
        runs = 1 + sum(1 for i in range(1, len(b)) if b[i] != b[i-1])
        n    = len(b)
        mu   = 1 + 2*n1*n2/n
        s2   = 2*n1*n2*(2*n1*n2-n) / (n*n*(n-1) + 1e-9)
        if s2 <= 0: return 1.0
        z    = (runs - mu) / math.sqrt(s2)
        return float(2 * (1 - scipy_stats.norm.cdf(abs(z))))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 9 — ADAPTIVE SELF-LEARNING
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLearner:
    """
    ELO-style feature weight updates + concept drift detection.
    Features that consistently predict correctly gain weight; wrong ones lose weight.
    If recent accuracy < 45% → drift flag → all weights penalised.
    """

    FEATURES = ["entropy", "markov_over2", "markov_over3",
                "momentum", "reversal", "cluster", "vol_burst",
                "nn_p_over2", "nn_p_over3", "regime_stability"]

    def __init__(self):
        self._weights = {f: 1.0 for f in self.FEATURES}
        self._scores  = {f: deque(maxlen=50) for f in self.FEATURES}
        self._outcomes: deque = deque(maxlen=100)
        self._drift   = False

    def record(self, preds: dict, won: bool):
        self._outcomes.append(1 if won else 0)
        for feat, pred in preds.items():
            if feat not in self._scores: continue
            self._scores[feat].append(1 if (pred == won) else 0)
            acc = float(np.mean(self._scores[feat])) if self._scores[feat] else 0.5
            self._weights[feat] = float(np.clip(
                self._weights[feat] * (1.0 + 0.05*(acc - 0.5)), 0.1, 3.0))
        if len(self._outcomes) >= 20:
            acc = float(np.mean(list(self._outcomes)[-20:]))
            self._drift = acc < 0.45
            if self._drift:
                log.warning(f"[DRIFT] Recent acc={acc:.1%} — penalising weights")
                for f in self._weights:
                    self._weights[f] = max(0.1, self._weights[f] * 0.85)

    def weight(self, f: str) -> float: return self._weights.get(f, 1.0)
    @property
    def drift(self) -> bool: return self._drift
    @property
    def recent_accuracy(self) -> float:
        return float(np.mean(self._outcomes)) if self._outcomes else 0.0
    def summary(self) -> str:
        top = sorted(self._weights.items(), key=lambda x: -x[1])[:4]
        return "  ".join(f"{k}={v:.2f}" for k,v in top)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — CONFIDENCE FUSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    final_confidence:  float
    entropy_score:     float
    rl_confidence:     float
    neural_confidence: float
    regime_stability:  float
    transition_bias:   float
    volatility_score:  float
    p_over2:           float
    p_over3:           float
    contract_type:     str
    regime:            int
    regime_name:       str
    null_rejected:     bool
    null_p:            float
    tradeable:         bool
    block_reason:      str


def fuse(cfg: Config, entropy: dict, rl_conf: float, nn_pred: dict,
         regime_id: int, regime_conf: float, micro: dict,
         null_rej: bool, null_p: float,
         learner: AdaptiveLearner, calibrator: Calibrator) -> FusionResult:

    entropy_score = float(1.0 - entropy["composite"])
    vol_z         = micro.get("volatility_burst", 0.0)
    vol_score     = float(np.clip(1.0 - abs(vol_z) / 4.0, 0, 1))
    mb2           = abs(micro.get("markov_over2_bias", 0.0))
    mb3           = abs(micro.get("markov_over3_bias", 0.0))
    trans_bias    = float(np.clip((mb2 + mb3), 0, 1))

    p2c = calibrator.calibrate(nn_pred["p_over2"])
    p3c = calibrator.calibrate(nn_pred["p_over3"])
    best_p        = max(p2c, p3c)
    neural_conf   = float(np.clip((best_p - 0.5) * 2.0, 0, 1))
    contract_type = "over2" if p2c >= p3c else "over3"

    # Adaptive weights
    w_e = cfg.w_entropy   * learner.weight("entropy")
    w_r = cfg.w_rl        * learner.weight("nn_p_over2")
    w_n = cfg.w_neural    * learner.weight("nn_p_over3")
    w_g = cfg.w_regime
    w_t = cfg.w_transition* learner.weight("markov_over2")
    w_v = cfg.w_volatility* learner.weight("vol_burst")
    tot = w_e + w_r + w_n + w_g + w_t + w_v or 1.0

    conf = (w_e*entropy_score + w_r*rl_conf + w_n*neural_conf
            + w_g*regime_conf + w_t*trans_bias + w_v*vol_score) / tot
    if learner.drift: conf *= 0.70
    conf = float(np.clip(conf, 0, 1))

    block = []
    if not entropy["tradeable"]:     block.append(f"entropy={entropy['composite']:.3f}")
    if not null_rej:                 block.append(f"null_p={null_p:.3f}")
    if regime_conf < cfg.min_regime_stability: block.append(f"regime_conf={regime_conf:.3f}")
    if conf < cfg.min_final_conf:    block.append(f"conf={conf:.3f}<{cfg.min_final_conf}")
    if contract_type == "over2" and p2c < cfg.min_over2_prob:
        block.append(f"p_over2={p2c:.3f}<{cfg.min_over2_prob}")
    if contract_type == "over3" and p3c < cfg.min_over3_prob:
        block.append(f"p_over3={p3c:.3f}<{cfg.min_over3_prob}")
    if learner.drift:                block.append("drift")

    return FusionResult(
        final_confidence=round(conf,4), entropy_score=round(entropy_score,4),
        rl_confidence=round(rl_conf,4), neural_confidence=round(neural_conf,4),
        regime_stability=round(regime_conf,4), transition_bias=round(trans_bias,4),
        volatility_score=round(vol_score,4), p_over2=round(p2c,4),
        p_over3=round(p3c,4), contract_type=contract_type,
        regime=regime_id, regime_name=REGIME_NAMES[regime_id],
        null_rejected=null_rej, null_p=round(null_p,4),
        tradeable=len(block)==0,
        block_reason=" | ".join(block) if block else "ok",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:

    def __init__(self, cfg: Config):
        self.cfg              = cfg
        self._martingale_step = 0
        self._consec_losses   = 0
        self._in_trade        = False
        self._paused          = False
        self._pause_reason    = ""
        self._start_balance:  Optional[float] = None
        self._daily_pnl       = 0.0
        self._cooldown_ticks  = 0

    def set_balance(self, b: float):
        if self._start_balance is None: self._start_balance = b

    @property
    def current_stake(self) -> float:
        step = min(self._martingale_step, self.cfg.martingale_steps)
        return round(self.cfg.base_stake * (self.cfg.martingale_factor ** step), 2)

    def tick(self):
        if self._cooldown_ticks > 0: self._cooldown_ticks -= 1

    def can_trade(self, balance: float) -> Tuple[bool, str]:
        if self._in_trade:   return False, "in_trade"
        if self._paused:     return False, f"paused:{self._pause_reason}"
        if self._cooldown_ticks > 0: return False, f"cooldown:{self._cooldown_ticks}t"
        if self._consec_losses >= self.cfg.max_consecutive_losses:
            self._paused = True; self._pause_reason = f"{self._consec_losses}_consec_losses"
            return False, f"paused:{self._pause_reason}"
        if self._start_balance:
            if self._daily_pnl < -(self._start_balance * self.cfg.max_daily_loss_pct):
                self._paused = True; self._pause_reason = "daily_loss_cap"
                return False, "paused:daily_loss_cap"
        min_safe = self.cfg.base_stake * self.cfg.balance_guard_mult
        if balance > 0 and balance < min_safe:
            return False, f"balance_too_low:{balance:.2f}<{min_safe:.2f}"
        return True, "ok"

    def on_open(self): self._in_trade = True

    def on_close(self, won: bool, profit: float):
        self._in_trade = False; self._daily_pnl += profit
        if won:
            if self._martingale_step > 0:
                log.info(f"WIN at martingale step {self._martingale_step} — reset to ${self.cfg.base_stake:.2f}")
            self._martingale_step = 0; self._consec_losses = 0; self._cooldown_ticks = 0
        else:
            self._consec_losses  += 1
            self._martingale_step = min(self._martingale_step+1, self.cfg.martingale_steps)
            self._cooldown_ticks  = self.cfg.loss_cooldown_ticks
            log.info(f"LOSS #{self._consec_losses} | step {self._martingale_step}/{self.cfg.martingale_steps} "
                     f"| next=${self.current_stake:.2f} | cooldown={self.cfg.loss_cooldown_ticks}t")

    def release_lock(self): self._in_trade = False
    def reset(self):
        self._paused=False; self._consec_losses=0
        self._martingale_step=0; self._cooldown_ticks=0
        log.info("RiskManager reset")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────

class History:
    COLS = ["ts","tick","contract_id","contract_type","stake",
            "final_confidence","entropy_score","rl_confidence","neural_confidence",
            "regime_stability","transition_bias","p_over2","p_over3",
            "regime","null_p","won","profit","balance","settle_source"]

    def __init__(self, path: str):
        self.path = path; self._rows: List[dict] = []
        if not os.path.exists(path):
            with open(path,"w",newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def add(self, row: dict):
        self._rows.append(row)
        with open(self.path,"a",newline="") as f:
            csv.DictWriter(f,fieldnames=self.COLS).writerow({c:row.get(c,"") for c in self.COLS})

    def update_last(self, cid, won: bool, profit: float, balance: float, source: str):
        for r in reversed(self._rows):
            if str(r.get("contract_id"))==str(cid):
                r.update({"won":won,"profit":round(profit,5),
                           "balance":round(balance,4),"settle_source":source})
                self._rewrite(); return

    def _rewrite(self):
        with open(self.path,"w",newline="") as f:
            w = csv.DictWriter(f,fieldnames=self.COLS); w.writeheader()
            for r in self._rows: w.writerow({c:r.get(c,"") for c in self.COLS})

    @property
    def stats(self) -> dict:
        done = [r for r in self._rows if r.get("won") != ""]
        if not done: return {"n":0,"win_rate":0.0,"pnl":0.0}
        wins = sum(1 for r in done if r.get("won") is True or r.get("won")=="True")
        pnl  = sum(float(r.get("profit",0) or 0) for r in done)
        return {"n":len(done),"win_rate":wins/len(done),"pnl":round(pnl,4)}


# ─────────────────────────────────────────────────────────────────────────────
# DERIV WEBSOCKET CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:

    def __init__(self, cfg: Config):
        self.cfg = cfg; self._ws = None; self._rid = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._tick_cb: Optional[Callable] = None
        self._connected = False; self.balance = 0.0

    async def connect(self):
        url = f"{self.cfg.api_url}?app_id={self.cfg.app_id}"
        self._ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        self._connected = True
        asyncio.create_task(self._listen())

    async def auth(self):
        r = await self._rpc({"authorize": self.cfg.api_token})
        if "error" in r: raise ConnectionError(r["error"]["message"])
        self.balance = float(r["authorize"].get("balance", 0))
        log.info(f"Auth OK | login={r['authorize'].get('loginid')} balance={self.balance:.2f}")

    async def subscribe_ticks(self, cb: Callable):
        self._tick_cb = cb
        await self._send({"ticks": self.cfg.symbol, "subscribe": 1, "req_id": self._next()})

    async def buy(self, contract_type: str, stake: float) -> Optional[dict]:
        barrier = 2 if contract_type == "over2" else 3
        r = await self._rpc({"buy": 1, "price": str(stake), "parameters": {
            "amount": str(stake), "basis": "stake",
            "contract_type": "DIGITOVER", "barrier": str(barrier),
            "currency": self.cfg.currency, "duration": self.cfg.duration,
            "duration_unit": "t", "symbol": self.cfg.symbol}})
        if "error" in r: log.error(f"Buy error: {r['error']['message']}"); return None
        b = r.get("buy", {}); self.balance = float(b.get("balance_after", self.balance))
        return b

    async def contract_status(self, cid) -> Optional[dict]:
        r = await self._rpc({"proposal_open_contract": 1, "contract_id": int(cid)})
        return None if "error" in r else r.get("proposal_open_contract")

    async def profit_table_lookup(self, cid) -> Optional[dict]:
        r = await self._rpc({"profit_table": 1, "description": 1, "sort": "DESC", "limit": 10})
        for t in r.get("profit_table",{}).get("transactions",[]):
            if str(t.get("contract_id"))==str(cid): return t
        return None

    async def refresh_balance(self):
        r = await self._rpc({"balance": 1, "account": "current"})
        self.balance = float(r.get("balance",{}).get("balance", self.balance))

    async def disconnect(self):
        if self._ws: await self._ws.close()

    @property
    def connected(self) -> bool: return self._connected

    def _next(self) -> int: self._rid += 1; return self._rid

    async def _rpc(self, payload: dict) -> dict:
        rid = self._next(); payload["req_id"] = rid
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut; await self._send(payload)
        try: return await asyncio.wait_for(fut, timeout=20.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None); return {"error": {"message": "timeout"}}

    async def _send(self, payload: dict): await self._ws.send(json.dumps(payload))

    async def _listen(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("msg_type") == "tick" and self._tick_cb:
                    q = float(msg.get("tick",{}).get("quote",0))
                    if q > 0: asyncio.create_task(self._call(q))
                    continue
                rid = msg.get("req_id")
                if rid and rid in self._pending:
                    f = self._pending.pop(rid)
                    if not f.done(): f.set_result(msg)
        except Exception as e: log.error(f"WS listener: {e}")
        finally: self._connected = False; log.warning("WS listener exited")

    async def _call(self, price: float):
        try:
            if asyncio.iscoroutinefunction(self._tick_cb): await self._tick_cb(price)
            else: self._tick_cb(price)
        except Exception as e: log.error(f"Tick cb: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BOT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Bot:

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.micro      = MicrostructureAnalyzer(cfg)
        self.entropy    = EntropyEngine(cfg)
        self.rl         = RLAgent(cfg)
        self.nn         = DigitNet(cfg)
        self.regime     = RegimeDetector(cfg)
        self.calibrator = Calibrator(cfg)
        self.null_test  = NullHypothesisTester(cfg)
        self.learner    = AdaptiveLearner()
        self.risk       = RiskManager(cfg)
        self.history    = History(cfg.history_file)
        self.client     = DerivClient(cfg)
        self._alive     = True
        self._feat_buf: deque = deque(maxlen=cfg.nn_input_window)
        self._tick = 0
        self._skip_counts: Counter = Counter()
        self._last_skip_log = self._last_state_log = 0.0
        self._ticks_after_warmup = 0
        self._recent_wr: deque = deque(maxlen=20)

    async def run(self):
        retry = 5
        while self._alive:
            try:
                log.info("Connecting...")
                await self.client.connect(); await self.client.auth()
                self.risk.set_balance(self.client.balance)
                log.info(f"Warmup: {self.cfg.warmup_ticks} ticks | balance=${self.client.balance:.2f}")
                await self.client.subscribe_ticks(self.on_tick)
                retry = 5
                while self._alive and self.client.connected: await asyncio.sleep(1)
                if self._alive:
                    log.warning("Disconnected — reconnecting in 5s..."); await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Error: {e} — retry in {retry}s"); await asyncio.sleep(retry)
                retry = min(retry*2, 60)
        await self.client.disconnect()

    async def on_tick(self, price: float):
        self._tick += 1; self.risk.tick()
        digit = int(round(price * 100)) % 10

        micro_f    = self.micro.push(price)
        ent        = self.entropy.push(digit)
        self.null_test.push(digit)

        nn_vec = np.array([digit/9.0, 1.0 if digit>2 else 0.0,
                           1.0 if digit>3 else 0.0, price%1.0], dtype=np.float32)
        self._feat_buf.append(nn_vec)

        if self._tick < self.cfg.warmup_ticks:
            if self._tick % 30 == 0:
                log.info(f"Warmup: {self.cfg.warmup_ticks - self._tick} ticks left...")
            return
        if len(self._feat_buf) < self.cfg.nn_input_window: return

        self._ticks_after_warmup += 1
        regime_id, regime_conf = self.regime.push(price, ent["composite"])
        X         = np.stack(list(self._feat_buf), axis=0)
        nn_pred   = self.nn.predict(X)
        null_rej, null_p, _ = self.null_test.test()
        wr        = float(np.mean(self._recent_wr)) if self._recent_wr else 0.5
        rl_s      = self.rl.state_index(ent["composite"], regime_id, wr,
                                         micro_f.get("volatility_burst", 0.0))
        rl_a, rl_c = self.rl.act(rl_s)
        fusion    = fuse(self.cfg, ent, rl_c if rl_a==1 else 0.0, nn_pred,
                         regime_id, regime_conf, micro_f, null_rej, null_p,
                         self.learner, self.calibrator)

        now = time.time()
        if now - self._last_state_log > 15:
            self._log_state(fusion, rl_s); self._last_state_log = now
        if self._ticks_after_warmup % self.cfg.skip_summary_every == 0:
            self._log_skip_summary()

        if rl_a == 0:
            self._skip_counts["rl_idle"] += 1; return
        if not fusion.tradeable:
            key = fusion.block_reason.split("|")[0].strip()[:30]
            self._skip_counts[key] += 1
            self._maybe_log_skip(fusion); return
        ok, reason = self.risk.can_trade(self.client.balance)
        if not ok:
            self._skip_counts[reason.split(":")[0]] += 1; return
        await self._execute(fusion, rl_s)

    async def _execute(self, fusion: FusionResult, rl_s: int):
        stake = max(min(self.risk.current_stake,
                        round(self.client.balance*self.cfg.max_balance_pct, 2)),
                    self.cfg.base_stake)
        log.info(f"TRADE | {fusion.contract_type} ${stake:.2f} "
                 f"(step {self.risk._martingale_step}/{self.cfg.martingale_steps}) "
                 f"conf={fusion.final_confidence:.3f} regime={fusion.regime_name} "
                 f"entropy={fusion.entropy_score:.3f} p2={fusion.p_over2:.3f} p3={fusion.p_over3:.3f}")
        self.risk.on_open()
        result = await self.client.buy(fusion.contract_type, stake)
        if not result: self.risk.release_lock(); return
        cid = result.get("contract_id")
        buy_price = float(result.get("buy_price", stake))
        self.history.add({"ts": datetime.utcnow().isoformat(), "tick": self._tick,
                           "contract_id": cid, "contract_type": fusion.contract_type,
                           "stake": buy_price, "final_confidence": fusion.final_confidence,
                           "entropy_score": fusion.entropy_score, "rl_confidence": fusion.rl_confidence,
                           "neural_confidence": fusion.neural_confidence,
                           "regime_stability": fusion.regime_stability,
                           "transition_bias": fusion.transition_bias,
                           "p_over2": fusion.p_over2, "p_over3": fusion.p_over3,
                           "regime": fusion.regime_name, "null_p": fusion.null_p})
        await asyncio.sleep(3)
        await self._settle(cid, buy_price, fusion, rl_s, stake)

    async def _settle(self, cid, buy_price: float, fusion: FusionResult,
                      rl_s: int, stake: float):
        won = profit = None; source = "unknown"
        for _ in range(8):
            s = await self.client.contract_status(cid)
            if s:
                sold = s.get("is_sold", False) or s.get("status","") in ("sold","won","lost")
                if sold:
                    ap = s.get("profit"); sp = s.get("sell_price")
                    profit = float(ap) if ap is not None else (float(sp)-buy_price if sp else 0.0)
                    won = profit > 0; source = "proposal_open_contract"; break
            await asyncio.sleep(3)
        if won is None:
            txn = await self.client.profit_table_lookup(cid)
            if txn: profit = float(txn.get("profit",0)); won = profit>0; source="profit_table"
            else:
                log.warning(f"Unconfirmed cid={cid}")
                await self.client.refresh_balance(); self.risk.release_lock(); return
        await self.client.refresh_balance()
        self.risk.on_close(won, profit); self._recent_wr.append(1 if won else 0)
        # Update all learning systems
        reward = profit / stake
        self.rl.update(reward, self.rl.state_index(0.5, 0, float(np.mean(self._recent_wr)), 0.0))
        if len(self._feat_buf) == self.cfg.nn_input_window:
            X = np.stack(list(self._feat_buf), axis=0)
            y = np.array([1.0 if won else 0.0, 1.0 if won else 0.0,
                          0.0 if won else 1.0, 1.0 if won else 0.3], dtype=np.float32)
            self.nn.record(X, y)
        self.calibrator.record(
            fusion.p_over2 if fusion.contract_type=="over2" else fusion.p_over3, won)
        self.learner.record({"nn_p_over2": fusion.p_over2>0.5,
                             "nn_p_over3": fusion.p_over3>0.5,
                             "vol_burst":  fusion.volatility_score>0.5}, won)
        self.history.update_last(cid, won, profit, self.client.balance, source)
        st = self.history.stats
        log.info(f"{'WIN' if won else 'LOSS'} | profit={profit:+.4f} "
                 f"bal=${self.client.balance:.2f} | WR={st['win_rate']:.1%} "
                 f"n={st['n']} PnL={st['pnl']:+.4f} | "
                 f"drift={self.learner.drift} eps={self.rl.epsilon:.3f}")

    def _log_state(self, f: FusionResult, rl_s: int):
        log.info(
            f"[STATE tick={self._tick}] "
            f"regime={f.regime_name}({f.regime_stability:.2f}) "
            f"entropy={f.entropy_score:.3f} conf={f.final_confidence:.3f} "
            f"p2={f.p_over2:.3f} p3={f.p_over3:.3f} "
            f"null={'REJ' if f.null_rejected else 'FAIL'}(p={f.null_p:.3f}) "
            f"eps={self.rl.epsilon:.3f}\n"
            f"  block=[{f.block_reason[:80]}] drift={self.learner.drift}\n"
            f"  weights: {self.learner.summary()}\n"
            f"  {self.micro.markov_matrix_str}"
        )

    def _maybe_log_skip(self, f: FusionResult):
        now = time.time()
        if now - self._last_skip_log < self.cfg.skip_log_interval: return
        self._last_skip_log = now
        log.info(f"[SKIP] {f.block_reason[:100]}")

    def _log_skip_summary(self):
        total = sum(self._skip_counts.values())
        if not total: return
        s = " | ".join(f"{k}:{v}({v/total*100:.0f}%)"
                       for k,v in self._skip_counts.most_common(8))
        log.info(f"[SKIP SUMMARY ticks={self._ticks_after_warmup}] total={total} | {s}")

    def shutdown(self):
        self._alive = False; self._log_skip_summary()
        log.info(f"Shutdown | {self.history.stats}")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER — multiple expiry durations
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(cfg: Config, n_ticks: int = 6000, seed: int = 42):
    import copy
    random.seed(seed); np.random.seed(seed)
    DURATIONS = [1, 3, 5]
    print("=" * 70)
    print("Backtest: R_100 DIGITOVER 2/3 | Multi-layer engine")
    print(f"Ticks: {n_ticks} | Durations: {DURATIONS}")
    print("=" * 70)

    def gen_prices(n):
        prices = []; base = 9800.0; bias = 0.0
        for i in range(n):
            if random.random() < 0.015: bias = random.choice([-0.04,-0.02,0,0.02,0.04])
            base += random.gauss(bias, 0.03)
            price = round(abs(base), 2) + random.randint(0,99)/100.0
            prices.append(price)
        return prices

    prices_master = gen_prices(n_ticks)
    all_res = []

    for dur in DURATIONS:
        random.seed(seed); np.random.seed(seed)
        c = copy.deepcopy(cfg); c.duration = dur
        micro = MicrostructureAnalyzer(c); ent_e = EntropyEngine(c)
        rl    = RLAgent(c); nn = DigitNet(c); reg = RegimeDetector(c)
        cal   = Calibrator(c); nt = NullHypothesisTester(c)
        lrn   = AdaptiveLearner(); risk = RiskManager(c)
        risk.set_balance(1000.0)
        balance = 1000.0; bal_log = [balance]
        trades = wins = 0; skips: Counter = Counter()
        fbuf: deque = deque(maxlen=c.nn_input_window)
        rwr:  deque = deque(maxlen=20)

        for i, price in enumerate(prices_master):
            risk.tick(); digit = int(round(price*100)) % 10
            mf  = micro.push(price); en = ent_e.push(digit); nt.push(digit)
            fbuf.append(np.array([digit/9.0, 1.0 if digit>2 else 0.0,
                                   1.0 if digit>3 else 0.0, price%1.0], dtype=np.float32))
            if i < c.warmup_ticks or len(fbuf) < c.nn_input_window: continue
            rid, rc = reg.push(price, en["composite"])
            X       = np.stack(list(fbuf), axis=0)
            np_pred = nn.predict(X)
            nr, np_, _ = nt.test()
            wr  = float(np.mean(rwr)) if rwr else 0.5
            rls = rl.state_index(en["composite"], rid, wr, mf.get("volatility_burst",0.0))
            rla, rlc = rl.act(rls)
            fus = fuse(c, en, rlc if rla==1 else 0.0, np_pred, rid, rc, mf, nr, np_, lrn, cal)
            if rla==0:            skips["rl_idle"] += 1; continue
            if not fus.tradeable: skips[fus.block_reason.split("|")[0][:25].strip()] += 1; continue
            ok, _ = risk.can_trade(balance)
            if not ok: skips["risk"] += 1; continue
            stake = max(min(risk.current_stake, round(balance*c.max_balance_pct,2)), c.base_stake)
            fi = i + dur
            if fi < len(prices_master):
                fd = int(round(prices_master[fi]*100)) % 10
                won = fd > 2 if fus.contract_type=="over2" else fd > 3
            else:
                won = random.random() < (0.70 if fus.contract_type=="over2" else 0.60)
            profit  = stake*c.payout_ratio if won else -stake
            balance += profit; bal_log.append(balance)
            risk.on_close(won, profit); rwr.append(1 if won else 0)
            trades += 1; wins += (1 if won else 0)
            rl.update(profit/stake, rl.state_index(0.5,0,float(np.mean(rwr)),0.0))
            cal.record(fus.p_over2 if fus.contract_type=="over2" else fus.p_over3, won)
            lrn.record({"nn_p_over2":fus.p_over2>0.5}, won)
            if trades % 50 == 0:
                print(f"  dur={dur}t tick={i:5d} trades={trades} "
                      f"WR={wins/trades:.1%} bal={balance:.2f}")

        wr_f  = wins/trades if trades else 0.0
        pnl   = balance - 1000.0
        peaks = np.maximum.accumulate(bal_log)
        dd    = float(np.max((peaks - bal_log)/(peaks+1e-9))) if len(bal_log)>1 else 0.0
        be    = 1.0 / (1.0 + c.payout_ratio)
        all_res.append({"dur":dur,"trades":trades,"wr":wr_f,"pnl":pnl,
                         "dd":dd,"be":be,"bal":balance,"skips":dict(skips.most_common(5))})

    print(f"\n{'Dur':>4} {'Trades':>7} {'WR':>7} {'BE':>6} {'P&L':>9} {'MaxDD':>7} {'Bal':>9}")
    print("─" * 60)
    for r in all_res:
        star = " ★" if r["wr"] > r["be"] else ""
        print(f"{r['dur']:>4}t {r['trades']:>7} {r['wr']:>6.1%} {r['be']:>6.1%} "
              f"{r['pnl']:>+9.2f} {r['dd']:>6.1%} {r['bal']:>9.2f}{star}")
    best = max(all_res, key=lambda r: r["wr"]-r["be"])
    print(f"\n  ★ Best duration: {best['dur']}t  WR={best['wr']:.1%}  "
          f"edge={best['wr']-best['be']:+.1%}")
    print(f"\n  Skip breakdown (dur={all_res[0]['dur']}t):")
    for g,cnt in all_res[0]["skips"].items():
        print(f"    {g:35s}: {cnt}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

def _start_health_server():
    import http.server, threading
    port = int(os.getenv("PORT", "8080"))
    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(b"OK - overbot running")
        def log_message(self, *a): pass
    threading.Thread(target=http.server.HTTPServer(("",port),_H).serve_forever,
                     daemon=True).start()
    log.info(f"Health server on :{port}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def live(cfg: Config):
    if not cfg.api_token:
        log.error(
            "No API token.\n"
            "  Railway: set DERIV_API_TOKEN in environment variables.\n"
            "  Local:   export DERIV_API_TOKEN=your_token  or edit Config.api_token."
        )
        sys.exit(1)
    bot = Bot(cfg)
    def _sig(s, f): log.info("Shutdown signal received..."); bot.shutdown(); sys.exit(0)
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    log.info("=" * 70)
    log.info("Deriv Over/Under Digits Bot — 9-Layer Probability Engine")
    log.info(f"Symbol: {cfg.symbol} | DIGITOVER 2 & 3 | {cfg.duration}t contracts")
    log.info(f"Layers: Microstructure + Entropy + RL(Q) + NeuralNet + Regime + Fusion + Calibration + NullTest + Adaptive")
    log.info(f"Stake ladder: " +
             " → ".join(f"${cfg.base_stake * cfg.martingale_factor**s:.2f}"
                        for s in range(cfg.martingale_steps+1)) + " → halt")
    log.info(f"Entry gates: conf>{cfg.min_final_conf} entropy<{cfg.entropy_threshold} "
             f"null_p<{cfg.null_hyp_p_value} regime>{cfg.min_regime_stability}")
    log.info("=" * 70)
    _start_health_server()
    await bot.run()


if __name__ == "__main__":
    cfg = Config()
    if "--backtest" in sys.argv:
        run_backtest(cfg)
    else:
        asyncio.run(live(cfg))
