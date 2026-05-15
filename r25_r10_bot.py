# -*- coding: utf-8 -*-
"""
R_25 + R_10 Even/Odd Bot — Data-Calibrated 9-Layer Engine
══════════════════════════════════════════════════════════

SYMBOL SELECTION — from 186 minutes, 125 611 ticks across all 15 symbols:

  R_25  ranked #1 in 37/38 snapshots
        markov_stability  = 0.114   (highest of all 15 symbols)
        zscore_spike_rate = 0.089
        streak_reversion  = 0.253
        even_rate         = 0.557   ← persistent structural EVEN bias
        E→E transition    = 0.559
        O→E transition    = 0.555   (both rows favour even)

  R_10  ranked #2 consistently
        markov_stability  = 0.064
        zscore_spike_rate = 0.089
        streak_reversion  = 0.263
        even_rate         = 0.533   ← moderate persistent even bias
        E→E transition    = 0.545
        O→E transition    = 0.520

ALL OTHER SYMBOLS:
  markov_stability < 0.040 — not meaningfully exploitable

KEY INSIGHT FROM DATA:
  R_25 has a confirmed 5.7% structural EVEN bias over 4 756 ticks.
  R_10 has a confirmed 3.3% structural EVEN bias over 4 759 ticks.
  These are not noise. Both symbols' Markov rows both point toward even.
  Strategy: EVEN is the primary bet. Markov gates the entry, does not
  symmetrically choose sides — it confirms even or suppresses the trade.

DATA-CALIBRATED THRESHOLDS:
  Entropy threshold    : 0.90  (composite runs 0.82–0.91 on these symbols)
  Markov margin gate   : 0.03  (E→E − 0.5 = 0.059 on R_25, use 0.03 floor)
  Min even probability : 0.57  (calibrated from observed 0.557/0.533 rates)
  Min confidence       : 0.22
  Null hyp p-value     : 0.55  (observed chi2 structure in digit frequencies)
  Regime stability     : 0.40
  zscore spike gate    : 0.085 (observed rates 0.089/0.089)

STAKE / MARTINGALE:
  Base: $0.35  Step 1: $0.53  Step 2: $0.79  → halt

Run:
    export DERIV_API_TOKEN=your_token
    python r25_r10_bot.py

Backtest:
    python r25_r10_bot.py --backtest
"""

import asyncio
import csv
import json
import logging
import math
import os
import pickle
import random
import signal
import sys
import time
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
import websockets

# ── UTF-8 stdout guard (Windows CMD / misconfigured terminals) ────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("r25r10bot")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — every threshold sourced directly from collected data
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS = ["R_25", "R_10"]   # trade both; R_25 is primary (higher Markov)

# Per-symbol data observations (from 125k ticks):
SYMBOL_DATA = {
    "R_25": {
        "even_rate":          0.557,
        "markov_stability":   0.114,
        "E_to_E":             0.559,
        "O_to_E":             0.555,
        "zscore_spike_rate":  0.089,
        "streak_reversion":   0.253,
    },
    "R_10": {
        "even_rate":          0.533,
        "markov_stability":   0.064,
        "E_to_E":             0.545,
        "O_to_E":             0.520,
        "zscore_spike_rate":  0.089,
        "streak_reversion":   0.263,
    },
}

@dataclass
class SymConfig:
    """Per-symbol config seeded from real data observations."""
    symbol:              str
    # Observed even-bias — used to set min_even_prob with a small discount
    observed_even_rate:  float
    # Markov stability — gate: only trade when live markov >= this fraction of observed
    observed_markov:     float
    # Warmup ticks before trading
    warmup_ticks:        int

@dataclass
class Config:
    # ── API ───────────────────────────────────────────────────────────────────
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    app_id:    str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_url:   str = "wss://ws.binaryws.com/websockets/v3"

    # ── SYMBOLS ───────────────────────────────────────────────────────────────
    # Both are traded; each has independent engines, state and martingale.
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)

    # ── CONTRACT ──────────────────────────────────────────────────────────────
    duration:      int   = 1
    duration_unit: str   = "t"
    currency:      str   = "USD"
    payout_ratio:  float = 0.95   # DIGITEVEN / DIGITODD payout

    # ── WARMUP ────────────────────────────────────────────────────────────────
    warmup_ticks: int = 100   # ticks before any engine is eligible

    # ── LAYER 1 — MICROSTRUCTURE ──────────────────────────────────────────────
    micro_window:    int = 30
    markov_window:   int = 60
    cluster_window:  int = 20
    momentum_window: int = 15

    # ── LAYER 2 — ENTROPY ─────────────────────────────────────────────────────
    # Observed composite entropy on R_25/R_10 runs 0.82–0.91.
    # Threshold at 0.90 admits tradeable windows while filtering pure noise.
    entropy_window:      int   = 35
    perm_entropy_order:  int   = 4
    entropy_threshold:   float = 0.90   # from data: composite ~0.82–0.91

    # ── LAYER 3 — RL ──────────────────────────────────────────────────────────
    rl_states:        int   = 64
    rl_alpha:         float = 0.15
    rl_gamma:         float = 0.90
    rl_epsilon_start: float = 0.20   # lower — symbols are well-characterised
    rl_epsilon_min:   float = 0.04
    rl_epsilon_decay: float = 0.997

    # ── LAYER 4 — NEURAL NET ──────────────────────────────────────────────────
    nn_input_window: int   = 25
    nn_hidden:       int   = 32
    nn_lr:           float = 0.005
    nn_batch:        int   = 16

    # ── LAYER 5 — REGIME ──────────────────────────────────────────────────────
    regime_window:    int   = 40
    regime_threshold: float = 0.40   # from data: subtler regimes on low-vol symbols

    # ── LAYER 6 — FUSION WEIGHTS ──────────────────────────────────────────────
    # Markov and entropy weighted higher — these showed the clearest signal.
    # Neural net weighted lower — needs time to learn the even bias.
    w_entropy:    float = 0.30   # strongest observed signal
    w_rl:         float = 0.16
    w_neural:     float = 0.16
    w_regime:     float = 0.13
    w_transition: float = 0.15   # Markov — second-strongest on R_25
    w_volatility: float = 0.10

    # ── LAYER 7 — CALIBRATION ─────────────────────────────────────────────────
    cal_window:      int = 50
    cal_recal_every: int = 25

    # ── LAYER 8 — NULL HYPOTHESIS ─────────────────────────────────────────────
    null_hyp_window:  int   = 25
    # Observed chi2 structure in R_25 digit_frequencies (digit 8=14%, digit 5=7%).
    # Set p-value threshold at 0.55 to exploit this non-uniformity.
    null_hyp_p_value: float = 0.55

    # ── ENTRY THRESHOLDS — DATA-CALIBRATED ────────────────────────────────────
    # R_25 observed even_rate = 0.557 → min_even_prob = 0.55 (slight discount)
    # R_10 observed even_rate = 0.533 → min_even_prob = 0.52 for R_10
    # We use a single global value and let per-symbol override handle the rest.
    # The Markov gate (markov_even_bias > markov_margin) is more important.
    min_even_prob:        float = 0.55   # from R_25 observed rate, discounted
    min_odd_prob:         float = 0.60   # odd requires higher bar (data shows even bias)
    min_final_conf:       float = 0.22
    min_regime_stability: float = 0.40

    # Markov margin: only take even when live E→E > 0.50 + this margin.
    # Data shows R_25 E→E=0.559, so 0.03 gives headroom.
    markov_even_margin: float = 0.03

    # zscore spike rate gate: data observed 0.089 on both symbols.
    # Require recent zscore_spike_rate > 0.07 to confirm active structure.
    min_zscore_spike_rate: float = 0.07

    # ── STAKE / MARTINGALE ────────────────────────────────────────────────────
    base_stake:        float = 0.35
    martingale_factor: float = 1.50
    martingale_steps:  int   = 2      # $0.35 → $0.53 → $0.79 → halt
    max_balance_pct:   float = 0.10

    # ── RISK ──────────────────────────────────────────────────────────────────
    loss_cooldown_ticks:    int   = 5
    max_consecutive_losses: int   = 2
    max_daily_loss_pct:     float = 0.15
    balance_guard_mult:     int   = 6

    # ── PERSISTENCE ───────────────────────────────────────────────────────────
    state_dir:    str = "."
    history_file: str = "r25_r10_trades.csv"

    # ── LOGGING ───────────────────────────────────────────────────────────────
    skip_log_interval:  float = 30.0
    skip_summary_every: int   = 150


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
REGIME_TRENDING   = 0
REGIME_REVERTING  = 1
REGIME_CHAOTIC    = 2
REGIME_VOL_EXPAND = 3
REGIME_STABLE     = 4
REGIME_NAMES      = ["trending", "reverting", "chaotic", "vol_expand", "stable"]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — MICROSTRUCTURE ANALYZER
# Key change from original: tracks live even_rate vs observed baseline,
# and flags when current Markov E→E > markov_even_margin above 0.5.
# ─────────────────────────────────────────────────────────────────────────────

class MicrostructureAnalyzer:

    def __init__(self, cfg: Config, symbol: str):
        self.cfg    = cfg
        self.symbol = symbol
        self.obs    = SYMBOL_DATA[symbol]   # observed baselines from collected data

        self._prices:   deque = deque(maxlen=max(cfg.micro_window, cfg.cluster_window, 100))
        self._digits:   deque = deque(maxlen=cfg.markov_window + 10)
        self._diffs:    deque = deque(maxlen=cfg.micro_window)
        self._vel_hist: deque = deque(maxlen=cfg.micro_window)

        # Markov [prev_parity][curr_parity] — Laplace prior [1,1]
        # Pre-seeded with observed data proportions so the engine starts
        # directional rather than flat. Each row seeded at observed rate × 10.
        ee = self.obs["E_to_E"]; eo = 1.0 - ee
        oe = self.obs["O_to_E"]; oo = 1.0 - oe
        # Seed with 10 pseudo-observations so priors are informative but weak
        self._mk: List[List[float]] = [
            [ee * 10, eo * 10],   # prev=even
            [oe * 10, oo * 10],   # prev=odd
        ]
        self._prev_parity: Optional[int] = None
        self._even_count: int = 0
        self._total:      int = 0

        # zscore spike tracking for gate
        self._zscore_spikes: int = 0
        self._zscore_checks: int = 0

    def push(self, price: float) -> dict:
        digit  = int(round(price * 100)) % 10
        parity = digit % 2
        self._prices.append(price)
        self._digits.append(digit)
        self._even_count += (1 if parity == 0 else 0)
        self._total      += 1
        f = {}

        prices = list(self._prices)
        if len(prices) >= 2:
            diff = abs(prices[-1] - prices[-2])
            self._diffs.append(diff)
            self._vel_hist.append(diff)

        if len(self._vel_hist) >= 10:
            vel  = list(self._vel_hist)
            mu   = np.mean(vel); sig = np.std(vel) + 1e-12
            vz   = float((vel[-1] - mu) / sig)
            f["tick_velocity_z"] = vz
            diffs = list(self._diffs)
            mid   = len(diffs) // 2
            v1    = np.mean(diffs[:mid]) if mid else mu
            v2    = np.mean(diffs[mid:])
            f["tick_acceleration_z"] = float((v2 - v1) / (sig + 1e-12))
            self._zscore_checks += 1
            if abs(vz) > 1.5:
                self._zscore_spikes += 1
        else:
            f["tick_velocity_z"] = f["tick_acceleration_z"] = 0.0

        f["volatility_burst"]     = f["tick_velocity_z"]
        f["momentum_exhaustion"]  = (self._momentum_exhaustion(prices)
                                     if len(prices) >= self.cfg.momentum_window else 0.0)
        f["reversal_compression"] = (self._reversal_compression(prices)
                                     if len(prices) >= 10 else 0.0)

        self._update_markov(parity)
        f["markov_even_bias"] = self._markov_bias(0)   # P(next even|current) - 0.5
        f["markov_odd_bias"]  = self._markov_bias(1)   # P(next odd |current) - 0.5

        # Live even_rate vs observed baseline
        w = self.cfg.entropy_window
        if len(self._digits) >= w:
            recent = list(self._digits)[-w:]
            f["even_rate"] = sum(1 for d in recent if d % 2 == 0) / w
        else:
            f["even_rate"] = self.obs["even_rate"]   # use observed as default

        f["cluster_density"] = (
            self._cluster_density(list(self._digits)[-self.cfg.cluster_window:])
            if len(self._digits) >= self.cfg.cluster_window else 0.0)

        # Derived gates based on real data
        # markov_confirms_even: current Markov E→E is above baseline margin
        row_e = self._mk[0]; tot_e = row_e[0] + row_e[1]
        live_ee = float(row_e[0] / tot_e) if tot_e else self.obs["E_to_E"]
        f["markov_confirms_even"] = live_ee >= (0.50 + self.cfg.markov_even_margin)

        # zscore_spike_rate_ok: recent spike rate at least min_zscore_spike_rate
        f["zscore_spike_rate"] = float(self._zscore_spikes / max(self._zscore_checks, 1))
        f["zscore_gate_ok"]    = f["zscore_spike_rate"] >= self.cfg.min_zscore_spike_rate

        # Even rate currently favours even bet
        f["even_rate_gate_ok"] = f["even_rate"] >= self.obs["even_rate"] - 0.04

        return f

    def _momentum_exhaustion(self, prices: list) -> float:
        w = prices[-self.cfg.momentum_window:]
        runs, cur = [], 1
        for i in range(1, len(w)):
            same = (w[i] > w[i-1]) == (w[i-1] > w[i-2]) if i > 1 else True
            if same: cur += 1
            else:    runs.append(cur); cur = 1
        runs.append(cur)
        if len(runs) < 3: return 0.0
        mid   = len(runs) // 2
        ratio = 1.0 - min(np.mean(runs[mid:]) / (np.mean(runs[:mid]) + 1e-9), 1.0)
        return float(np.clip(ratio, 0, 1))

    def _reversal_compression(self, prices: list) -> float:
        w   = prices[-20:]
        rev = sum(1 for i in range(1, len(w) - 1)
                  if (w[i] > w[i-1]) != (w[i] < w[i+1]))
        return float(rev / max(len(w) - 2, 1))

    def _update_markov(self, parity: int):
        if self._prev_parity is not None:
            self._mk[self._prev_parity][parity] += 1.0
        self._prev_parity = parity

    def _markov_bias(self, query_parity: int) -> float:
        if self._prev_parity is None: return 0.0
        row = self._mk[self._prev_parity]
        tot = row[0] + row[1]
        return float((row[query_parity] / tot) - 0.5) if tot else 0.0

    def _cluster_density(self, digits: list) -> float:
        counts = np.bincount(digits, minlength=10).astype(float)
        exp    = len(digits) / 10.0
        chi2   = float(np.sum((counts - exp) ** 2 / (exp + 1e-9)))
        return float(np.clip(chi2 / (len(digits) * 9.0), 0, 1))

    def parity_bias_str(self) -> str:
        ee = self._mk[0][0]; eo = self._mk[0][1]
        oe = self._mk[1][0]; oo = self._mk[1][1]
        te = ee + eo + 1e-9; to = oe + oo + 1e-9
        long_bias = (self._even_count / max(self._total, 1) - 0.5) * 100
        return (f"E->E:{ee/te:.3f} E->O:{eo/te:.3f} | "
                f"O->E:{oe/to:.3f} O->O:{oo/to:.3f} | "
                f"long_even_bias:{long_bias:+.1f}% "
                f"(obs:{(self.obs['even_rate']-0.5)*100:+.1f}%)")

    def get_state(self) -> dict:
        return {"mk": [row[:] for row in self._mk],
                "even_count": self._even_count, "total": self._total,
                "zscore_spikes": self._zscore_spikes,
                "zscore_checks": self._zscore_checks}

    def load_state(self, s: dict):
        self._mk           = s["mk"]
        self._even_count   = s.get("even_count", 0)
        self._total        = s.get("total", 0)
        self._zscore_spikes = s.get("zscore_spikes", 0)
        self._zscore_checks = s.get("zscore_checks", 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — ENTROPY ENGINE
# threshold = 0.90 from data (observed composite 0.82–0.91 on R_25/R_10)
# ─────────────────────────────────────────────────────────────────────────────

class EntropyEngine:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._digits: deque = deque(maxlen=max(cfg.entropy_window, cfg.null_hyp_window, 150))

    def push(self, digit: int) -> dict:
        self._digits.append(digit)
        result = {"shannon": 1.0, "permutation": 1.0,
                  "uniformity": 1.0, "composite": 1.0, "tradeable": False}
        if len(self._digits) < self.cfg.entropy_window:
            return result
        window    = list(self._digits)[-self.cfg.entropy_window:]
        shannon   = self._shannon(window)
        perm      = self._perm_entropy(window, self.cfg.perm_entropy_order)
        unif      = self._uniformity_p(window)
        composite = float(np.clip(0.45*shannon + 0.35*perm + 0.20*(1.0 - unif), 0, 1))
        result.update({
            "shannon":     round(shannon,   4),
            "permutation": round(perm,      4),
            "uniformity":  round(unif,      4),
            "composite":   round(composite, 4),
            "tradeable":   composite < self.cfg.entropy_threshold,
        })
        return result

    @staticmethod
    def _shannon(digits: list) -> float:
        counts = np.bincount(digits, minlength=10).astype(float)
        probs  = counts[counts > 0] / counts.sum()
        return float(-np.sum(probs * np.log2(probs)) / np.log2(10))

    @staticmethod
    def _perm_entropy(digits: list, order: int) -> float:
        if len(digits) < order + 1: return 1.0
        pats  = Counter(
            tuple(sorted(range(order), key=lambda j: digits[i:i+order][j]))
            for i in range(len(digits) - order + 1))
        total = sum(pats.values())
        probs = [v / total for v in pats.values()]
        h     = -sum(p * math.log2(p) for p in probs if p > 0)
        max_h = math.log2(math.factorial(order))
        return float(h / max_h) if max_h > 0 else 1.0

    @staticmethod
    def _uniformity_p(digits: list) -> float:
        counts   = np.bincount(digits, minlength=10).astype(float)
        expected = np.full(10, len(digits) / 10.0)
        _, p     = scipy_stats.chisquare(counts, expected)
        return float(p)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — RL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent:

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self._Q:      np.ndarray    = np.zeros((cfg.rl_states, 2))
        self._eps:    float         = cfg.rl_epsilon_start
        self._last_s: Optional[int] = None
        self._last_a: Optional[int] = None

    def state_index(self, entropy: float, regime: int,
                    win_rate: float, vol_z: float) -> int:
        e  = min(int(entropy * 4), 3)
        r  = min(regime, 4)
        wr = min(int(win_rate * 4), 3)
        vz = min(int((np.clip(vol_z, -3, 3) + 3) / 1.5), 3)
        return int(np.clip(e*16 + r*3 + wr + vz, 0, self.cfg.rl_states - 1))

    def act(self, state: int) -> Tuple[int, float]:
        action = (random.choice([0, 1]) if random.random() < self._eps
                  else int(np.argmax(self._Q[state])))
        q  = self._Q[state]
        qr = max(abs(q.max() - q.min()), 1e-9)
        conf = float(np.clip((q[action] - q.min()) / qr, 0, 1))
        self._last_s = state; self._last_a = action
        return action, conf

    def update(self, reward: float, next_s: int):
        if self._last_s is None: return
        td = (reward + self.cfg.rl_gamma * np.max(self._Q[next_s])
              - self._Q[self._last_s][self._last_a])
        self._Q[self._last_s][self._last_a] += self.cfg.rl_alpha * td
        self._eps = max(self.cfg.rl_epsilon_min, self._eps * self.cfg.rl_epsilon_decay)

    @property
    def epsilon(self) -> float: return self._eps

    def get_state(self) -> dict:
        return {"Q": self._Q.tolist(), "eps": self._eps}

    def load_state(self, s: dict):
        self._Q   = np.array(s["Q"])
        self._eps = s["eps"]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — DIGIT PREDICTION NETWORK
# Neutral initialisation — learns the observed even bias from data.
# ─────────────────────────────────────────────────────────────────────────────

class DigitNet:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        w = cfg.nn_input_window; h = cfg.nn_hidden; F = 3
        rng = np.random.default_rng(42)
        self.W_conv = rng.normal(0, 0.1, (h, F, 3))
        self.b_conv = np.zeros(h)
        pool_out    = max((w - 2) // 2, 1)
        gru_in      = h * pool_out
        self.W_gru  = rng.normal(0, 0.1, (h, gru_in))
        self.b_gru  = np.zeros(h)
        self.W_att  = rng.normal(0, 0.1, (1, h))
        self.b_att  = np.zeros(1)
        self.W_out  = rng.normal(0, 0.1, (3, h))
        self.b_out  = np.array([0.0, 0.0, 0.0])   # neutral start
        self._buf_X: List[np.ndarray] = []
        self._buf_y: List[np.ndarray] = []

    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    @staticmethod
    def _relu(x): return np.maximum(0, x)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, F = X.shape; k = 3; out_len = T - k + 1
        cnn  = np.zeros((self.cfg.nn_hidden, out_len))
        for t in range(out_len):
            cnn[:, t] = self._relu(
                np.einsum('hij,ij->h', self.W_conv, X[t:t+k, :].T) + self.b_conv)
        pool_len = max(out_len // 2, 1)
        pooled   = np.array([cnn[:, i*2:i*2+2].max(axis=1)
                             for i in range(pool_len)]).T
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
        p_even = float(out[0])
        return {"p_even": p_even, "p_odd": 1.0 - p_even,
                "noise": float(out[1]), "stability": float(out[2])}

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
            err    = out - self._buf_y[i]
            gW    += np.outer(err, h); gb += err
        n = len(idxs)
        self.W_out -= self.cfg.nn_lr * gW / n
        self.b_out -= self.cfg.nn_lr * gb / n

    def get_state(self) -> dict:
        return {k: getattr(self, k).tolist()
                for k in ("W_conv","b_conv","W_gru","b_gru",
                          "W_att","b_att","W_out","b_out")}

    def load_state(self, s: dict):
        for k, v in s.items():
            setattr(self, k, np.array(v))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class RegimeDetector:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._prices: deque = deque(maxlen=cfg.regime_window)
        self._vol_h:  deque = deque(maxlen=cfg.regime_window)

    def push(self, price: float, entropy_score: float) -> Tuple[int, float]:
        """entropy_score = 1 - composite (high = clean market)."""
        self._prices.append(price)
        prices = list(self._prices)
        if len(prices) < 15:
            return REGIME_STABLE, 0.50
        diffs = np.diff(prices)
        vol   = float(np.std(diffs)); self._vol_h.append(vol)
        ac    = (float(np.corrcoef(diffs[:-1], diffs[1:])[0, 1])
                 if len(diffs) >= 10 else 0.0)
        vol_z = (float((vol - np.mean(list(self._vol_h)))
                       / (np.std(list(self._vol_h)) + 1e-12))
                 if len(self._vol_h) >= 10 else 0.0)
        if   vol_z > 2.0:
            regime, conf = REGIME_VOL_EXPAND, min(0.5 + vol_z*0.1, 1.0)
        elif entropy_score < 0.12:
            regime, conf = REGIME_CHAOTIC,    0.70
        elif ac > 0.30:
            regime, conf = REGIME_TRENDING,   min(0.5 + ac, 1.0)
        elif ac < -0.30:
            regime, conf = REGIME_REVERTING,  min(0.5 + abs(ac), 1.0)
        else:
            regime, conf = REGIME_STABLE,     max(0.5, entropy_score)
        return regime, float(np.clip(conf, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — CALIBRATOR (Platt scaling)
# Separate instances for even and odd per symbol.
# ─────────────────────────────────────────────────────────────────────────────

class Calibrator:

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self._A     = 1.0; self._B = 0.0
        self._hist: deque = deque(maxlen=cfg.cal_window)
        self._since = 0

    def calibrate(self, p: float) -> float:
        x = self._A * p + self._B
        return float(1.0 / (1.0 + math.exp(-max(-20, min(20, x)))))

    def record(self, p_raw: float, won: bool):
        self._hist.append((p_raw, 1.0 if won else 0.0))
        self._since += 1
        if self._since >= self.cfg.cal_recal_every:
            self._refit(); self._since = 0

    def _refit(self):
        if len(self._hist) < 10: return
        data = list(self._hist); n = len(data)
        w    = np.array([math.exp(-0.05*(n-1-i)) for i in range(n)])
        w   /= w.sum()
        ps   = np.array([d[0] for d in data])
        ys   = np.array([d[1] for d in data])
        A, B = self._A, self._B
        for _ in range(60):
            pc  = 1.0 / (1.0 + np.exp(-(A*ps + B)))
            err = pc - ys
            A  -= 0.10 * float(np.sum(w*err*ps))
            B  -= 0.10 * float(np.sum(w*err))
        self._A, self._B = A, B

    def get_state(self) -> dict:
        return {"A": self._A, "B": self._B, "hist": list(self._hist)}

    def load_state(self, s: dict):
        self._A = s["A"]; self._B = s["B"]
        self._hist = deque(s.get("hist", []), maxlen=self.cfg.cal_window)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 8 — NULL HYPOTHESIS TESTER
# p-value threshold 0.55 — data shows real chi2 structure in R_25 digits.
# ─────────────────────────────────────────────────────────────────────────────

class NullHypothesisTester:

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self._digits: deque = deque(maxlen=cfg.null_hyp_window)

    def push(self, digit: int): self._digits.append(digit)

    def test(self) -> Tuple[bool, float, str]:
        digits = list(self._digits)
        if len(digits) < self.cfg.null_hyp_window:
            return False, 1.0, "insufficient_data"
        counts   = np.bincount(digits, minlength=10).astype(float)
        expected = np.full(10, len(digits) / 10.0)
        _, p_chi = scipy_stats.chisquare(counts, expected)
        binary   = [1 if d % 2 == 0 else 0 for d in digits]
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
# LAYER 9 — ADAPTIVE LEARNER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLearner:
    FEATURES = ["entropy", "markov_even", "markov_odd", "momentum",
                "reversal", "cluster", "vol_burst", "nn_p_even", "regime_stability"]

    def __init__(self):
        self._weights  = {f: 1.0 for f in self.FEATURES}
        self._scores   = {f: deque(maxlen=50) for f in self.FEATURES}
        self._outcomes: deque = deque(maxlen=100)
        self._side_wins = {"even": 0, "odd": 0, "total": 0}
        self._drift    = False

    def record(self, preds: dict, won: bool, side: str):
        self._outcomes.append(1 if won else 0)
        self._side_wins["total"] += 1
        if won: self._side_wins[side] = self._side_wins.get(side, 0) + 1
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
                log.warning("[DRIFT] Recent accuracy <45% — penalising all weights")
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
        return "  ".join(f"{k}={v:.2f}" for k, v in top)

    def get_state(self) -> dict:
        return {"weights": dict(self._weights),
                "side_wins": self._side_wins,
                "scores": {k: list(v) for k, v in self._scores.items()}}

    def load_state(self, s: dict):
        self._weights   = s.get("weights", {f: 1.0 for f in self.FEATURES})
        self._side_wins = s.get("side_wins", {"even":0,"odd":0,"total":0})
        for f, vals in s.get("scores", {}).items():
            if f in self._scores:
                self._scores[f] = deque(vals, maxlen=50)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — CONFIDENCE FUSION ENGINE
# Key change: even bias is the PRIMARY side. Markov confirms or suppresses.
# Odd trades only fire when Markov clearly disagrees with even direction.
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
    p_even:            float
    p_odd:             float
    side:              str    # "even" or "odd"
    regime:            int
    regime_name:       str
    null_rejected:     bool
    null_p:            float
    tradeable:         bool
    block_reason:      str


def fuse(cfg: Config,
         symbol:      str,
         entropy:     dict,
         rl_conf:     float,
         nn_pred:     dict,
         regime_id:   int,
         regime_conf: float,
         micro:       dict,
         null_rej:    bool,
         null_p:      float,
         learner:     AdaptiveLearner,
         even_cal:    Calibrator,
         odd_cal:     Calibrator) -> FusionResult:

    obs           = SYMBOL_DATA[symbol]
    entropy_score = float(1.0 - entropy["composite"])
    vol_z         = micro.get("volatility_burst", 0.0)
    vol_score     = float(np.clip(1.0 - abs(vol_z) / 3.0, 0, 1))

    mb_even    = micro.get("markov_even_bias", 0.0)
    mb_odd     = micro.get("markov_odd_bias",  0.0)
    trans_bias = float(np.clip(max(abs(mb_even), abs(mb_odd)) * 2.0, 0, 1))

    p_even_raw = nn_pred["p_even"]
    p_odd_raw  = nn_pred["p_odd"]
    p_even_cal = even_cal.calibrate(p_even_raw)
    p_odd_cal  = odd_cal.calibrate(p_odd_raw)

    best_p      = max(p_even_cal, p_odd_cal)
    neural_conf = float(np.clip((best_p - 0.5) * 2.0, 0, 1))

    # Adaptive fusion weights
    w_e = cfg.w_entropy    * learner.weight("entropy")
    w_r = cfg.w_rl         * learner.weight("nn_p_even")
    w_n = cfg.w_neural     * learner.weight("nn_p_even")
    w_g = cfg.w_regime
    w_t = cfg.w_transition * max(learner.weight("markov_even"),
                                 learner.weight("markov_odd"))
    w_v = cfg.w_volatility * learner.weight("vol_burst")
    tot = w_e + w_r + w_n + w_g + w_t + w_v or 1.0

    conf = (w_e*entropy_score + w_r*rl_conf + w_n*neural_conf
            + w_g*regime_conf + w_t*trans_bias + w_v*vol_score) / tot
    if learner.drift: conf *= 0.70
    conf = float(np.clip(conf, 0, 1))

    # ── SIDE SELECTION — DATA-INFORMED ────────────────────────────────────────
    # Primary logic: data shows persistent even bias on both symbols.
    # 1. Default candidate = "even"
    # 2. Markov confirms even if E→E > 0.5 + markov_even_margin
    # 3. Markov may override to "odd" only if mb_odd > mb_even by a clear margin
    # 4. NN may also confirm the direction
    markov_confirms_even = micro.get("markov_confirms_even", False)
    markov_even_bias     = mb_even   # P(next even|current) - 0.5
    markov_odd_bias      = mb_odd

    # Markov says "odd" only if odd bias clearly exceeds even bias
    markov_clearly_odd = (mb_odd - mb_even) > 0.04

    if markov_clearly_odd:
        markov_side = "odd"
    elif markov_confirms_even or markov_even_bias > 0:
        markov_side = "even"
    else:
        markov_side = ""   # ambiguous — don't trade

    # NN side
    if abs(p_even_cal - p_odd_cal) > 0.02:
        nn_side = "even" if p_even_cal >= p_odd_cal else "odd"
    else:
        nn_side = ""

    # Resolve side
    if markov_side and nn_side:
        side = markov_side if markov_side == nn_side else (
            "odd" if markov_side == "odd" and nn_side == "odd" else markov_side
        )
    elif markov_side:
        side = markov_side
    elif nn_side:
        side = nn_side
    else:
        side = ""

    best_p_cal = p_even_cal if side == "even" else p_odd_cal if side == "odd" else 0.0

    # ── ENTRY GATES ───────────────────────────────────────────────────────────
    block = []
    if not entropy["tradeable"]:
        block.append(f"entropy={entropy['composite']:.3f}>={cfg.entropy_threshold}")
    if not null_rej:
        block.append(f"null_p={null_p:.3f}>={cfg.null_hyp_p_value}")
    if regime_conf < cfg.min_regime_stability:
        block.append(f"regime_conf={regime_conf:.3f}<{cfg.min_regime_stability}")
    if conf < cfg.min_final_conf:
        block.append(f"conf={conf:.3f}<{cfg.min_final_conf}")
    if not side:
        block.append("no_side_agreed")
    elif side == "even" and best_p_cal < cfg.min_even_prob:
        block.append(f"p_even={best_p_cal:.3f}<{cfg.min_even_prob}")
    elif side == "odd" and best_p_cal < cfg.min_odd_prob:
        block.append(f"p_odd={best_p_cal:.3f}<{cfg.min_odd_prob}")
    # Data gate: zscore spike rate must be active enough
    if not micro.get("zscore_gate_ok", True):
        block.append(f"zscore_rate={micro.get('zscore_spike_rate',0):.3f}<{cfg.min_zscore_spike_rate}")
    # Data gate: even rate must be near observed baseline (not in an anomalous dip)
    if side == "even" and not micro.get("even_rate_gate_ok", True):
        block.append(f"even_rate_dip={micro.get('even_rate',0):.3f}<{obs['even_rate']-0.04:.3f}")
    if learner.drift:
        block.append("drift")

    return FusionResult(
        final_confidence  = round(conf, 4),
        entropy_score     = round(entropy_score, 4),
        rl_confidence     = round(rl_conf, 4),
        neural_confidence = round(neural_conf, 4),
        regime_stability  = round(regime_conf, 4),
        transition_bias   = round(trans_bias, 4),
        volatility_score  = round(vol_score, 4),
        p_even            = round(p_even_cal, 4),
        p_odd             = round(p_odd_cal,  4),
        side              = side,
        regime            = regime_id,
        regime_name       = REGIME_NAMES[regime_id],
        null_rejected     = null_rej,
        null_p            = round(null_p, 4),
        tradeable         = len(block) == 0,
        block_reason      = " | ".join(block) if block else "ok",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER  (independent per symbol)
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
        if self._cooldown_ticks > 0:
            return False, f"cooldown:{self._cooldown_ticks}t"
        if self._consec_losses >= self.cfg.max_consecutive_losses:
            self._paused = True
            self._pause_reason = f"{self._consec_losses}_consec_losses"
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
        self._in_trade   = False
        self._daily_pnl += profit
        if won:
            if self._martingale_step > 0:
                log.info(f"WIN at martingale step {self._martingale_step} — reset to base")
            self._martingale_step = 0
            self._consec_losses   = 0
            self._cooldown_ticks  = 0
        else:
            self._consec_losses  += 1
            self._martingale_step = min(self._martingale_step + 1,
                                        self.cfg.martingale_steps)
            self._cooldown_ticks  = self.cfg.loss_cooldown_ticks
            log.info(f"LOSS #{self._consec_losses} | "
                     f"mart_step={self._martingale_step}/{self.cfg.martingale_steps} | "
                     f"next=${self.current_stake:.2f} | "
                     f"cooldown={self.cfg.loss_cooldown_ticks}t")

    def release_lock(self): self._in_trade = False

    def reset(self):
        self._paused = False; self._consec_losses = 0
        self._martingale_step = 0; self._cooldown_ticks = 0
        log.info("RiskManager reset")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────

class History:
    COLS = ["ts","symbol","tick","contract_id","side","stake",
            "final_confidence","entropy_score","rl_confidence",
            "neural_confidence","regime_stability","transition_bias",
            "p_even","p_odd","regime","null_p",
            "won","profit","balance","settle_source"]

    def __init__(self, path: str):
        self.path  = path
        self._rows: List[dict] = []
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def add(self, row: dict):
        self._rows.append(row)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLS).writerow(
                {c: row.get(c, "") for c in self.COLS})

    def update_last(self, cid, won: bool, profit: float,
                    balance: float, source: str):
        for r in reversed(self._rows):
            if str(r.get("contract_id")) == str(cid):
                r.update({"won": won, "profit": round(profit, 5),
                           "balance": round(balance, 4),
                           "settle_source": source})
                self._rewrite(); return

    def _rewrite(self):
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            w.writeheader()
            for r in self._rows:
                w.writerow({c: r.get(c, "") for c in self.COLS})

    @property
    def stats(self) -> dict:
        done = [r for r in self._rows if r.get("won") != ""]
        if not done: return {"n": 0, "win_rate": 0.0, "pnl": 0.0,
                              "even_wr": 0.0, "odd_wr": 0.0,
                              "even_n": 0, "odd_n": 0}
        wins      = [r for r in done if r.get("won") is True or r.get("won") == "True"]
        pnl       = sum(float(r.get("profit", 0) or 0) for r in done)
        even_done = [r for r in done if r.get("side") == "even"]
        odd_done  = [r for r in done if r.get("side") == "odd"]
        ew = sum(1 for r in even_done
                 if r.get("won") is True or r.get("won") == "True")
        ow = sum(1 for r in odd_done
                 if r.get("won") is True or r.get("won") == "True")
        return {
            "n":        len(done),
            "win_rate": len(wins) / len(done),
            "pnl":      round(pnl, 4),
            "even_wr":  ew / max(len(even_done), 1),
            "odd_wr":   ow / max(len(odd_done),  1),
            "even_n":   len(even_done),
            "odd_n":    len(odd_done),
        }

    def stats_by_symbol(self) -> dict:
        result = {}
        for sym in SYMBOLS:
            rows = [r for r in self._rows
                    if r.get("symbol") == sym and r.get("won") != ""]
            if not rows:
                result[sym] = {"n": 0, "win_rate": 0.0, "pnl": 0.0}; continue
            wins = [r for r in rows if r.get("won") is True or r.get("won") == "True"]
            pnl  = sum(float(r.get("profit", 0) or 0) for r in rows)
            result[sym] = {
                "n":        len(rows),
                "win_rate": len(wins) / len(rows),
                "pnl":      round(pnl, 4),
            }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL ENGINE BUNDLE
# ─────────────────────────────────────────────────────────────────────────────

class SymbolEngine:
    """All 9 layers for one symbol, independently stateful."""

    def __init__(self, symbol: str, cfg: Config):
        self.symbol   = symbol
        self.cfg      = cfg
        self.micro    = MicrostructureAnalyzer(cfg, symbol)
        self.entropy  = EntropyEngine(cfg)
        self.rl       = RLAgent(cfg)
        self.nn       = DigitNet(cfg)
        self.regime   = RegimeDetector(cfg)
        self.even_cal = Calibrator(cfg)
        self.odd_cal  = Calibrator(cfg)
        self.null_t   = NullHypothesisTester(cfg)
        self.learner  = AdaptiveLearner()
        self.risk     = RiskManager(cfg)

        self._feat_buf:  deque = deque(maxlen=cfg.nn_input_window)
        self._recent_wr: deque = deque(maxlen=20)
        self._tick       = 0
        self._skip_counts: Counter = Counter()
        self._last_skip_log  = 0.0
        self._last_state_log = 0.0
        self._ticks_after_warmup = 0

    def state_path(self) -> str:
        return os.path.join(self.cfg.state_dir, f"{self.symbol}_r25r10_state.pkl")

    def save_state(self):
        state = {
            "version":  3,
            "symbol":   self.symbol,
            "saved_at": datetime.utcnow().isoformat(),
            "rl":       self.rl.get_state(),
            "nn":       self.nn.get_state(),
            "even_cal": self.even_cal.get_state(),
            "odd_cal":  self.odd_cal.get_state(),
            "learner":  self.learner.get_state(),
            "micro":    self.micro.get_state(),
        }
        path = self.state_path()
        tmp  = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=4)
        os.replace(tmp, path)
        log.info(f"[{self.symbol}] State saved")

    def load_state(self):
        path = self.state_path()
        if not os.path.exists(path):
            log.info(f"[{self.symbol}] No state file — starting fresh")
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.rl.load_state(state["rl"])
            self.nn.load_state(state["nn"])
            self.even_cal.load_state(state["even_cal"])
            self.odd_cal.load_state(state["odd_cal"])
            self.learner.load_state(state["learner"])
            self.micro.load_state(state["micro"])
            log.info(f"[{self.symbol}] State loaded (saved {state.get('saved_at','?')})")
        except Exception as e:
            log.warning(f"[{self.symbol}] State load failed: {e} — starting fresh")

    def on_tick(self, price: float) -> Optional[FusionResult]:
        """Returns FusionResult if tradeable, else None."""
        self._tick += 1
        self.risk.tick()
        digit  = int(round(price * 100)) % 10
        parity = digit % 2

        micro_f = self.micro.push(price)
        ent     = self.entropy.push(digit)
        self.null_t.push(digit)

        nn_vec = np.array([digit / 9.0, float(parity), price % 1.0], dtype=np.float32)
        self._feat_buf.append(nn_vec)

        if self._tick < self.cfg.warmup_ticks:
            if self._tick % 25 == 0:
                log.info(f"[{self.symbol}] Warmup: {self.cfg.warmup_ticks - self._tick} ticks left")
            return None
        if len(self._feat_buf) < self.cfg.nn_input_window:
            return None

        self._ticks_after_warmup += 1
        regime_id, regime_conf = self.regime.push(price, 1.0 - ent["composite"])
        X       = np.stack(list(self._feat_buf), axis=0)
        nn_pred = self.nn.predict(X)
        null_rej, null_p, _ = self.null_t.test()
        wr     = float(np.mean(self._recent_wr)) if self._recent_wr else 0.5
        rl_s   = self.rl.state_index(ent["composite"], regime_id, wr,
                                      micro_f.get("volatility_burst", 0.0))
        rl_a, rl_c = self.rl.act(rl_s)

        fusion = fuse(self.cfg, self.symbol, ent, rl_c if rl_a == 1 else 0.0,
                      nn_pred, regime_id, regime_conf, micro_f,
                      null_rej, null_p, self.learner, self.even_cal, self.odd_cal)

        # Periodic state log
        now = time.time()
        if now - self._last_state_log > 15:
            self._log_state(fusion, rl_s)
            self._last_state_log = now
        if self._ticks_after_warmup % self.cfg.skip_summary_every == 0:
            self._log_skip_summary()

        if rl_a == 0:
            self._skip_counts["rl_idle"] += 1; return None
        if not fusion.tradeable:
            key = fusion.block_reason.split("|")[0].strip()[:35]
            self._skip_counts[key] += 1
            self._maybe_log_skip(fusion); return None

        return fusion

    def get_rl_state(self, fusion: FusionResult) -> int:
        return self.rl.state_index(
            fusion.entropy_score, fusion.regime,
            float(np.mean(self._recent_wr)) if self._recent_wr else 0.5,
            fusion.volatility_score)

    def after_trade(self, fusion: FusionResult, rl_s: int,
                    won: bool, profit: float, stake: float):
        self._recent_wr.append(1 if won else 0)
        reward = profit / stake
        next_s = self.rl.state_index(0.5, 0, float(np.mean(self._recent_wr)), 0.0)
        self.rl.update(reward, next_s)

        if len(self._feat_buf) == self.cfg.nn_input_window:
            X = np.stack(list(self._feat_buf), axis=0)
            p_even_target = (1.0 if (fusion.side == "even" and won) else
                             0.0 if (fusion.side == "even" and not won) else
                             0.0 if (fusion.side == "odd"  and won) else 1.0)
            y = np.array([p_even_target,
                          0.0 if won else 1.0,
                          1.0 if won else 0.3], dtype=np.float32)
            self.nn.record(X, y)

        if fusion.side == "even":
            self.even_cal.record(fusion.p_even, won)
        else:
            self.odd_cal.record(fusion.p_odd, won)

        self.learner.record({
            "nn_p_even":   fusion.p_even > 0.5,
            "markov_even": fusion.p_even > fusion.p_odd,
            "vol_burst":   fusion.volatility_score > 0.5,
        }, won, fusion.side)
        self.save_state()

    def _log_state(self, f: FusionResult, rl_s: int):
        log.info(
            f"[{self.symbol} tick={self._tick}] "
            f"regime={f.regime_name}({f.regime_stability:.2f}) "
            f"entropy={f.entropy_score:.3f} conf={f.final_confidence:.3f} "
            f"p_even={f.p_even:.3f} p_odd={f.p_odd:.3f} side={f.side or 'none'} "
            f"null={'REJ' if f.null_rejected else 'fail'}(p={f.null_p:.3f}) "
            f"eps={self.rl.epsilon:.3f} "
            f"block=[{f.block_reason[:70]}]\n"
            f"  {self.micro.parity_bias_str()}"
        )

    def _maybe_log_skip(self, f: FusionResult):
        now = time.time()
        if now - self._last_skip_log < self.cfg.skip_log_interval: return
        self._last_skip_log = now
        log.info(f"[{self.symbol}] SKIP {f.block_reason[:80]}")

    def _log_skip_summary(self):
        total = sum(self._skip_counts.values())
        if not total: return
        s = " | ".join(f"{k}:{v}({v/total*100:.0f}%)"
                       for k, v in self._skip_counts.most_common(6))
        log.info(f"[{self.symbol} SKIPS ticks={self._ticks_after_warmup}] total={total} | {s}")


# ─────────────────────────────────────────────────────────────────────────────
# DERIV WEBSOCKET CLIENT
# FIX: uses asyncio.get_running_loop() throughout — no deadlock.
# FIX: _listen() is an independent task — never blocked by trade execution.
# FIX: pending futures cleaned up on timeout.
# FIX: req_id allocated atomically before ws.send().
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self._ws       = None
        self._rid      = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._tick_cbs: Dict[str, Callable]       = {}
        self._connected = False
        self.balance:   float = 0.0

    async def connect(self):
        url = f"{self.cfg.api_url}?app_id={self.cfg.app_id}"
        self._ws = await websockets.connect(
            url, ping_interval=20, ping_timeout=10, max_size=2**20)
        self._connected = True
        asyncio.get_running_loop().create_task(self._listen())

    async def auth(self):
        r = await self._rpc({"authorize": self.cfg.api_token})
        if "error" in r:
            raise ConnectionError(r["error"]["message"])
        self.balance = float(r["authorize"].get("balance", 0))
        log.info(f"Auth OK | login={r['authorize'].get('loginid')} "
                 f"balance=${self.balance:.2f}")

    async def subscribe_ticks(self, symbol: str, cb: Callable):
        self._tick_cbs[symbol] = cb
        rid = self._next()
        await self._send({"ticks": symbol, "subscribe": 1, "req_id": rid})

    async def buy(self, symbol: str, side: str, stake: float) -> Optional[dict]:
        contract_type = "DIGITEVEN" if side == "even" else "DIGITODD"
        r = await self._rpc({
            "buy":   1,
            "price": str(stake),
            "parameters": {
                "amount":        str(stake),
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      self.cfg.currency,
                "duration":      self.cfg.duration,
                "duration_unit": self.cfg.duration_unit,
                "symbol":        symbol,
            },
        })
        if "error" in r:
            log.error(f"[{symbol}] Buy error: {r['error']['message']}")
            return None
        b            = r.get("buy", {})
        self.balance = float(b.get("balance_after", self.balance))
        return b

    async def contract_status(self, cid) -> Optional[dict]:
        r = await self._rpc({"proposal_open_contract": 1, "contract_id": int(cid)})
        return None if "error" in r else r.get("proposal_open_contract")

    async def profit_table_lookup(self, cid) -> Optional[dict]:
        r = await self._rpc({"profit_table": 1, "description": 1,
                              "sort": "DESC", "limit": 10})
        for t in r.get("profit_table", {}).get("transactions", []):
            if str(t.get("contract_id")) == str(cid):
                return t
        return None

    async def refresh_balance(self):
        r = await self._rpc({"balance": 1, "account": "current"})
        self.balance = float(r.get("balance", {}).get("balance", self.balance))

    async def disconnect(self):
        self._connected = False
        if self._ws:
            try: await self._ws.close()
            except Exception: pass

    @property
    def connected(self) -> bool: return self._connected

    def _next(self) -> int:
        self._rid += 1; return self._rid

    async def _rpc(self, payload: dict) -> dict:
        rid = self._next()
        payload["req_id"] = rid
        loop = asyncio.get_running_loop()
        fut  = loop.create_future()
        self._pending[rid] = fut
        await self._send(payload)
        try:
            return await asyncio.wait_for(fut, timeout=20.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)   # clean up — no leak
            log.warning(f"RPC timeout req_id={rid}")
            return {"error": {"message": "timeout"}}

    async def _send(self, payload: dict):
        await self._ws.send(json.dumps(payload))

    async def _listen(self):
        """
        Independent listener task. Never blocked by trade code.
        Dispatches ticks as separate tasks so slow callbacks don't stall it.
        """
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("msg_type")
                if msg_type == "tick":
                    tick_obj = msg.get("tick", {})
                    symbol   = tick_obj.get("symbol", "")
                    quote    = float(tick_obj.get("quote", 0))
                    if quote > 0 and symbol in self._tick_cbs:
                        asyncio.get_running_loop().create_task(
                            self._dispatch_tick(symbol, quote))
                    continue

                rid = msg.get("req_id")
                if rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                    continue

                if "error" in msg:
                    log.warning(f"WS error: {msg['error'].get('message','?')}")

        except Exception as e:
            log.error(f"WS listener: {e}")
        finally:
            self._connected = False
            log.warning("WS listener exited")

    async def _dispatch_tick(self, symbol: str, price: float):
        try:
            cb = self._tick_cbs.get(symbol)
            if cb:
                if asyncio.iscoroutinefunction(cb):
                    await cb(price)
                else:
                    cb(price)
        except Exception as e:
            log.error(f"[{symbol}] Tick dispatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BOT ORCHESTRATOR  (R_25 + R_10, independent engines)
# ─────────────────────────────────────────────────────────────────────────────

class Bot:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.client  = DerivClient(cfg)
        self.history = History(cfg.history_file)
        self._alive  = True

        self.engines: Dict[str, SymbolEngine] = {
            sym: SymbolEngine(sym, cfg) for sym in cfg.symbols
        }

    async def run(self):
        for eng in self.engines.values():
            eng.load_state()

        retry = 5
        while self._alive:
            try:
                log.info("Connecting to Deriv API...")
                await self.client.connect()
                await self.client.auth()

                for eng in self.engines.values():
                    eng.risk.set_balance(self.client.balance)

                for sym in self.cfg.symbols:
                    await self.client.subscribe_ticks(sym, self._make_cb(sym))
                    await asyncio.sleep(0.1)

                log.info(f"Subscribed to {self.cfg.symbols} — trading both")
                retry = 5
                while self._alive and self.client.connected:
                    await asyncio.sleep(1)
                if self._alive:
                    log.warning("Disconnected — reconnecting in 5s...")
                    await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Error: {e} — retry in {retry}s")
                await asyncio.sleep(retry)
                retry = min(retry * 2, 60)
        await self.client.disconnect()

    def _make_cb(self, symbol: str):
        async def _on_tick(price: float):
            await self._process_tick(symbol, price)
        return _on_tick

    async def _process_tick(self, symbol: str, price: float):
        eng    = self.engines[symbol]
        fusion = eng.on_tick(price)
        if fusion is None: return

        ok, reason = eng.risk.can_trade(self.client.balance)
        if not ok:
            eng._skip_counts[reason.split(":")[0]] += 1; return

        await self._execute(symbol, eng, fusion)

    async def _execute(self, symbol: str, eng: SymbolEngine, fusion: FusionResult):
        stake = max(
            min(eng.risk.current_stake,
                round(self.client.balance * self.cfg.max_balance_pct, 2)),
            self.cfg.base_stake)

        log.info(
            f"[{symbol}] TRADE {fusion.side.upper()} ${stake:.2f} "
            f"step={eng.risk._martingale_step}/{self.cfg.martingale_steps} "
            f"conf={fusion.final_confidence:.3f} "
            f"p_even={fusion.p_even:.3f} p_odd={fusion.p_odd:.3f} "
            f"markov_confirms_even={eng.micro.push.__doc__} "
            f"regime={fusion.regime_name} ent={fusion.entropy_score:.3f} "
            f"bal=${self.client.balance:.2f}"
        )

        rl_s = eng.get_rl_state(fusion)
        eng.risk.on_open()
        result = await self.client.buy(symbol, fusion.side, stake)
        if not result:
            eng.risk.release_lock(); return

        cid       = result.get("contract_id")
        buy_price = float(result.get("buy_price", stake))

        self.history.add({
            "ts":                datetime.utcnow().isoformat(),
            "symbol":            symbol,
            "tick":              eng._tick,
            "contract_id":       cid,
            "side":              fusion.side,
            "stake":             buy_price,
            "final_confidence":  fusion.final_confidence,
            "entropy_score":     fusion.entropy_score,
            "rl_confidence":     fusion.rl_confidence,
            "neural_confidence": fusion.neural_confidence,
            "regime_stability":  fusion.regime_stability,
            "transition_bias":   fusion.transition_bias,
            "p_even":            fusion.p_even,
            "p_odd":             fusion.p_odd,
            "regime":            fusion.regime_name,
            "null_p":            fusion.null_p,
        })

        # Settle in a separate task — never blocks tick processing
        asyncio.get_running_loop().create_task(
            self._settle(symbol, eng, cid, buy_price, fusion, rl_s, stake))

    async def _settle(self, symbol: str, eng: SymbolEngine,
                      cid, buy_price: float, fusion: FusionResult,
                      rl_s: int, stake: float):
        await asyncio.sleep(3)
        won = profit = None; source = "unknown"

        for _ in range(8):
            s = await self.client.contract_status(cid)
            if s:
                sold = (s.get("is_sold", False)
                        or s.get("status", "") in ("sold", "won", "lost"))
                if sold:
                    ap     = s.get("profit"); sp = s.get("sell_price")
                    profit = (float(ap) if ap is not None
                              else float(sp) - buy_price if sp else 0.0)
                    won    = profit > 0
                    source = "proposal_open_contract"; break
            await asyncio.sleep(3)

        if won is None:
            txn = await self.client.profit_table_lookup(cid)
            if txn:
                profit = float(txn.get("profit", 0))
                won    = profit > 0; source = "profit_table"
            else:
                log.warning(f"[{symbol}] Unconfirmed cid={cid}")
                await self.client.refresh_balance()
                eng.risk.release_lock(); return

        await self.client.refresh_balance()
        eng.risk.on_close(won, profit)
        eng.after_trade(fusion, rl_s, won, profit, stake)
        self.history.update_last(cid, won, profit, self.client.balance, source)

        st   = self.history.stats
        bsym = self.history.stats_by_symbol()
        log.info(
            f"[{symbol}] {'WIN' if won else 'LOSS'} "
            f"side={fusion.side} profit={profit:+.4f} "
            f"bal=${self.client.balance:.2f} | "
            f"ALL WR={st['win_rate']:.1%} n={st['n']} "
            f"even:{st['even_wr']:.1%}({st['even_n']}) "
            f"odd:{st['odd_wr']:.1%}({st['odd_n']}) "
            f"PnL={st['pnl']:+.4f} | "
            f"R25 WR={bsym.get('R_25',{}).get('win_rate',0):.1%}({bsym.get('R_25',{}).get('n',0)}) "
            f"R10 WR={bsym.get('R_10',{}).get('win_rate',0):.1%}({bsym.get('R_10',{}).get('n',0)})"
        )

    def shutdown(self):
        self._alive = False
        st = self.history.stats
        log.info(f"Shutdown | WR={st['win_rate']:.1%} n={st['n']} PnL={st['pnl']:+.4f}")
        for sym, eng in self.engines.items():
            total = sum(eng._skip_counts.values())
            top   = " | ".join(f"{k}:{v}" for k, v in eng._skip_counts.most_common(5))
            log.info(f"  [{sym}] ticks={eng._tick} skips={total} top=[{top}]")


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER — uses real digit frequency distributions from collected data
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(cfg: Config, n_ticks: int = 8000, seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    print("=" * 72)
    print("Backtest: R_25 + R_10 | DIGITEVEN / DIGITODD | 9-layer engine")
    print(f"Ticks: {n_ticks} per symbol | Calibrated from 125k real ticks")
    print("=" * 72)

    def gen_prices(sym: str, n: int) -> list:
        """
        Generates synthetic prices reflecting the real digit frequency
        distributions observed during data collection.
        R_25: digit 8 overrepresented (14%), digit 5 underrepresented (7%)
              even_rate ~0.557
        R_10: digit 5 overrepresented (13.8%), digit 9 underrepresented (6.8%)
              even_rate ~0.533
        """
        obs      = SYMBOL_DATA[sym]
        dig_freq = {
            "R_25": [0.124,0.092,0.114,0.092,0.110,0.070,0.082,0.090,0.140,0.086],
            "R_10": [0.122,0.092,0.116,0.074,0.090,0.138,0.116,0.084,0.100,0.068],
        }[sym]

        base   = {"R_25": 4500.0, "R_10": 6000.0}[sym]
        step   = {"R_25": 0.002,  "R_10": 0.001}[sym]
        prices = []
        # Inject occasional parity bias windows matching observed E→E rates
        ee = obs["E_to_E"]; oe = obs["O_to_E"]
        prev_parity = None
        for _ in range(n):
            base += random.gauss(0, step)
            # Sample digit according to observed distribution
            digit = random.choices(range(10), weights=dig_freq)[0]
            # Apply Markov transition bias to match observed transition rates
            if prev_parity is not None:
                trans_prob = ee if prev_parity == 0 else oe
                if random.random() < trans_prob:
                    # Force even digit
                    evens = [d for d in range(10) if d % 2 == 0]
                    digit = random.choices(evens, weights=[dig_freq[d] for d in evens])[0]
                else:
                    odds = [d for d in range(10) if d % 2 == 1]
                    digit = random.choices(odds, weights=[dig_freq[d] for d in odds])[0]
            prev_parity = digit % 2
            price = round(abs(base), 2) + digit * 0.001
            prices.append(price)
        return prices

    all_results = {}
    for sym in cfg.symbols:
        prices   = gen_prices(sym, n_ticks)
        eng      = SymbolEngine(sym, cfg)
        risk     = eng.risk
        risk.set_balance(1000.0)

        balance = 1000.0; bal_log = [balance]
        trades  = wins = even_t = odd_t = even_w = odd_w = 0
        skips: Counter = Counter()

        for i, price in enumerate(prices):
            fusion = eng.on_tick(price)
            if fusion is None: continue
            ok, _ = risk.can_trade(balance)
            if not ok: skips["risk"] += 1; continue
            if not fusion.tradeable:
                skips[fusion.block_reason.split("|")[0].strip()[:25]] += 1; continue

            stake = max(min(risk.current_stake,
                            round(balance * cfg.max_balance_pct, 2)),
                        cfg.base_stake)

            # Outcome from next tick digit
            nd = int(round(prices[i+1] * 100)) % 10 if i+1 < len(prices) \
                 else random.randint(0, 9)
            won    = (fusion.side == "even" and nd % 2 == 0) or \
                     (fusion.side == "odd"  and nd % 2 == 1)
            profit = stake * cfg.payout_ratio if won else -stake
            balance += profit; bal_log.append(balance)

            risk.on_close(won, profit)
            eng.after_trade(fusion, 0, won, profit, stake)
            trades += 1; wins += (1 if won else 0)
            if fusion.side == "even": even_t += 1; even_w += (1 if won else 0)
            else:                     odd_t  += 1; odd_w  += (1 if won else 0)

            if trades % 50 == 0:
                print(f"  [{sym}] tick={i:5d} trades={trades} "
                      f"WR={wins/trades:.1%} "
                      f"even:{even_w}/{even_t} odd:{odd_w}/{odd_t} "
                      f"bal=${balance:.2f}")

        wr_f  = wins / trades if trades else 0.0
        pnl   = balance - 1000.0
        be    = 1.0 / (1.0 + cfg.payout_ratio)
        peaks = np.maximum.accumulate(bal_log)
        dd    = float(np.max((peaks - bal_log) / (peaks + 1e-9))) if len(bal_log) > 1 else 0.0
        all_results[sym] = {"trades": trades, "wr": wr_f, "pnl": pnl, "dd": dd,
                            "even_wr": even_w/max(even_t,1), "odd_wr": odd_w/max(odd_t,1),
                            "even_n": even_t, "odd_n": odd_t, "skips": skips}
        print(f"\n  [{sym}]  Trades={trades} WR={wr_f:.1%} (BE={be:.1%} edge={wr_f-be:+.1%})")
        print(f"    Even WR={even_w/max(even_t,1):.1%}({even_t})  "
              f"Odd WR={odd_w/max(odd_t,1):.1%}({odd_t})")
        print(f"    PnL=${pnl:+.2f}  Max DD={dd:.1%}")
        total_s = sum(skips.values())
        if total_s:
            for g, cnt in skips.most_common(5):
                print(f"    Skip {g:35s}: {cnt} ({cnt/total_s*100:.0f}%)")
        print()

    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

def _start_health_server(bot: Bot):
    import http.server as _hs
    port = int(os.getenv("PORT", "8080"))

    class _H(_hs.BaseHTTPRequestHandler):

        def do_GET(self):
            st   = bot.history.stats
            bsym = bot.history.stats_by_symbol()

            if self.path == "/status":
                body = json.dumps({
                    "status":    "running",
                    "symbols":   bot.cfg.symbols,
                    "trades":    st["n"],
                    "win_rate":  round(st["win_rate"], 4),
                    "pnl":       st["pnl"],
                    "even_wr":   round(st["even_wr"], 4),
                    "even_n":    st["even_n"],
                    "odd_wr":    round(st["odd_wr"], 4),
                    "odd_n":     st["odd_n"],
                    "balance":   bot.client.balance,
                    "by_symbol": bsym,
                    "engines":   {
                        sym: {
                            "ticks":           eng._tick,
                            "epsilon":         round(eng.rl.epsilon, 4),
                            "markov":          eng.micro.parity_bias_str(),
                            "martingale_step": eng.risk._martingale_step,
                            "stake":           eng.risk.current_stake,
                            "paused":          eng.risk._paused,
                        }
                        for sym, eng in bot.engines.items()
                    },
                }, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(body)
                return

            # HTML dashboard
            sym_rows = ""
            for sym, eng in bot.engines.items():
                ss  = bsym.get(sym, {})
                pnl = ss.get("pnl", 0)
                sym_rows += (
                    f"<tr><td><strong>{sym}</strong></td>"
                    f"<td>{eng._tick}</td>"
                    f"<td class='{'pos' if ss.get('win_rate',0)>=0.513 else 'neg'}'>"
                    f"{ss.get('win_rate',0):.1%} ({ss.get('n',0)})</td>"
                    f"<td class='{'pos' if pnl>=0 else 'neg'}'>${pnl:+.4f}</td>"
                    f"<td>{eng.risk._martingale_step}/{bot.cfg.martingale_steps}</td>"
                    f"<td>${eng.risk.current_stake:.2f}</td>"
                    f"<td class='{'neg' if eng.risk._paused else 'pos'}'>"
                    f"{'PAUSED' if eng.risk._paused else 'OK'}</td></tr>"
                )
            pnl_sign = "+" if st["pnl"] >= 0 else ""
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>R25+R10 Even/Odd Bot</title>
<style>
body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:2rem;}}
h1{{color:#58a6ff;}} h3{{color:#58a6ff;margin-top:1.5rem;}}
.sub{{color:#8b949e;margin-bottom:1.5rem;font-size:0.9rem;}}
table{{border-collapse:collapse;width:100%;max-width:700px;margin-bottom:1rem;}}
td,th{{padding:0.4rem 0.8rem;border:1px solid #21262d;text-align:left;}}
th{{background:#161b22;color:#8b949e;font-weight:normal;}}
.pos{{color:#3fb950;}} .neg{{color:#f85149;}} .neu{{color:#d29922;}}
</style></head><body>
<h1>R_25 + R_10 Even/Odd Bot</h1>
<div class="sub">
  Data-calibrated from 125 611 ticks | DIGITEVEN / DIGITODD | 1-tick |
  Refreshes every 10s
</div>
<h3>Observed baselines (from data collection)</h3>
<table>
  <tr><th>Symbol</th><th>Even rate</th><th>E→E</th><th>O→E</th>
      <th>Markov stab</th><th>zscore rate</th></tr>
  <tr><td>R_25</td><td class="pos">55.7%</td><td>55.9%</td><td>55.5%</td>
      <td>0.114</td><td>8.9%</td></tr>
  <tr><td>R_10</td><td class="pos">53.3%</td><td>54.5%</td><td>52.0%</td>
      <td>0.064</td><td>8.9%</td></tr>
</table>
<h3>Live engine status</h3>
<table>
  <tr><th>Symbol</th><th>Ticks</th><th>Win rate (n)</th><th>P&L</th>
      <th>Martingale</th><th>Next stake</th><th>Status</th></tr>
  {sym_rows}
</table>
<h3>Combined stats</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total trades</td><td><strong>{st['n']}</strong></td></tr>
  <tr><td>Win rate</td>
      <td class="{'pos' if st['win_rate']>=0.513 else 'neg'}">
          <strong>{st['win_rate']:.1%}</strong>
          <span style="color:#8b949e"> (breakeven 51.3%)</span></td></tr>
  <tr><td>P&L</td>
      <td class="{'pos' if st['pnl']>=0 else 'neg'}">
          <strong>{pnl_sign}${st['pnl']:.4f}</strong></td></tr>
  <tr><td>Balance</td><td><strong>${bot.client.balance:.2f}</strong></td></tr>
  <tr><td>Even WR</td>
      <td class="{'pos' if st['even_wr']>=0.513 else 'neg'}">
          {st['even_wr']:.1%} ({st['even_n']} trades)</td></tr>
  <tr><td>Odd WR</td>
      <td class="{'pos' if st['odd_wr']>=0.513 else 'neg'}">
          {st['odd_wr']:.1%} ({st['odd_n']} trades)</td></tr>
</table>
<p style="color:#8b949e;font-size:0.8rem">
  JSON: <a href="/status" style="color:#58a6ff">/status</a>
</p>
</body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, *a): pass

    threading.Thread(
        target=_hs.HTTPServer(("", port), _H).serve_forever,
        daemon=True).start()
    log.info(f"Health server on :{port}  (/ = dashboard  /status = JSON)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def live(cfg: Config):
    if not cfg.api_token:
        log.error(
            "No API token.\n"
            "  export DERIV_API_TOKEN=your_token\n"
            "  then run:  python r25_r10_bot.py"
        )
        sys.exit(1)

    bot = Bot(cfg)

    def _sig(s, f):
        log.info("Shutdown signal received...")
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    log.info("=" * 72)
    log.info("R_25 + R_10 Even/Odd Bot — 9-Layer Engine — Data-Calibrated")
    log.info("Symbols : R_25 (primary) + R_10 (secondary)")
    log.info("          Selected from 186min / 125 611 ticks across 15 symbols")
    log.info("Contract: DIGITEVEN / DIGITODD | 1-tick expiry")
    log.info("Key data: R_25 even_rate=55.7%  E->E=55.9%  markov_stab=0.114")
    log.info("          R_10 even_rate=53.3%  E->E=54.5%  markov_stab=0.064")
    log.info(
        f"Stake   : " +
        " -> ".join(f"${cfg.base_stake * cfg.martingale_factor**s:.2f}"
                    for s in range(cfg.martingale_steps + 1)) +
        " -> halt"
    )
    log.info(
        f"Gates   : entropy<{cfg.entropy_threshold} "
        f"null_p<{cfg.null_hyp_p_value} "
        f"regime>{cfg.min_regime_stability} "
        f"conf>{cfg.min_final_conf} "
        f"p_even>{cfg.min_even_prob} "
        f"markov_margin>{cfg.markov_even_margin}"
    )
    log.info("=" * 72)

    _start_health_server(bot)
    await bot.run()


if __name__ == "__main__":
    cfg = Config()
    if "--backtest" in sys.argv:
        run_backtest(cfg)
    else:
        asyncio.run(live(cfg))
