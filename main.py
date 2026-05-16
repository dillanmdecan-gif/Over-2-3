# -*- coding: utf-8 -*-
"""
Adaptive Digits Bot — R_10 + R_25 + R_100
══════════════════════════════════════════
Trades all three symbols simultaneously with a built-in live data
collector that re-calibrates every engine threshold as market structure
shifts — no manual restarts needed.

HOW ADAPTATION WORKS
─────────────────────
Every tick on every symbol feeds a rolling SymbolProfile that tracks:
  • Live even_rate        (200-tick EMA)
  • Live Markov E→E, O→E  (rolling 200-tick window, Laplace-smoothed)
  • Live zscore_spike_rate (100-tick window)
  • Live streak_reversion  (100-tick window)
  • Live ranking_score     = markov_stability × zscore_spike_rate × streak_reversion

Every ADAPT_EVERY ticks (default 300) the SymbolProfile recomputes:
  • min_even_prob   ← live even_rate − 0.04  (4% discount, floor 0.50)
  • min_odd_prob    ← same mirror logic
  • markov_margin   ← max(0.02, live_markov_stability × 0.25)
  • entropy_thresh  ← stays fixed at 0.90 (stable across sessions)
  • null_p_value    ← 0.40 if digit_chi2_p < 0.30 else 0.55
  • The primary side default shifts with live even_rate:
      even_rate > 0.52 → primary = EVEN
      even_rate < 0.48 → primary = ODD
      0.48–0.52        → SYMMETRIC (Markov picks)

Logs a [ADAPT] line every recalibration showing what changed and why.
State is saved after every trade — survives restarts with full history.

SYMBOLS
───────
  R_10   — traded (even bias ~53%)
  R_25   — traded (even bias ~56%, strongest Markov stability)
  R_100  — traded (symmetric even ~49.5%, strong O→O momentum)

Run:
    export DERIV_API_TOKEN=your_token
    python adaptive_digits_bot.py

Dashboard:  http://localhost:8080/
JSON:       http://localhost:8080/status
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
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
import websockets

# ── UTF-8 stdout guard ────────────────────────────────────────────────────────
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
log = logging.getLogger("adaptbot")

# ─────────────────────────────────────────────────────────────────────────────
# STATIC SYMBOL LIST
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = ["R_10",  "R_100"]

# Bootstrap priors seeded from collected data (255 min / 191 739 ticks).
# These are STARTING points only — the live collector overwrites them quickly.
BOOTSTRAP = {
    "R_10": {
        "even_rate": 0.533, "E_to_E": 0.577, "O_to_E": 0.485,
        "markov_stability": 0.092, "zscore_spike_rate": 0.092,
        "streak_reversion": 0.282,
        "digit_freq": [0.116,0.124,0.084,0.096,0.148,0.056,0.120,0.088,0.104,0.064],
    },
    "R_25": {
        "even_rate": 0.559, "E_to_E": 0.597, "O_to_E": 0.512,
        "markov_stability": 0.109, "zscore_spike_rate": 0.092,
        "streak_reversion": 0.276,
        "digit_freq": [0.128,0.128,0.088,0.084,0.128,0.064,0.136,0.088,0.108,0.048],
    },
    "R_100": {
        "even_rate": 0.495, "E_to_E": 0.549, "O_to_E": 0.443,
        "markov_stability": 0.106, "zscore_spike_rate": 0.094,
        "streak_reversion": 0.290,
        "digit_freq": [0.084,0.120,0.120,0.060,0.116,0.068,0.072,0.112,0.128,0.120],
    },
}

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
# STATIC CONFIG  — things that don't self-adapt
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── API ───────────────────────────────────────────────────────────────────
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    app_id:    str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", "1089"))
    api_url:   str = "wss://ws.binaryws.com/websockets/v3"

    # ── SYMBOLS ───────────────────────────────────────────────────────────────
    symbols: List[str] = field(default_factory=lambda: SYMBOLS)

    # ── CONTRACT ──────────────────────────────────────────────────────────────
    duration:      int   = 1
    duration_unit: str   = "t"
    currency:      str   = "USD"
    payout_ratio:  float = 0.95

    # ── WARMUP — ticks before first trade ─────────────────────────────────────
    warmup_ticks: int = 120

    # ── ADAPTATION ────────────────────────────────────────────────────────────
    adapt_every:        int = 300    # recalibrate thresholds every N ticks
    adapt_window_short: int = 100    # short window: zscore, streak
    adapt_window_long:  int = 300    # long window: even_rate, Markov

    # ── LAYER WINDOWS (fixed) ─────────────────────────────────────────────────
    micro_window:       int = 30
    markov_window:      int = 60
    cluster_window:     int = 20
    momentum_window:    int = 15
    entropy_window:     int = 35
    perm_entropy_order: int = 4
    regime_window:      int = 40
    nn_input_window:    int = 25
    nn_hidden:          int = 32
    nn_lr:              float = 0.005
    nn_batch:           int   = 16
    null_hyp_window:    int   = 25
    cal_window:         int   = 50
    cal_recal_every:    int   = 25

    # ── RL ────────────────────────────────────────────────────────────────────
    rl_states:        int   = 64
    rl_alpha:         float = 0.15
    rl_gamma:         float = 0.90
    rl_epsilon_start: float = 0.20
    rl_epsilon_min:   float = 0.04
    rl_epsilon_decay: float = 0.997

    # ── FUSION WEIGHTS (adapted per symbol by AdaptiveLearner) ────────────────
    w_entropy:    float = 0.30
    w_rl:         float = 0.16
    w_neural:     float = 0.16
    w_regime:     float = 0.13
    w_transition: float = 0.15
    w_volatility: float = 0.10

    # ── FIXED ENTRY FLOORS — adaptation can only tighten, not exceed these ────
    # These are absolute floors. Adaptation raises these as evidence grows.
    min_final_conf:       float = 0.20
    min_regime_stability: float = 0.38
    entropy_threshold:    float = 0.90   # fixed — stable across all sessions

    # ── STAKE / MARTINGALE ────────────────────────────────────────────────────
    base_stake:        float = 0.35
    martingale_factor: float = 1.15
    martingale_steps:  int   = 3
    max_balance_pct:   float = 0.10

    # ── RISK ──────────────────────────────────────────────────────────────────
    loss_cooldown_ticks:    int   = 5
    max_consecutive_losses: int   = 3
    max_daily_loss_pct:     float = 0.15
    balance_guard_mult:     int   = 3

    # ── PERSISTENCE ───────────────────────────────────────────────────────────
    state_dir:    str = "."
    history_file: str = "adaptive_trades.csv"

    # ── LOGGING ───────────────────────────────────────────────────────────────
    skip_log_interval:  float = 30.0
    skip_summary_every: int   = 200


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SYMBOL PROFILE  — the inbuilt collector + threshold adapter
# One instance per symbol. Updated every tick. Recomputes thresholds
# every cfg.adapt_every ticks.
# ─────────────────────────────────────────────────────────────────────────────
class SymbolProfile:
    """
    Rolling live stats for one symbol.
    Computes and owns all adaptive thresholds.
    """

    def __init__(self, symbol: str, cfg: Config):
        self.symbol = symbol
        self.cfg    = cfg
        boot        = BOOTSTRAP[symbol]

        # ── Rolling windows for live stats ────────────────────────────────────
        self._digits_long:  deque = deque(maxlen=cfg.adapt_window_long)
        self._digits_short: deque = deque(maxlen=cfg.adapt_window_short)
        self._prices:       deque = deque(maxlen=cfg.adapt_window_long)
        self._vel_hist:     deque = deque(maxlen=cfg.adapt_window_short)

        # Markov counts for long window [prev][curr]
        ee = boot["E_to_E"]; oe = boot["O_to_E"]
        self._mk = [
            [ee * 20, (1 - ee) * 20],   # prev=even — seed with 20 pseudo-obs
            [oe * 20, (1 - oe) * 20],   # prev=odd
        ]
        self._mk_prev: Optional[int] = None

        # Streak tracking
        self._streak:              int = 0
        self._streak_prev_parity:  Optional[int] = None
        self._streak_reversions:   int = 0
        self._streak_total:        int = 0

        # zscore spikes
        self._zscore_spikes: int = 0
        self._zscore_checks: int = 0

        # ── Live computed stats (updated per adapt cycle) ─────────────────────
        self.live_even_rate:      float = boot["even_rate"]
        self.live_E_to_E:         float = boot["E_to_E"]
        self.live_O_to_E:         float = boot["O_to_E"]
        self.live_markov_stab:    float = boot["markov_stability"]
        self.live_zscore_rate:    float = boot["zscore_spike_rate"]
        self.live_streak_rev:     float = boot["streak_reversion"]
        self.live_ranking_score:  float = boot["markov_stability"] * boot["zscore_spike_rate"] * boot["streak_reversion"]
        self.live_digit_freq:     List[float] = boot["digit_freq"][:]
        self.live_null_p_value:   float = 0.55   # updated by adapt()
        self.live_chi2_p:         float = 1.0    # last chi2 p-value

        # ── Adaptive thresholds (what the engine actually uses) ───────────────
        self._compute_thresholds()

        # Internal tick counter for adaptation trigger
        self._ticks: int = 0

    # ── Per-tick update ───────────────────────────────────────────────────────
    def push(self, price: float):
        digit  = int(round(price * 100)) % 10
        parity = digit % 2

        self._prices.append(price)
        self._digits_long.append(digit)
        self._digits_short.append(digit)

        # Markov update
        if self._mk_prev is not None:
            self._mk[self._mk_prev][parity] += 1.0
        self._mk_prev = parity

        # Streak update
        if self._streak_prev_parity is None:
            self._streak = 1
        elif parity == self._streak_prev_parity:
            self._streak += 1
        else:
            if self._streak >= 3:
                self._streak_reversions += 1
            self._streak_total += 1
            self._streak = 1
        self._streak_prev_parity = parity

        # Velocity / zscore update
        prices = list(self._prices)
        if len(prices) >= 2:
            diff = abs(prices[-1] - prices[-2])
            self._vel_hist.append(diff)
            if len(self._vel_hist) >= 10:
                vel = list(self._vel_hist)
                mu = np.mean(vel); sig = np.std(vel) + 1e-12
                vz = (vel[-1] - mu) / sig
                self._zscore_checks += 1
                if abs(vz) > 1.5:
                    self._zscore_spikes += 1

        self._ticks += 1
        if self._ticks % self.cfg.adapt_every == 0:
            self._adapt()

    # ── Adaptation cycle ──────────────────────────────────────────────────────
    def _adapt(self):
        prev = {
            "even_rate":   self.live_even_rate,
            "markov_stab": self.live_markov_stab,
            "min_even":    self.min_even_prob,
            "min_odd":     self.min_odd_prob,
            "markov_margin": self.markov_even_margin,
            "null_p":      self.live_null_p_value,
            "primary":     self.primary_side,
        }

        # Recompute live stats
        dlong = list(self._digits_long)
        if len(dlong) >= 20:
            self.live_even_rate = sum(1 for d in dlong if d % 2 == 0) / len(dlong)

            # Markov from rolling window
            ee_row = self._mk[0]; oe_row = self._mk[1]
            ee_tot = ee_row[0] + ee_row[1]; oe_tot = oe_row[0] + oe_row[1]
            self.live_E_to_E = float(ee_row[0] / ee_tot) if ee_tot else 0.5
            self.live_O_to_E = float(oe_row[0] / oe_tot) if oe_tot else 0.5
            self.live_markov_stab = (abs(self.live_E_to_E - 0.5) +
                                     abs(self.live_O_to_E - 0.5))

            # Digit frequencies
            counts = np.bincount(dlong, minlength=10).astype(float)
            self.live_digit_freq = (counts / counts.sum()).tolist()

            # chi2 test on digit uniformity
            expected = np.full(10, len(dlong) / 10.0)
            _, self.live_chi2_p = scipy_stats.chisquare(counts, expected)

        # zscore spike rate and streak reversion from short window
        if self._zscore_checks > 0:
            self.live_zscore_rate = self._zscore_spikes / self._zscore_checks
        if self._streak_total > 0:
            self.live_streak_rev = self._streak_reversions / self._streak_total

        self.live_ranking_score = (self.live_markov_stab *
                                   self.live_zscore_rate *
                                   self.live_streak_rev)
        # null p-value: tighter when digit structure is clearer
        self.live_null_p_value = 0.40 if self.live_chi2_p < 0.30 else 0.55

        self._compute_thresholds()

        # Log what changed
        changes = []
        if abs(self.live_even_rate - prev["even_rate"]) > 0.005:
            changes.append(f"even_rate {prev['even_rate']:.3f}->{self.live_even_rate:.3f}")
        if abs(self.live_markov_stab - prev["markov_stab"]) > 0.005:
            changes.append(f"markov_stab {prev['markov_stab']:.3f}->{self.live_markov_stab:.3f}")
        if abs(self.min_even_prob - prev["min_even"]) > 0.005:
            changes.append(f"min_even {prev['min_even']:.3f}->{self.min_even_prob:.3f}")
        if abs(self.markov_even_margin - prev["markov_margin"]) > 0.005:
            changes.append(f"markov_margin {prev['markov_margin']:.3f}->{self.markov_even_margin:.3f}")
        if self.live_null_p_value != prev["null_p"]:
            changes.append(f"null_p {prev['null_p']:.2f}->{self.live_null_p_value:.2f}")
        if self.primary_side != prev["primary"]:
            changes.append(f"primary {prev['primary']}->{self.primary_side}")

        if changes:
            log.info(f"[{self.symbol} ADAPT tick={self._ticks}] " + " | ".join(changes))
        else:
            log.info(f"[{self.symbol} ADAPT tick={self._ticks}] "
                     f"stable — even={self.live_even_rate:.3f} "
                     f"markov={self.live_markov_stab:.3f} "
                     f"primary={self.primary_side} "
                     f"min_even={self.min_even_prob:.3f} "
                     f"null_p={self.live_null_p_value:.2f}")

    def _compute_thresholds(self):
        """
        Derive all adaptive thresholds from current live stats.
        Called at startup (from bootstrap) and every adapt_every ticks.
        """
        er = self.live_even_rate

        # Primary side
        if er > 0.522:
            self.primary_side = "even"
        elif er < 0.478:
            self.primary_side = "odd"
        else:
            self.primary_side = "symmetric"   # pure Markov pick

        # min_even_prob / min_odd_prob — live rate minus 4% discount, floor 0.50
        self.min_even_prob = max(0.500, round(er - 0.04, 3))
        self.min_odd_prob  = max(0.500, round((1.0 - er) - 0.04, 3))

        # markov_even_margin — 25% of live stability, min 0.02
        self.markov_even_margin = max(0.020, round(self.live_markov_stab * 0.25, 3))

        # zscore gate — 80% of live rate, min 0.055
        self.min_zscore_spike_rate = max(0.055, round(self.live_zscore_rate * 0.80, 3))

        # Whether this symbol has a meaningful even bias (used in side selection)
        self.has_even_bias = er >= 0.522
        self.has_odd_bias  = er <= 0.478

    def markov_priors(self) -> Tuple[float, float]:
        """Return (E_to_E, O_to_E) from the live rolling Markov."""
        return self.live_E_to_E, self.live_O_to_E

    def summary(self) -> str:
        return (f"even={self.live_even_rate:.3f} "
                f"E->E={self.live_E_to_E:.3f} O->E={self.live_O_to_E:.3f} "
                f"stab={self.live_markov_stab:.3f} "
                f"primary={self.primary_side} "
                f"min_even={self.min_even_prob:.3f} "
                f"null_p={self.live_null_p_value:.2f} "
                f"score={self.live_ranking_score:.5f}")

    def get_state(self) -> dict:
        return {
            "mk": [row[:] for row in self._mk],
            "live_even_rate":  self.live_even_rate,
            "live_E_to_E":     self.live_E_to_E,
            "live_O_to_E":     self.live_O_to_E,
            "live_markov_stab": self.live_markov_stab,
            "live_zscore_rate": self.live_zscore_rate,
            "live_streak_rev":  self.live_streak_rev,
            "zscore_spikes":   self._zscore_spikes,
            "zscore_checks":   self._zscore_checks,
            "streak_rev":      self._streak_reversions,
            "streak_tot":      self._streak_total,
        }

    def load_state(self, s: dict):
        self._mk              = s.get("mk", self._mk)
        self.live_even_rate   = s.get("live_even_rate",   self.live_even_rate)
        self.live_E_to_E      = s.get("live_E_to_E",      self.live_E_to_E)
        self.live_O_to_E      = s.get("live_O_to_E",      self.live_O_to_E)
        self.live_markov_stab = s.get("live_markov_stab", self.live_markov_stab)
        self.live_zscore_rate = s.get("live_zscore_rate", self.live_zscore_rate)
        self.live_streak_rev  = s.get("live_streak_rev",  self.live_streak_rev)
        self._zscore_spikes   = s.get("zscore_spikes",    0)
        self._zscore_checks   = s.get("zscore_checks",    0)
        self._streak_reversions = s.get("streak_rev",     0)
        self._streak_total      = s.get("streak_tot",     0)
        self._compute_thresholds()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — MICROSTRUCTURE ANALYZER
# Pre-seeded from the live SymbolProfile (not static BOOTSTRAP).
# ─────────────────────────────────────────────────────────────────────────────
class MicrostructureAnalyzer:

    def __init__(self, cfg: Config, profile: SymbolProfile):
        self.cfg     = cfg
        self.profile = profile

        self._prices:   deque = deque(maxlen=max(cfg.micro_window, cfg.cluster_window, 100))
        self._digits:   deque = deque(maxlen=cfg.markov_window + 10)
        self._diffs:    deque = deque(maxlen=cfg.micro_window)
        self._vel_hist: deque = deque(maxlen=cfg.micro_window)

        # Markov matrix pre-seeded from profile priors
        ee, oe = profile.live_E_to_E, profile.live_O_to_E
        self._mk = [[ee*10, (1-ee)*10], [oe*10, (1-oe)*10]]
        self._prev_parity: Optional[int] = None
        self._even_count: int = 0
        self._total:      int = 0

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
            self._diffs.append(diff); self._vel_hist.append(diff)

        if len(self._vel_hist) >= 10:
            vel  = list(self._vel_hist)
            mu   = np.mean(vel); sig = np.std(vel) + 1e-12
            vz   = float((vel[-1] - mu) / sig)
            f["tick_velocity_z"] = vz
            diffs = list(self._diffs); mid = len(diffs) // 2
            v1 = np.mean(diffs[:mid]) if mid else mu; v2 = np.mean(diffs[mid:])
            f["tick_acceleration_z"] = float((v2 - v1) / (sig + 1e-12))
        else:
            f["tick_velocity_z"] = f["tick_acceleration_z"] = 0.0

        f["volatility_burst"]     = f["tick_velocity_z"]
        f["momentum_exhaustion"]  = (self._momentum_exhaustion(prices)
                                     if len(prices) >= self.cfg.momentum_window else 0.0)
        f["reversal_compression"] = (self._reversal_compression(prices)
                                     if len(prices) >= 10 else 0.0)

        self._update_markov(parity)
        f["markov_even_bias"] = self._markov_bias(0)
        f["markov_odd_bias"]  = self._markov_bias(1)

        w = self.cfg.entropy_window
        if len(self._digits) >= w:
            recent = list(self._digits)[-w:]
            f["even_rate"] = sum(1 for d in recent if d % 2 == 0) / w
        else:
            f["even_rate"] = self.profile.live_even_rate

        f["cluster_density"] = (
            self._cluster_density(list(self._digits)[-self.cfg.cluster_window:])
            if len(self._digits) >= self.cfg.cluster_window else 0.0)

        # Gate fields — derived from live profile thresholds
        row_e = self._mk[0]; tot_e = row_e[0] + row_e[1]
        live_ee = float(row_e[0] / tot_e) if tot_e else self.profile.live_E_to_E
        f["markov_confirms_even"] = live_ee >= (0.50 + self.profile.markov_even_margin)
        f["zscore_spike_rate"]    = self.profile.live_zscore_rate
        f["zscore_gate_ok"]       = (self.profile.live_zscore_rate >=
                                     self.profile.min_zscore_spike_rate)
        f["even_rate_gate_ok"]    = f["even_rate"] >= (self.profile.live_even_rate - 0.04)
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
        mid = len(runs) // 2
        return float(np.clip(1.0 - min(np.mean(runs[mid:]) /
                                       (np.mean(runs[:mid]) + 1e-9), 1.0), 0, 1))

    def _reversal_compression(self, prices: list) -> float:
        w = prices[-20:]
        rev = sum(1 for i in range(1, len(w)-1)
                  if (w[i] > w[i-1]) != (w[i] < w[i+1]))
        return float(rev / max(len(w)-2, 1))

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
        chi2   = float(np.sum((counts - exp)**2 / (exp + 1e-9)))
        return float(np.clip(chi2 / (len(digits) * 9.0), 0, 1))

    def parity_bias_str(self) -> str:
        ee=self._mk[0][0]; eo=self._mk[0][1]
        oe=self._mk[1][0]; oo=self._mk[1][1]
        te=ee+eo+1e-9; to=oe+oo+1e-9
        lb=(self._even_count/max(self._total,1)-0.5)*100
        return (f"E->E:{ee/te:.3f} E->O:{eo/te:.3f} | "
                f"O->E:{oe/to:.3f} O->O:{oo/to:.3f} | bias:{lb:+.1f}%")

    def get_state(self) -> dict:
        return {"mk": [r[:] for r in self._mk],
                "even_count": self._even_count, "total": self._total}

    def load_state(self, s: dict):
        self._mk         = s["mk"]
        self._even_count = s.get("even_count", 0)
        self._total      = s.get("total", 0)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — ENTROPY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EntropyEngine:
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self._digits: deque = deque(maxlen=max(cfg.entropy_window, cfg.null_hyp_window, 150))

    def push(self, digit: int) -> dict:
        self._digits.append(digit)
        result = {"shannon":1.0,"permutation":1.0,"uniformity":1.0,"composite":1.0,"tradeable":False}
        if len(self._digits) < self.cfg.entropy_window:
            return result
        window  = list(self._digits)[-self.cfg.entropy_window:]
        shannon = self._shannon(window)
        perm    = self._perm_entropy(window, self.cfg.perm_entropy_order)
        unif    = self._uniformity_p(window)
        comp    = float(np.clip(0.45*shannon + 0.35*perm + 0.20*(1.0-unif), 0, 1))
        result.update({"shannon":round(shannon,4),"permutation":round(perm,4),
                       "uniformity":round(unif,4),"composite":round(comp,4),
                       "tradeable": comp < self.cfg.entropy_threshold})
        return result

    @staticmethod
    def _shannon(digits):
        counts = np.bincount(digits, minlength=10).astype(float)
        probs  = counts[counts>0]/counts.sum()
        return float(-np.sum(probs*np.log2(probs))/np.log2(10))

    @staticmethod
    def _perm_entropy(digits, order):
        if len(digits) < order+1: return 1.0
        pats  = Counter(tuple(sorted(range(order), key=lambda j: digits[i:i+order][j]))
                        for i in range(len(digits)-order+1))
        total = sum(pats.values())
        probs = [v/total for v in pats.values()]
        h     = -sum(p*math.log2(p) for p in probs if p > 0)
        mh    = math.log2(math.factorial(order))
        return float(h/mh) if mh > 0 else 1.0

    @staticmethod
    def _uniformity_p(digits):
        counts = np.bincount(digits, minlength=10).astype(float)
        _, p   = scipy_stats.chisquare(counts, np.full(10, len(digits)/10.0))
        return float(p)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — RL AGENT
# ─────────────────────────────────────────────────────────────────────────────
class RLAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._Q  = np.zeros((cfg.rl_states, 2))
        self._eps = cfg.rl_epsilon_start
        self._last_s = self._last_a = None

    def state_index(self, entropy, regime, win_rate, vol_z) -> int:
        e  = min(int(entropy*4), 3)
        r  = min(regime, 4)
        wr = min(int(win_rate*4), 3)
        vz = min(int((np.clip(vol_z,-3,3)+3)/1.5), 3)
        return int(np.clip(e*16+r*3+wr+vz, 0, self.cfg.rl_states-1))

    def act(self, state: int) -> Tuple[int, float]:
        a = (random.choice([0,1]) if random.random()<self._eps
             else int(np.argmax(self._Q[state])))
        q = self._Q[state]; qr = max(abs(q.max()-q.min()), 1e-9)
        c = float(np.clip((q[a]-q.min())/qr, 0, 1))
        self._last_s = state; self._last_a = a
        return a, c

    def update(self, reward, next_s):
        if self._last_s is None: return
        td = (reward + self.cfg.rl_gamma*np.max(self._Q[next_s])
              - self._Q[self._last_s][self._last_a])
        self._Q[self._last_s][self._last_a] += self.cfg.rl_alpha*td
        self._eps = max(self.cfg.rl_epsilon_min, self._eps*self.cfg.rl_epsilon_decay)

    @property
    def epsilon(self): return self._eps
    def get_state(self): return {"Q": self._Q.tolist(), "eps": self._eps}
    def load_state(self, s):
        self._Q = np.array(s["Q"]); self._eps = s["eps"]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — DIGIT NET
# ─────────────────────────────────────────────────────────────────────────────
class DigitNet:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        w=cfg.nn_input_window; h=cfg.nn_hidden; F=3
        rng=np.random.default_rng(42)
        self.W_conv=rng.normal(0,0.1,(h,F,3)); self.b_conv=np.zeros(h)
        pool_out=max((w-2)//2,1); gru_in=h*pool_out
        self.W_gru=rng.normal(0,0.1,(h,gru_in)); self.b_gru=np.zeros(h)
        self.W_att=rng.normal(0,0.1,(1,h)); self.b_att=np.zeros(1)
        self.W_out=rng.normal(0,0.1,(3,h)); self.b_out=np.zeros(3)
        self._buf_X=[]; self._buf_y=[]

    @staticmethod
    def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
    @staticmethod
    def _relu(x): return np.maximum(0,x)

    def _forward(self, X):
        T,F=X.shape; k=3; out_len=T-k+1
        cnn=np.zeros((self.cfg.nn_hidden,out_len))
        for t in range(out_len):
            cnn[:,t]=self._relu(np.einsum('hij,ij->h',self.W_conv,X[t:t+k,:].T)+self.b_conv)
        pool_len=max(out_len//2,1)
        pooled=np.array([cnn[:,i*2:i*2+2].max(axis=1) for i in range(pool_len)]).T
        flat=pooled.flatten(); gru_in=self.W_gru.shape[1]
        flat=(flat[:gru_in] if len(flat)>=gru_in else np.pad(flat,(0,gru_in-len(flat))))
        h_gru=self._relu(self.W_gru@flat+self.b_gru)
        h_att=h_gru*self._sig(self.W_att@h_gru+self.b_att)[0]
        return self._sig(self.W_out@h_att+self.b_out), h_att

    def predict(self, X):
        out,_=self._forward(X); p=float(out[0])
        return {"p_even":p,"p_odd":1.0-p,"noise":float(out[1]),"stability":float(out[2])}

    def record(self, X, y):
        self._buf_X.append(X.copy()); self._buf_y.append(y.copy())
        if len(self._buf_X)>self.cfg.nn_batch*4:
            self._buf_X.pop(0); self._buf_y.pop(0)
        if len(self._buf_X)>=self.cfg.nn_batch: self._train_step()

    def _train_step(self):
        idxs=random.sample(range(len(self._buf_X)),min(self.cfg.nn_batch,len(self._buf_X)))
        gW=np.zeros_like(self.W_out); gb=np.zeros_like(self.b_out)
        for i in idxs:
            out,h=self._forward(self._buf_X[i]); err=out-self._buf_y[i]
            gW+=np.outer(err,h); gb+=err
        n=len(idxs); self.W_out-=self.cfg.nn_lr*gW/n; self.b_out-=self.cfg.nn_lr*gb/n

    def get_state(self):
        return {k:getattr(self,k).tolist() for k in
                ("W_conv","b_conv","W_gru","b_gru","W_att","b_att","W_out","b_out")}
    def load_state(self, s):
        for k,v in s.items(): setattr(self,k,np.array(v))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class RegimeDetector:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._prices=deque(maxlen=cfg.regime_window)
        self._vol_h=deque(maxlen=cfg.regime_window)

    def push(self, price, entropy_score) -> Tuple[int, float]:
        self._prices.append(price); prices=list(self._prices)
        if len(prices)<15: return REGIME_STABLE,0.50
        diffs=np.diff(prices); vol=float(np.std(diffs)); self._vol_h.append(vol)
        ac=(float(np.corrcoef(diffs[:-1],diffs[1:])[0,1]) if len(diffs)>=10 else 0.0)
        vol_z=(float((vol-np.mean(list(self._vol_h)))/(np.std(list(self._vol_h))+1e-12))
               if len(self._vol_h)>=10 else 0.0)
        if   vol_z>2.0: return REGIME_VOL_EXPAND, min(0.5+vol_z*0.1,1.0)
        elif entropy_score<0.12: return REGIME_CHAOTIC, 0.70
        elif ac>0.30:   return REGIME_TRENDING,  min(0.5+ac,1.0)
        elif ac<-0.30:  return REGIME_REVERTING, min(0.5+abs(ac),1.0)
        else:           return REGIME_STABLE,    max(0.5,entropy_score)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────
class Calibrator:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._A=1.0; self._B=0.0
        self._hist=deque(maxlen=cfg.cal_window); self._since=0

    def calibrate(self, p):
        x=self._A*p+self._B
        return float(1.0/(1.0+math.exp(-max(-20,min(20,x)))))

    def record(self, p_raw, won):
        self._hist.append((p_raw,1.0 if won else 0.0)); self._since+=1
        if self._since>=self.cfg.cal_recal_every: self._refit(); self._since=0

    def _refit(self):
        if len(self._hist)<10: return
        data=list(self._hist); n=len(data)
        w=np.array([math.exp(-0.05*(n-1-i)) for i in range(n)]); w/=w.sum()
        ps=np.array([d[0] for d in data]); ys=np.array([d[1] for d in data])
        A,B=self._A,self._B
        for _ in range(60):
            pc=1.0/(1.0+np.exp(-(A*ps+B))); err=pc-ys
            A-=0.10*float(np.sum(w*err*ps)); B-=0.10*float(np.sum(w*err))
        self._A,self._B=A,B

    def get_state(self): return {"A":self._A,"B":self._B,"hist":list(self._hist)}
    def load_state(self,s):
        self._A=s["A"]; self._B=s["B"]
        self._hist=deque(s.get("hist",[]),maxlen=self.cfg.cal_window)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 8 — NULL HYPOTHESIS TESTER
# Uses live null_p_value from SymbolProfile (adapts with chi2 structure).
# ─────────────────────────────────────────────────────────────────────────────
class NullHypothesisTester:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._digits=deque(maxlen=cfg.null_hyp_window)

    def push(self, digit): self._digits.append(digit)

    def test(self, null_p_value: float) -> Tuple[bool, float, str]:
        digits=list(self._digits)
        if len(digits)<self.cfg.null_hyp_window: return False,1.0,"insufficient_data"
        counts=np.bincount(digits,minlength=10).astype(float)
        _,p_chi=scipy_stats.chisquare(counts,np.full(10,len(digits)/10.0))
        binary=[1 if d%2==0 else 0 for d in digits]
        p_runs=self._runs_test(binary)
        p_comb=min(float(p_chi),float(p_runs))
        return p_comb<null_p_value, p_comb, "chi2+runs"

    @staticmethod
    def _runs_test(b):
        n1=sum(b); n2=len(b)-n1
        if n1==0 or n2==0: return 1.0
        runs=1+sum(1 for i in range(1,len(b)) if b[i]!=b[i-1]); n=len(b)
        mu=1+2*n1*n2/n; s2=2*n1*n2*(2*n1*n2-n)/(n*n*(n-1)+1e-9)
        if s2<=0: return 1.0
        return float(2*(1-scipy_stats.norm.cdf(abs((runs-mu)/math.sqrt(s2)))))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 9 — ADAPTIVE LEARNER
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveLearner:
    FEATURES = ["entropy","markov_even","markov_odd","momentum",
                "reversal","cluster","vol_burst","nn_p_even","regime_stability"]

    def __init__(self):
        self._weights  = {f:1.0 for f in self.FEATURES}
        self._scores   = {f:deque(maxlen=50) for f in self.FEATURES}
        self._outcomes = deque(maxlen=100)
        self._side_wins= {"even":0,"odd":0,"total":0}
        self._drift    = False

    def record(self, preds, won, side):
        self._outcomes.append(1 if won else 0)
        self._side_wins["total"]+=1
        if won: self._side_wins[side]=self._side_wins.get(side,0)+1
        for feat,pred in preds.items():
            if feat not in self._scores: continue
            self._scores[feat].append(1 if (pred==won) else 0)
            acc=float(np.mean(self._scores[feat])) if self._scores[feat] else 0.5
            self._weights[feat]=float(np.clip(
                self._weights[feat]*(1.0+0.05*(acc-0.5)),0.1,3.0))
        if len(self._outcomes)>=20:
            acc=float(np.mean(list(self._outcomes)[-20:]))
            self._drift=acc<0.45
            if self._drift:
                log.warning(f"[DRIFT] Recent acc<45% — penalising weights")
                for f in self._weights: self._weights[f]=max(0.1,self._weights[f]*0.85)

    def weight(self, f): return self._weights.get(f,1.0)
    @property
    def drift(self): return self._drift
    @property
    def recent_accuracy(self): return float(np.mean(self._outcomes)) if self._outcomes else 0.0
    def get_state(self): return {"weights":dict(self._weights),"side_wins":self._side_wins,
                                  "scores":{k:list(v) for k,v in self._scores.items()}}
    def load_state(self,s):
        self._weights=s.get("weights",{f:1.0 for f in self.FEATURES})
        self._side_wins=s.get("side_wins",{"even":0,"odd":0,"total":0})
        for f,vals in s.get("scores",{}).items():
            if f in self._scores: self._scores[f]=deque(vals,maxlen=50)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — CONFIDENCE FUSION
# All thresholds pulled from the live SymbolProfile (not static Config).
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
    p_even:  float; p_odd: float; side: str
    regime: int; regime_name: str
    null_rejected: bool; null_p: float
    tradeable: bool; block_reason: str
    # Snapshot of adaptive thresholds used for this decision
    snap_min_even:   float = 0.0
    snap_min_odd:    float = 0.0
    snap_markov_margin: float = 0.0
    snap_null_p:     float = 0.0
    snap_primary:    str   = ""


def fuse(cfg: Config,
         profile:    SymbolProfile,
         entropy:    dict,
         rl_conf:    float,
         nn_pred:    dict,
         regime_id:  int,
         regime_conf: float,
         micro:      dict,
         null_rej:   bool,
         null_p:     float,
         learner:    AdaptiveLearner,
         even_cal:   Calibrator,
         odd_cal:    Calibrator) -> FusionResult:

    # Snapshot current adaptive thresholds for logging
    snap_min_even      = profile.min_even_prob
    snap_min_odd       = profile.min_odd_prob
    snap_markov_margin = profile.markov_even_margin
    snap_null_p        = profile.live_null_p_value
    snap_primary       = profile.primary_side

    entropy_score = float(1.0 - entropy["composite"])
    vol_z         = micro.get("volatility_burst", 0.0)
    vol_score     = float(np.clip(1.0 - abs(vol_z) / 3.0, 0, 1))
    mb_even       = micro.get("markov_even_bias", 0.0)
    mb_odd        = micro.get("markov_odd_bias",  0.0)
    trans_bias    = float(np.clip(max(abs(mb_even), abs(mb_odd)) * 2.0, 0, 1))

    p_even_cal = even_cal.calibrate(nn_pred["p_even"])
    p_odd_cal  = odd_cal.calibrate(nn_pred["p_odd"])
    best_p     = max(p_even_cal, p_odd_cal)
    neural_conf = float(np.clip((best_p - 0.5) * 2.0, 0, 1))

    w_e = cfg.w_entropy    * learner.weight("entropy")
    w_r = cfg.w_rl         * learner.weight("nn_p_even")
    w_n = cfg.w_neural     * learner.weight("nn_p_even")
    w_g = cfg.w_regime
    w_t = cfg.w_transition * max(learner.weight("markov_even"), learner.weight("markov_odd"))
    w_v = cfg.w_volatility * learner.weight("vol_burst")
    tot = w_e+w_r+w_n+w_g+w_t+w_v or 1.0

    conf = (w_e*entropy_score + w_r*rl_conf + w_n*neural_conf
            + w_g*regime_conf + w_t*trans_bias + w_v*vol_score) / tot
    if learner.drift: conf *= 0.70
    conf = float(np.clip(conf, 0, 1))

    # ── ADAPTIVE SIDE SELECTION ───────────────────────────────────────────────
    markov_clearly_odd  = (mb_odd - mb_even) > snap_markov_margin
    markov_clearly_even = (mb_even - mb_odd) > snap_markov_margin or \
                          micro.get("markov_confirms_even", False)

    if snap_primary == "even":
        # Even-biased symbol (R_10, R_25): default even, override to odd only if Markov clearly says so
        markov_side = "odd" if markov_clearly_odd else "even"
    elif snap_primary == "odd":
        # Odd-biased symbol: default odd, override to even only if Markov clearly says so
        markov_side = "even" if markov_clearly_even else "odd"
    else:
        # Symmetric (R_100 when near 50%): pure Markov
        if markov_clearly_odd:   markov_side = "odd"
        elif markov_clearly_even: markov_side = "even"
        else:                     markov_side = ""

    nn_side = ("even" if p_even_cal >= p_odd_cal else "odd") \
               if abs(p_even_cal - p_odd_cal) > 0.02 else ""

    if markov_side and nn_side:
        side = markov_side if markov_side == nn_side else markov_side
    elif markov_side:
        side = markov_side
    elif nn_side:
        side = nn_side
    else:
        side = ""

    best_p_cal = p_even_cal if side=="even" else p_odd_cal if side=="odd" else 0.0

    # ── GATES (using live adaptive thresholds from profile) ───────────────────
    block = []
    if not entropy["tradeable"]:
        block.append(f"entropy={entropy['composite']:.3f}>={cfg.entropy_threshold}")
    if not null_rej:
        block.append(f"null_p={null_p:.3f}>={snap_null_p:.2f}")
    if regime_conf < cfg.min_regime_stability:
        block.append(f"regime={regime_conf:.3f}<{cfg.min_regime_stability}")
    if conf < cfg.min_final_conf:
        block.append(f"conf={conf:.3f}<{cfg.min_final_conf}")
    if not side:
        block.append("no_side")
    elif side == "even" and best_p_cal < snap_min_even:
        block.append(f"p_even={best_p_cal:.3f}<{snap_min_even:.3f}")
    elif side == "odd" and best_p_cal < snap_min_odd:
        block.append(f"p_odd={best_p_cal:.3f}<{snap_min_odd:.3f}")
    if not micro.get("zscore_gate_ok", True):
        block.append(f"zscore={profile.live_zscore_rate:.3f}<{profile.min_zscore_spike_rate:.3f}")
    if learner.drift:
        block.append("drift")

    return FusionResult(
        final_confidence  = round(conf,4),
        entropy_score     = round(entropy_score,4),
        rl_confidence     = round(rl_conf,4),
        neural_confidence = round(neural_conf,4),
        regime_stability  = round(regime_conf,4),
        transition_bias   = round(trans_bias,4),
        volatility_score  = round(vol_score,4),
        p_even            = round(p_even_cal,4),
        p_odd             = round(p_odd_cal,4),
        side              = side,
        regime            = regime_id,
        regime_name       = REGIME_NAMES[regime_id],
        null_rejected     = null_rej,
        null_p            = round(null_p,4),
        tradeable         = len(block)==0,
        block_reason      = " | ".join(block) if block else "ok",
        snap_min_even     = snap_min_even,
        snap_min_odd      = snap_min_odd,
        snap_markov_margin = snap_markov_margin,
        snap_null_p       = snap_null_p,
        snap_primary      = snap_primary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._martingale_step=0; self._consec_losses=0
        self._in_trade=False; self._paused=False; self._pause_reason=""
        self._start_balance=None; self._daily_pnl=0.0; self._cooldown_ticks=0

    def set_balance(self, b):
        if self._start_balance is None: self._start_balance=b

    @property
    def current_stake(self):
        return round(self.cfg.base_stake*(self.cfg.martingale_factor**
                     min(self._martingale_step,self.cfg.martingale_steps)),2)

    def tick(self):
        if self._cooldown_ticks>0: self._cooldown_ticks-=1

    def can_trade(self, balance) -> Tuple[bool, str]:
        if self._in_trade:    return False,"in_trade"
        if self._paused:      return False,f"paused:{self._pause_reason}"
        if self._cooldown_ticks>0: return False,f"cooldown:{self._cooldown_ticks}t"
        if self._consec_losses>=self.cfg.max_consecutive_losses:
            self._paused=True; self._pause_reason=f"{self._consec_losses}_losses"
            return False,f"paused:{self._pause_reason}"
        if self._start_balance:
            if self._daily_pnl < -(self._start_balance*self.cfg.max_daily_loss_pct):
                self._paused=True; self._pause_reason="daily_loss_cap"
                return False,"paused:daily_loss_cap"
        if balance>0 and balance<self.cfg.base_stake*self.cfg.balance_guard_mult:
            return False,f"balance_too_low:{balance:.2f}"
        return True,"ok"

    def on_open(self): self._in_trade=True

    def on_close(self, won, profit):
        self._in_trade=False; self._daily_pnl+=profit
        if won:
            self._martingale_step=0; self._consec_losses=0; self._cooldown_ticks=0
        else:
            self._consec_losses+=1
            self._martingale_step=min(self._martingale_step+1,self.cfg.martingale_steps)
            self._cooldown_ticks=self.cfg.loss_cooldown_ticks

    def release_lock(self): self._in_trade=False

    def reset(self):
        self._paused=False; self._consec_losses=0
        self._martingale_step=0; self._cooldown_ticks=0


# ─────────────────────────────────────────────────────────────────────────────
# TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class History:
    COLS = ["ts","symbol","tick","contract_id","side","stake",
            "final_confidence","p_even","p_odd","regime","entropy_score",
            "snap_primary","snap_min_even","snap_null_p",
            "won","profit","balance","settle_source"]

    def __init__(self, path):
        self.path=path; self._rows=[]
        if not os.path.exists(path):
            with open(path,"w",newline="") as f:
                csv.DictWriter(f,fieldnames=self.COLS).writeheader()

    def add(self, row):
        self._rows.append(row)
        with open(self.path,"a",newline="") as f:
            csv.DictWriter(f,fieldnames=self.COLS).writerow(
                {c:row.get(c,"") for c in self.COLS})

    def update_last(self, cid, won, profit, balance, source):
        for r in reversed(self._rows):
            if str(r.get("contract_id"))==str(cid):
                r.update({"won":won,"profit":round(profit,5),
                           "balance":round(balance,4),"settle_source":source})
                self._rewrite(); return

    def _rewrite(self):
        with open(self.path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=self.COLS); w.writeheader()
            for r in self._rows: w.writerow({c:r.get(c,"") for c in self.COLS})

    @property
    def stats(self):
        done=[r for r in self._rows if r.get("won")!=""]
        if not done: return {"n":0,"win_rate":0.0,"pnl":0.0,"even_wr":0.0,
                              "odd_wr":0.0,"even_n":0,"odd_n":0}
        wins=[r for r in done if r.get("won") in (True,"True")]
        pnl=sum(float(r.get("profit",0) or 0) for r in done)
        en=[r for r in done if r.get("side")=="even"]
        od=[r for r in done if r.get("side")=="odd"]
        ew=sum(1 for r in en if r.get("won") in (True,"True"))
        ow=sum(1 for r in od if r.get("won") in (True,"True"))
        return {"n":len(done),"win_rate":len(wins)/len(done),"pnl":round(pnl,4),
                "even_wr":ew/max(len(en),1),"odd_wr":ow/max(len(od),1),
                "even_n":len(en),"odd_n":len(od)}

    def stats_by_symbol(self):
        result={}
        for sym in SYMBOLS:
            rows=[r for r in self._rows if r.get("symbol")==sym and r.get("won")!=""]
            if not rows: result[sym]={"n":0,"win_rate":0.0,"pnl":0.0}; continue
            wins=[r for r in rows if r.get("won") in (True,"True")]
            result[sym]={"n":len(rows),"win_rate":len(wins)/len(rows),
                         "pnl":round(sum(float(r.get("profit",0) or 0) for r in rows),4)}
        return result


# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL ENGINE — all 9 layers + live SymbolProfile
# ─────────────────────────────────────────────────────────────────────────────
class SymbolEngine:

    def __init__(self, symbol: str, cfg: Config):
        self.symbol  = symbol
        self.cfg     = cfg
        self.profile = SymbolProfile(symbol, cfg)     # inbuilt live collector
        self.micro   = MicrostructureAnalyzer(cfg, self.profile)
        self.entropy = EntropyEngine(cfg)
        self.rl      = RLAgent(cfg)
        self.nn      = DigitNet(cfg)
        self.regime  = RegimeDetector(cfg)
        self.even_cal= Calibrator(cfg)
        self.odd_cal = Calibrator(cfg)
        self.null_t  = NullHypothesisTester(cfg)
        self.learner = AdaptiveLearner()
        self.risk    = RiskManager(cfg)

        self._feat_buf  = deque(maxlen=cfg.nn_input_window)
        self._recent_wr = deque(maxlen=20)
        self._tick      = 0
        self._skip_counts: Counter = Counter()
        self._last_skip_log  = 0.0
        self._last_state_log = 0.0
        self._ticks_after_warmup = 0

        self.last_fusion: Optional[FusionResult] = None

    def state_path(self):
        return os.path.join(self.cfg.state_dir, f"{self.symbol}_adaptive_state.pkl")

    def save_state(self):
        state = {
            "version":  4,
            "symbol":   self.symbol,
            "saved_at": datetime.utcnow().isoformat(),
            "rl":       self.rl.get_state(),
            "nn":       self.nn.get_state(),
            "even_cal": self.even_cal.get_state(),
            "odd_cal":  self.odd_cal.get_state(),
            "learner":  self.learner.get_state(),
            "micro":    self.micro.get_state(),
            "profile":  self.profile.get_state(),   # save live collector state
        }
        path=self.state_path(); tmp=path+".tmp"
        with open(tmp,"wb") as f: pickle.dump(state,f,protocol=4)
        os.replace(tmp,path)

    def load_state(self):
        path=self.state_path()
        if not os.path.exists(path):
            log.info(f"[{self.symbol}] No state — starting fresh with bootstrap priors")
            return
        try:
            with open(path,"rb") as f: state=pickle.load(f)
            self.rl.load_state(state["rl"])
            self.nn.load_state(state["nn"])
            self.even_cal.load_state(state["even_cal"])
            self.odd_cal.load_state(state["odd_cal"])
            self.learner.load_state(state["learner"])
            self.micro.load_state(state["micro"])
            if "profile" in state:
                self.profile.load_state(state["profile"])
            log.info(f"[{self.symbol}] State loaded — "
                     f"saved {state.get('saved_at','?')} | "
                     f"profile: {self.profile.summary()}")
        except Exception as e:
            log.warning(f"[{self.symbol}] State load failed: {e} — fresh start")

    def on_tick(self, price: float) -> Optional[FusionResult]:
        self._tick += 1
        self.risk.tick()
        digit  = int(round(price * 100)) % 10
        parity = digit % 2

        # Feed the live collector FIRST (may trigger adaptation)
        self.profile.push(price)

        micro_f = self.micro.push(price)
        ent     = self.entropy.push(digit)
        self.null_t.push(digit)

        nn_vec = np.array([digit/9.0, float(parity), price%1.0], dtype=np.float32)
        self._feat_buf.append(nn_vec)

        if self._tick < self.cfg.warmup_ticks:
            if self._tick % 30 == 0:
                log.info(f"[{self.symbol}] Warmup {self._tick}/{self.cfg.warmup_ticks} | "
                         f"{self.profile.summary()}")
            return None
        if len(self._feat_buf) < self.cfg.nn_input_window:
            return None

        self._ticks_after_warmup += 1
        regime_id, regime_conf = self.regime.push(price, 1.0 - ent["composite"])
        X       = np.stack(list(self._feat_buf), axis=0)
        nn_pred = self.nn.predict(X)

        # Null test uses LIVE null_p_value from profile
        null_rej, null_p, _ = self.null_t.test(self.profile.live_null_p_value)

        wr    = float(np.mean(self._recent_wr)) if self._recent_wr else 0.5
        rl_s  = self.rl.state_index(ent["composite"], regime_id, wr,
                                     micro_f.get("volatility_burst", 0.0))
        rl_a, rl_c = self.rl.act(rl_s)

        fusion = fuse(self.cfg, self.profile, ent,
                      rl_c if rl_a == 1 else 0.0,
                      nn_pred, regime_id, regime_conf, micro_f,
                      null_rej, null_p, self.learner, self.even_cal, self.odd_cal)
        self.last_fusion = fusion

        now = time.time()
        if now - self._last_state_log > 20:
            log.info(f"[{self.symbol} t={self._tick}] "
                     f"regime={fusion.regime_name}({fusion.regime_stability:.2f}) "
                     f"ent={fusion.entropy_score:.3f} conf={fusion.final_confidence:.3f} "
                     f"side={fusion.side or 'none'} p_e={fusion.p_even:.3f} "
                     f"primary={fusion.snap_primary} "
                     f"null={'REJ' if fusion.null_rejected else 'fail'}(p={fusion.null_p:.3f}) "
                     f"block=[{fusion.block_reason[:60]}]")
            self._last_state_log = now

        if self._ticks_after_warmup % self.cfg.skip_summary_every == 0:
            self._log_skip_summary()

        if rl_a == 0:
            self._skip_counts["rl_idle"] += 1; return None
        if not fusion.tradeable:
            key = fusion.block_reason.split("|")[0].strip()[:35]
            self._skip_counts[key] += 1
            if now - self._last_skip_log > self.cfg.skip_log_interval:
                self._last_skip_log = now
                log.info(f"[{self.symbol}] SKIP {fusion.block_reason[:80]}")
            return None

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
            X=np.stack(list(self._feat_buf),axis=0)
            pet=(1.0 if (fusion.side=="even" and won) else
                 0.0 if (fusion.side=="even" and not won) else
                 0.0 if (fusion.side=="odd"  and won) else 1.0)
            y=np.array([pet,0.0 if won else 1.0,1.0 if won else 0.3],dtype=np.float32)
            self.nn.record(X,y)
        if fusion.side=="even": self.even_cal.record(fusion.p_even,won)
        else:                   self.odd_cal.record(fusion.p_odd,won)
        self.learner.record({"nn_p_even":fusion.p_even>0.5,
                             "markov_even":fusion.p_even>fusion.p_odd,
                             "vol_burst":fusion.volatility_score>0.5},won,fusion.side)
        self.save_state()

    def _log_skip_summary(self):
        total=sum(self._skip_counts.values())
        if not total: return
        s=" | ".join(f"{k}:{v}({v/total*100:.0f}%)"
                     for k,v in self._skip_counts.most_common(5))
        log.info(f"[{self.symbol} SKIPS t={self._ticks_after_warmup}] total={total} | {s}")


# ─────────────────────────────────────────────────────────────────────────────
# DERIV WEBSOCKET CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class DerivClient:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._ws=None; self._rid=0
        self._pending:  Dict[int, asyncio.Future] = {}
        self._tick_cbs: Dict[str, Callable]        = {}
        self._connected=False; self.balance=0.0

    async def connect(self):
        url=f"{self.cfg.api_url}?app_id={self.cfg.app_id}"
        self._ws=await websockets.connect(url,ping_interval=20,
                                           ping_timeout=10,max_size=2**20)
        self._connected=True
        asyncio.get_running_loop().create_task(self._listen())

    async def auth(self):
        r=await self._rpc({"authorize":self.cfg.api_token})
        if "error" in r: raise ConnectionError(r["error"]["message"])
        self.balance=float(r["authorize"].get("balance",0))
        log.info(f"Auth OK | {r['authorize'].get('loginid')} bal=${self.balance:.2f}")

    async def subscribe_ticks(self, symbol, cb):
        self._tick_cbs[symbol]=cb
        rid=self._next()
        await self._send({"ticks":symbol,"subscribe":1,"req_id":rid})

    async def buy(self, symbol, side, stake) -> Optional[dict]:
        r=await self._rpc({"buy":1,"price":str(stake),"parameters":{
            "amount":str(stake),"basis":"stake",
            "contract_type":"DIGITEVEN" if side=="even" else "DIGITODD",
            "currency":self.cfg.currency,"duration":self.cfg.duration,
            "duration_unit":self.cfg.duration_unit,"symbol":symbol}})
        if "error" in r:
            log.error(f"[{symbol}] Buy: {r['error']['message']}"); return None
        b=r.get("buy",{}); self.balance=float(b.get("balance_after",self.balance)); return b

    async def contract_status(self, cid) -> Optional[dict]:
        r=await self._rpc({"proposal_open_contract":1,"contract_id":int(cid)})
        return None if "error" in r else r.get("proposal_open_contract")

    async def profit_table_lookup(self, cid) -> Optional[dict]:
        r=await self._rpc({"profit_table":1,"description":1,"sort":"DESC","limit":10})
        for t in r.get("profit_table",{}).get("transactions",[]):
            if str(t.get("contract_id"))==str(cid): return t
        return None

    async def refresh_balance(self):
        r=await self._rpc({"balance":1,"account":"current"})
        self.balance=float(r.get("balance",{}).get("balance",self.balance))

    async def disconnect(self):
        self._connected=False
        if self._ws:
            try: await self._ws.close()
            except Exception: pass

    @property
    def connected(self): return self._connected

    def _next(self):
        self._rid+=1; return self._rid

    async def _rpc(self, payload) -> dict:
        rid=self._next(); payload["req_id"]=rid
        loop=asyncio.get_running_loop(); fut=loop.create_future()
        self._pending[rid]=fut; await self._send(payload)
        try:
            return await asyncio.wait_for(fut,timeout=20.0)
        except asyncio.TimeoutError:
            self._pending.pop(rid,None)
            log.warning(f"RPC timeout req_id={rid}")
            return {"error":{"message":"timeout"}}

    async def _send(self, payload):
        await self._ws.send(json.dumps(payload))

    async def _listen(self):
        try:
            async for raw in self._ws:
                try: msg=json.loads(raw)
                except json.JSONDecodeError: continue
                if msg.get("msg_type")=="tick":
                    t=msg.get("tick",{}); sym=t.get("symbol","")
                    q=float(t.get("quote",0))
                    if q>0 and sym in self._tick_cbs:
                        asyncio.get_running_loop().create_task(
                            self._dispatch_tick(sym,q))
                    continue
                rid=msg.get("req_id")
                if rid and rid in self._pending:
                    fut=self._pending.pop(rid)
                    if not fut.done(): fut.set_result(msg)
                    continue
                if "error" in msg:
                    log.warning(f"WS: {msg['error'].get('message','?')}")
        except Exception as e:
            log.error(f"WS listener: {e}")
        finally:
            self._connected=False; log.warning("WS listener exited")

    async def _dispatch_tick(self, symbol, price):
        try:
            cb=self._tick_cbs.get(symbol)
            if cb:
                if asyncio.iscoroutinefunction(cb): await cb(price)
                else: cb(price)
        except Exception as e:
            log.error(f"[{symbol}] tick dispatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BOT ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
class Bot:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self.client=DerivClient(cfg)
        self.history=History(cfg.history_file); self._alive=True
        self.engines={sym: SymbolEngine(sym,cfg) for sym in cfg.symbols}

    async def run(self):
        for eng in self.engines.values(): eng.load_state()
        retry=5
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
                log.info(f"Subscribed: {self.cfg.symbols} — live adaptation active")
                retry=5
                while self._alive and self.client.connected:
                    await asyncio.sleep(1)
                if self._alive:
                    log.warning("Disconnected — reconnecting in 5s...")
                    await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Bot error: {e} — retry in {retry}s")
                await asyncio.sleep(retry); retry=min(retry*2,60)
        await self.client.disconnect()

    def _make_cb(self, symbol):
        async def _on_tick(price):
            await self._process_tick(symbol, price)
        return _on_tick

    async def _process_tick(self, symbol, price):
        eng=self.engines[symbol]; fusion=eng.on_tick(price)
        if fusion is None: return
        ok,reason=eng.risk.can_trade(self.client.balance)
        if not ok:
            eng._skip_counts[reason.split(":")[0]]+=1; return
        await self._execute(symbol,eng,fusion)

    async def _execute(self, symbol, eng: SymbolEngine, fusion: FusionResult):
        stake=max(min(eng.risk.current_stake,
                      round(self.client.balance*self.cfg.max_balance_pct,2)),
                  self.cfg.base_stake)
        log.info(
            f"[{symbol}] TRADE {fusion.side.upper()} ${stake:.2f} "
            f"step={eng.risk._martingale_step}/{self.cfg.martingale_steps} "
            f"conf={fusion.final_confidence:.3f} "
            f"p_e={fusion.p_even:.3f} p_o={fusion.p_odd:.3f} "
            f"primary={fusion.snap_primary} "
            f"min_even={fusion.snap_min_even:.3f} null_p={fusion.snap_null_p:.2f} "
            f"regime={fusion.regime_name} bal=${self.client.balance:.2f}"
        )
        rl_s=eng.get_rl_state(fusion); eng.risk.on_open()
        result=await self.client.buy(symbol,fusion.side,stake)
        if not result:
            eng.risk.release_lock(); return
        cid=result.get("contract_id"); buy_price=float(result.get("buy_price",stake))
        self.history.add({
            "ts":datetime.utcnow().isoformat(),"symbol":symbol,
            "tick":eng._tick,"contract_id":cid,"side":fusion.side,
            "stake":buy_price,"final_confidence":fusion.final_confidence,
            "p_even":fusion.p_even,"p_odd":fusion.p_odd,
            "regime":fusion.regime_name,"entropy_score":fusion.entropy_score,
            "snap_primary":fusion.snap_primary,
            "snap_min_even":fusion.snap_min_even,
            "snap_null_p":fusion.snap_null_p,
        })
        asyncio.get_running_loop().create_task(
            self._settle(symbol,eng,cid,buy_price,fusion,rl_s,stake))

    async def _settle(self, symbol, eng: SymbolEngine,
                      cid, buy_price, fusion, rl_s, stake):
        await asyncio.sleep(3)
        won=profit=None; source="unknown"
        for _ in range(8):
            s=await self.client.contract_status(cid)
            if s:
                sold=(s.get("is_sold",False) or
                      s.get("status","") in ("sold","won","lost"))
                if sold:
                    ap=s.get("profit"); sp=s.get("sell_price")
                    profit=(float(ap) if ap is not None else
                            float(sp)-buy_price if sp else 0.0)
                    won=profit>0; source="proposal_open_contract"; break
            await asyncio.sleep(3)
        if won is None:
            txn=await self.client.profit_table_lookup(cid)
            if txn:
                profit=float(txn.get("profit",0)); won=profit>0; source="profit_table"
            else:
                log.warning(f"[{symbol}] Unconfirmed cid={cid}")
                await self.client.refresh_balance(); eng.risk.release_lock(); return
        await self.client.refresh_balance()
        eng.risk.on_close(won,profit)
        eng.after_trade(fusion,rl_s,won,profit,stake)
        self.history.update_last(cid,won,profit,self.client.balance,source)
        st=self.history.stats; bs=self.history.stats_by_symbol()
        log.info(
            f"[{symbol}] {'WIN' if won else 'LOSS'} "
            f"side={fusion.side} profit={profit:+.4f} bal=${self.client.balance:.2f} | "
            f"ALL WR={st['win_rate']:.1%} n={st['n']} PnL={st['pnl']:+.4f} | " +
            " ".join(f"{s} WR={bs.get(s,{}).get('win_rate',0):.1%}"
                     f"({bs.get(s,{}).get('n',0)})" for s in SYMBOLS)
        )

    def shutdown(self):
        self._alive=False
        st=self.history.stats
        log.info(f"Shutdown | WR={st['win_rate']:.1%} n={st['n']} PnL={st['pnl']:+.4f}")
        for sym,eng in self.engines.items():
            log.info(f"  [{sym}] ticks={eng._tick} | profile: {eng.profile.summary()}")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────
def _start_health_server(bot: Bot):
    import http.server as _hs
    port=int(os.getenv("PORT","8080"))

    class _H(_hs.BaseHTTPRequestHandler):
        def do_GET(self):
            st=bot.history.stats; bs=bot.history.stats_by_symbol()

            if self.path=="/status":
                body=json.dumps({
                    "status":"running","symbols":bot.cfg.symbols,
                    "trades":st["n"],"win_rate":round(st["win_rate"],4),
                    "pnl":st["pnl"],"balance":bot.client.balance,
                    "engines":{
                        sym:{
                            "ticks":eng._tick,
                            "profile":{"even_rate":round(eng.profile.live_even_rate,4),
                                       "markov_stab":round(eng.profile.live_markov_stab,4),
                                       "E_to_E":round(eng.profile.live_E_to_E,4),
                                       "O_to_E":round(eng.profile.live_O_to_E,4),
                                       "primary":eng.profile.primary_side,
                                       "ranking_score":round(eng.profile.live_ranking_score,6)},
                            "thresholds":{"min_even":eng.profile.min_even_prob,
                                          "min_odd":eng.profile.min_odd_prob,
                                          "markov_margin":eng.profile.markov_even_margin,
                                          "null_p":eng.profile.live_null_p_value,
                                          "zscore_gate":eng.profile.min_zscore_spike_rate},
                            "trades":bs.get(sym,{}).get("n",0),
                            "win_rate":round(bs.get(sym,{}).get("win_rate",0),4),
                            "pnl":bs.get(sym,{}).get("pnl",0),
                            "epsilon":round(eng.rl.epsilon,4),
                            "martingale":eng.risk._martingale_step,
                            "paused":eng.risk._paused,
                        }
                        for sym,eng in bot.engines.items()
                    }
                },indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.end_headers(); self.wfile.write(body); return

            rows=""
            for sym,eng in bot.engines.items():
                p=eng.profile; ss=bs.get(sym,{})
                pnl=ss.get("pnl",0)
                rows+=(f"<tr><td><strong>{sym}</strong></td>"
                       f"<td class='{'pos' if p.live_even_rate>=0.52 else 'neu'}'>"
                       f"{p.live_even_rate:.3f}</td>"
                       f"<td>{p.live_E_to_E:.3f} / {p.live_O_to_E:.3f}</td>"
                       f"<td>{p.live_markov_stab:.3f}</td>"
                       f"<td><strong>{p.primary_side}</strong></td>"
                       f"<td>{p.min_even_prob:.3f} / {p.min_odd_prob:.3f}</td>"
                       f"<td>{p.markov_even_margin:.3f}</td>"
                       f"<td>{p.live_null_p_value:.2f}</td>"
                       f"<td>{eng._tick}</td>"
                       f"<td class='{'pos' if ss.get('win_rate',0)>=0.513 else 'neg'}'>"
                       f"{ss.get('win_rate',0):.1%}({ss.get('n',0)})</td>"
                       f"<td class='{'pos' if pnl>=0 else 'neg'}'>${pnl:+.4f}</td>"
                       f"<td class='{'neg' if eng.risk._paused else 'pos'}'>"
                       f"{'PAUSED' if eng.risk._paused else 'OK'}</td></tr>")

            html=f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>Adaptive Digits Bot</title>
<style>
body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:2rem;}}
h1{{color:#58a6ff;}} h3{{color:#58a6ff;margin-top:1.5rem;}}
.sub{{color:#8b949e;margin-bottom:1.5rem;font-size:0.85rem;}}
table{{border-collapse:collapse;width:100%;margin-bottom:1rem;}}
td,th{{padding:0.35rem 0.7rem;border:1px solid #21262d;text-align:left;font-size:0.85rem;}}
th{{background:#161b22;color:#8b949e;font-weight:normal;}}
.pos{{color:#3fb950;}} .neg{{color:#f85149;}} .neu{{color:#d29922;}}
</style></head><body>
<h1>Adaptive Digits Bot</h1>
<div class="sub">R_10 + R_25 + R_100 | DIGITEVEN / DIGITODD | 1-tick |
Self-calibrating every {bot.cfg.adapt_every} ticks | Refreshes 10s</div>
<h3>Live Adaptive State (updates every {bot.cfg.adapt_every} ticks/symbol)</h3>
<table>
<tr><th>Symbol</th><th>Even rate</th><th>E->E / O->E</th><th>Markov stab</th>
<th>Primary</th><th>min_even / min_odd</th><th>Markov margin</th>
<th>null_p</th><th>Ticks</th><th>Win rate</th><th>P&L</th><th>Status</th></tr>
{rows}
</table>
<h3>Combined</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total trades</td><td><strong>{st['n']}</strong></td></tr>
<tr><td>Win rate</td>
<td class="{'pos' if st['win_rate']>=0.513 else 'neg'}">
<strong>{st['win_rate']:.1%}</strong>
<span style="color:#8b949e"> (breakeven ~51.3%)</span></td></tr>
<tr><td>P&L</td>
<td class="{'pos' if st['pnl']>=0 else 'neg'}"><strong>${st['pnl']:+.4f}</strong></td></tr>
<tr><td>Balance</td><td><strong>${bot.client.balance:.2f}</strong></td></tr>
<tr><td>Even WR</td>
<td class="{'pos' if st['even_wr']>=0.513 else 'neg'}">
{st['even_wr']:.1%} ({st['even_n']} trades)</td></tr>
<tr><td>Odd WR</td>
<td class="{'pos' if st['odd_wr']>=0.513 else 'neg'}">
{st['odd_wr']:.1%} ({st['odd_n']} trades)</td></tr>
</table>
<p style="color:#8b949e;font-size:0.8rem">
JSON: <a href="/status" style="color:#58a6ff">/status</a></p>
</body></html>"""
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers(); self.wfile.write(html.encode())

        def log_message(self,*a): pass

    threading.Thread(
        target=_hs.HTTPServer(("",port),_H).serve_forever,
        daemon=True).start()
    log.info(f"Health server :{port}  / = dashboard  /status = JSON")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
async def live(cfg: Config):
    if not cfg.api_token:
        log.error("Set DERIV_API_TOKEN environment variable"); sys.exit(1)

    bot=Bot(cfg)
    def _sig(s,f):
        log.info("Shutdown..."); bot.shutdown(); sys.exit(0)
    signal.signal(signal.SIGINT,_sig); signal.signal(signal.SIGTERM,_sig)

    log.info("="*72)
    log.info("Adaptive Digits Bot — R_10 + R_25 + R_100")
    log.info("Self-calibrating: thresholds update every "
             f"{cfg.adapt_every} ticks per symbol")
    log.info("Adapts: even_rate, Markov E->E/O->E, primary side,")
    log.info("        min_even_prob, min_odd_prob, markov_margin, null_p_value")
    log.info(f"Stake: " + " -> ".join(
        f"${cfg.base_stake*cfg.martingale_factor**s:.2f}"
        for s in range(cfg.martingale_steps+1)) + " -> halt")
    log.info("Bootstrap priors from 255min / 191 739 real ticks")
    log.info("="*72)

    _start_health_server(bot)
    await bot.run()


if __name__=="__main__":
    asyncio.run(live(Config()))
