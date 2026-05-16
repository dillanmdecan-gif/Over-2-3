# -*- coding: utf-8 -*-
"""
Adaptive DIGITOVER2 Bot — 1HZ100V + JD25 + 1HZ25V
═══════════════════════════════════════════════════
Contract : DIGITOVER 2  (wins if last digit ∈ {3,4,5,6,7,8,9})
Expiry   : 1 tick
Payout   : ~95%  →  breakeven win-rate = 51.28%

SYMBOL SELECTION — from digits_summary.json (2026-05-15, 191,739 ticks)
─────────────────────────────────────────────────────────────────────────
Ranking metric: over2_rate × markov_stability
(highest of any over2-positive symbol = most exploitable + most structured)

  Symbol     over2_rate  markov_stab  combo    ticks   d0     d1     d2
  1HZ100V    0.7160      0.09633      0.0690   14,799  0.108  0.056  0.120
  JD25       0.7600      0.06830      0.0519   16,146  0.076  0.104  0.060
  1HZ25V     0.7400      0.08186      0.0606   14,801  0.084  0.088  0.088

WHY NOT R_10 / R_25 / R_100?
  R_100  over2=0.676  (edge vs 70%: -2.4%)   ← digits 1,2 each at 12%
  R_10   over2=0.676  (edge vs 70%: -2.4%)   ← digit 1 at 12.4%
  R_25   over2=0.656  (edge vs 70%: -4.4%)   ← digits 0,1 each at 12.8%
  These three have MORE digits 0/1/2 than expected — wrong direction.

THE EDGE
─────────
  1HZ100V: digit 1 appears at only 5.6% (vs 10% expected) — biggest structural
            underweight of any losing digit across all 15 symbols.
  JD25   : digit 2 at 6.0%, digit 0 at 7.6% — both heavily suppressed.
  1HZ25V : all three losing digits uniformly low (8.4%, 8.8%, 8.8%).

ADAPTIVE MECHANISM (inbuilt live collector)
────────────────────────────────────────────
Every tick feeds a live SymbolProfile that tracks:
  • live_over2_rate      rolling window (adapt_window_long = 400 ticks)
  • live_loss_rate       = live d0+d1+d2 rate
  • live_d0/d1/d2        individual digit rolling frequencies
  • live_markov_stab     rolling Markov stability
  • live_zscore_rate     velocity spike rate

Every adapt_every=300 ticks the profile recomputes:
  • min_over2_prob  ← live_over2_rate − 0.04  (floor 0.65)
  • loss_gate       ← live_loss_rate + 0.02   (ceiling 0.32)
  • null_p_value    ← 0.35 if chi2 structure clear, else 0.50
  • markov_margin   ← max(0.02, live_markov_stab × 0.25)

Logs [ADAPT] lines showing every threshold change and why.

Run:
    export DERIV_API_TOKEN=your_token
    python over2_adaptive_bot.py

Dashboard:  http://localhost:8080/
JSON API:   http://localhost:8080/status
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("over2bot")

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOLS & BOOTSTRAP DATA  (from 191,739 real ticks, 2026-05-15)
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = ["1HZ100V", "JD25", "1HZ25V"]

BOOTSTRAP = {
    "1HZ100V": {
        "over2_rate":       0.7160,   # observed
        "loss_rate":        0.2840,   # d0+d1+d2
        "d0": 0.108, "d1": 0.056, "d2": 0.120,
        "markov_stability": 0.09633,
        "zscore_spike_rate":0.09189,
        "streak_reversion": 0.27984,
        # digit_freq[0..9]
        "digit_freq": [0.108,0.056,0.120,0.116,0.104,0.120,0.132,0.084,0.084,0.076],
        # Markov on parity (needed by MicrostructureAnalyzer seed)
        "E_to_E": 0.5469, "O_to_E": 0.4506,
    },
    "JD25": {
        "over2_rate":       0.7600,
        "loss_rate":        0.2400,
        "d0": 0.076, "d1": 0.104, "d2": 0.060,
        "markov_stability": 0.06830,
        "zscore_spike_rate":0.08738,
        "streak_reversion": 0.26220,
        "digit_freq": [0.076,0.104,0.060,0.140,0.124,0.096,0.068,0.136,0.124,0.072],
        "E_to_E": 0.5350, "O_to_E": 0.4667,
    },
    "1HZ25V": {
        "over2_rate":       0.7400,
        "loss_rate":        0.2600,
        "d0": 0.084, "d1": 0.088, "d2": 0.088,
        "markov_stability": 0.08186,
        "zscore_spike_rate":0.08891,
        "streak_reversion": 0.27200,
        "digit_freq": [0.084,0.088,0.088,0.088,0.116,0.124,0.092,0.104,0.116,0.100],
        "E_to_E": 0.5425, "O_to_E": 0.4607,
    },
}

REGIME_TRENDING   = 0
REGIME_REVERTING  = 1
REGIME_CHAOTIC    = 2
REGIME_VOL_EXPAND = 3
REGIME_STABLE     = 4
REGIME_NAMES      = ["trending","reverting","chaotic","vol_expand","stable"]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN",""))
    app_id:    str = field(default_factory=lambda: os.getenv("DERIV_APP_ID","1089"))
    api_url:   str = "wss://ws.binaryws.com/websockets/v3"
    symbols:   List[str] = field(default_factory=lambda: SYMBOLS)

    # Contract — DIGITOVER 2
    duration:        int   = 1
    duration_unit:   str   = "t"
    currency:        str   = "USD"
    payout_ratio:    float = 0.95    # 95% payout → breakeven 51.28%
    barrier:         str   = "2"     # DIGITOVER barrier

    # Warmup
    warmup_ticks: int = 150   # 1HZ ticks come fast, 150 is ~2.5 min

    # Adaptation
    adapt_every:        int = 300
    adapt_window_long:  int = 400   # rolling window for over2_rate, Markov
    adapt_window_short: int = 120   # rolling window for zscore, streak

    # Layer windows
    micro_window:       int = 30
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
    cal_window:         int   = 60
    cal_recal_every:    int   = 30

    # RL
    rl_states:        int   = 64
    rl_alpha:         float = 0.15
    rl_gamma:         float = 0.90
    rl_epsilon_start: float = 0.18  # lower — symbols are well characterised
    rl_epsilon_min:   float = 0.04
    rl_epsilon_decay: float = 0.997

    # Fusion weights — entropy + transition weighted higher for digit contracts
    w_entropy:    float = 0.32
    w_rl:         float = 0.15
    w_neural:     float = 0.15
    w_regime:     float = 0.12
    w_transition: float = 0.16   # digit-frequency Markov signal
    w_volatility: float = 0.10

    # Fixed floors — adaptation can only tighten from these
    min_final_conf:       float = 0.20
    min_regime_stability: float = 0.38
    entropy_threshold:    float = 0.91  # 1HZ symbols run slightly higher entropy

    # Stake / martingale
    base_stake:        float = 0.35
    martingale_factor: float = 1.50
    martingale_steps:  int   = 2     # $0.35 → $0.53 → $0.79 → halt
    max_balance_pct:   float = 0.10

    # Risk
    loss_cooldown_ticks:    int   = 4
    max_consecutive_losses: int   = 2
    max_daily_loss_pct:     float = 0.15
    balance_guard_mult:     int   = 0

    # Persistence
    state_dir:    str = "."
    history_file: str = "over2_trades.csv"

    # Logging
    skip_log_interval:  float = 30.0
    skip_summary_every: int   = 200


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SYMBOL PROFILE  — inbuilt collector + threshold adapter
# Specific to DIGITOVER2: tracks over2_rate and individual d0/d1/d2 rates.
# ─────────────────────────────────────────────────────────────────────────────
class SymbolProfile:
    """
    Rolling live stats for one symbol, specific to DIGITOVER2 strategy.

    What it tracks (updated every tick):
      • digit frequency counts [0..9] in a rolling window
      • live_over2_rate  = fraction of ticks with digit ∈ {3..9}
      • live_d0/d1/d2    = live frequencies of the three losing digits
      • live_loss_rate   = d0+d1+d2
      • live_markov_stab (parity-based, same as Even/Odd engine)
      • live_zscore_rate (velocity spike frequency)
      • live_streak_rev  (over2-streak reversion rate)

    What it computes every adapt_every ticks:
      • min_over2_prob  ← live_over2_rate − 0.04  (floor 0.65)
      • loss_gate       ← live_loss_rate  + 0.02  (max allowed d0+d1+d2)
      • null_p_value    ← 0.35 if chi2 p<0.25 (clear non-uniformity) else 0.50
      • markov_margin   ← max(0.02, live_markov_stab × 0.25)
      • zscore_gate     ← max(0.055, live_zscore_rate × 0.80)
    """

    def __init__(self, symbol: str, cfg: Config):
        self.symbol = symbol
        self.cfg    = cfg
        boot        = BOOTSTRAP[symbol]

        # Rolling digit counts for over2_rate (long window)
        self._digits_long:  deque = deque(maxlen=cfg.adapt_window_long)
        self._digits_short: deque = deque(maxlen=cfg.adapt_window_short)
        self._prices:       deque = deque(maxlen=cfg.adapt_window_long)
        self._vel_hist:     deque = deque(maxlen=cfg.adapt_window_short)

        # Parity Markov (same structure as even/odd engine)
        ee, oe = boot["E_to_E"], boot["O_to_E"]
        self._mk = [[ee*20, (1-ee)*20], [oe*20, (1-oe)*20]]
        self._mk_prev: Optional[int] = None

        # Over2-streak tracking
        self._streak         = 0
        self._streak_prev:   Optional[int] = None   # 1=over2, 0=loss
        self._streak_revs    = 0
        self._streak_total   = 0

        # Velocity zscore
        self._zscore_spikes  = 0
        self._zscore_checks  = 0

        # Digit frequency counts for the long window (Laplace-smoothed)
        self._digit_counts = {d: boot["digit_freq"][d]*cfg.adapt_window_long
                               for d in range(10)}

        # Bootstrap live stats
        self.live_over2_rate   = boot["over2_rate"]
        self.live_loss_rate    = boot["loss_rate"]
        self.live_d0           = boot["d0"]
        self.live_d1           = boot["d1"]
        self.live_d2           = boot["d2"]
        self.live_markov_stab  = boot["markov_stability"]
        self.live_zscore_rate  = boot["zscore_spike_rate"]
        self.live_streak_rev   = boot["streak_reversion"]
        self.live_chi2_p       = 1.0

        # Adaptive thresholds — computed from bootstrap first
        self._compute_thresholds()

        self._ticks = 0

    # ── Per-tick update ───────────────────────────────────────────────────────
    def push(self, price: float):
        digit   = int(round(price * 100)) % 10
        parity  = digit % 2
        is_over2 = 1 if digit >= 3 else 0

        self._prices.append(price)
        self._digits_long.append(digit)
        self._digits_short.append(digit)

        # Update rolling digit counts
        self._digit_counts[digit] = self._digit_counts.get(digit, 0) + 1

        # Parity Markov
        if self._mk_prev is not None:
            self._mk[self._mk_prev][parity] += 1.0
        self._mk_prev = parity

        # Over2-streak
        if self._streak_prev is None:
            self._streak = 1
        elif is_over2 == self._streak_prev:
            self._streak += 1
        else:
            if self._streak >= 3:
                self._streak_revs += 1
            self._streak_total += 1
            self._streak = 1
        self._streak_prev = is_over2

        # Velocity zscore
        prices = list(self._prices)
        if len(prices) >= 2:
            diff = abs(prices[-1] - prices[-2])
            self._vel_hist.append(diff)
            if len(self._vel_hist) >= 10:
                vel = list(self._vel_hist)
                mu = np.mean(vel); sig = np.std(vel) + 1e-12
                self._zscore_checks += 1
                if abs((vel[-1] - mu) / sig) > 1.5:
                    self._zscore_spikes += 1

        self._ticks += 1
        if self._ticks % self.cfg.adapt_every == 0:
            self._adapt()

    # ── Adaptation cycle ──────────────────────────────────────────────────────
    def _adapt(self):
        prev = {
            "over2":        self.live_over2_rate,
            "loss":         self.live_loss_rate,
            "markov":       self.live_markov_stab,
            "min_over2":    self.min_over2_prob,
            "loss_gate":    self.loss_gate,
            "null_p":       self.null_p_value,
            "markov_margin":self.markov_margin,
        }

        dlong = list(self._digits_long)
        if len(dlong) >= 50:
            n = len(dlong)
            self.live_over2_rate = sum(1 for d in dlong if d >= 3) / n
            self.live_d0 = sum(1 for d in dlong if d == 0) / n
            self.live_d1 = sum(1 for d in dlong if d == 1) / n
            self.live_d2 = sum(1 for d in dlong if d == 2) / n
            self.live_loss_rate = self.live_d0 + self.live_d1 + self.live_d2

            # Markov stability from rolling matrix
            ee_row = self._mk[0]; oe_row = self._mk[1]
            te = ee_row[0]+ee_row[1]; to = oe_row[0]+oe_row[1]
            live_ee = float(ee_row[0]/te) if te else 0.5
            live_oe = float(oe_row[0]/to) if to else 0.5
            self.live_markov_stab = abs(live_ee-0.5) + abs(live_oe-0.5)

            # Chi2 uniformity test on digit distribution
            counts = np.bincount(dlong, minlength=10).astype(float)
            _, self.live_chi2_p = scipy_stats.chisquare(
                counts, np.full(10, n/10.0))

        if self._zscore_checks > 0:
            self.live_zscore_rate = self._zscore_spikes / self._zscore_checks
        if self._streak_total > 0:
            self.live_streak_rev = self._streak_revs / self._streak_total

        self._compute_thresholds()

        # Log changes
        changes = []
        if abs(self.live_over2_rate - prev["over2"]) > 0.003:
            changes.append(f"over2 {prev['over2']:.4f}->{self.live_over2_rate:.4f}")
        if abs(self.live_loss_rate - prev["loss"]) > 0.003:
            changes.append(f"loss {prev['loss']:.4f}->{self.live_loss_rate:.4f}")
        if abs(self.live_markov_stab - prev["markov"]) > 0.005:
            changes.append(f"markov {prev['markov']:.4f}->{self.live_markov_stab:.4f}")
        if abs(self.min_over2_prob - prev["min_over2"]) > 0.003:
            changes.append(f"min_over2 {prev['min_over2']:.4f}->{self.min_over2_prob:.4f}")
        if abs(self.loss_gate - prev["loss_gate"]) > 0.003:
            changes.append(f"loss_gate {prev['loss_gate']:.4f}->{self.loss_gate:.4f}")
        if self.null_p_value != prev["null_p"]:
            changes.append(f"null_p {prev['null_p']:.2f}->{self.null_p_value:.2f}")

        if changes:
            log.info(f"[{self.symbol} ADAPT t={self._ticks}] " + " | ".join(changes))
        else:
            log.info(f"[{self.symbol} ADAPT t={self._ticks}] stable | "
                     f"over2={self.live_over2_rate:.4f} "
                     f"loss={self.live_loss_rate:.4f}(d0={self.live_d0:.3f} "
                     f"d1={self.live_d1:.3f} d2={self.live_d2:.3f}) "
                     f"min_over2={self.min_over2_prob:.4f} "
                     f"loss_gate={self.loss_gate:.4f} "
                     f"null_p={self.null_p_value:.2f}")

    def _compute_thresholds(self):
        """Derive all adaptive thresholds from current live stats."""
        # min_over2_prob: live rate minus 4% discount, floor at 0.65
        # (65% is still well above 51.28% breakeven)
        self.min_over2_prob = max(0.650, round(self.live_over2_rate - 0.04, 4))

        # loss_gate: maximum tolerated live d0+d1+d2 rate before suppressing trades.
        # If losing digits are rising above baseline, tighten.
        # Ceiling = live_loss_rate + 2% buffer
        self.loss_gate = min(0.320, round(self.live_loss_rate + 0.020, 4))

        # null_p_value: tighter gate when chi2 shows clear digit non-uniformity
        self.null_p_value = 0.35 if self.live_chi2_p < 0.25 else 0.50

        # markov_margin: how far E→E must exceed 0.5 to confirm entry
        self.markov_margin = max(0.020, round(self.live_markov_stab * 0.25, 4))

        # zscore gate
        self.zscore_gate = max(0.055, round(self.live_zscore_rate * 0.80, 4))

    def live_loss_rate_ok(self, window_digits: list) -> bool:
        """
        Returns False if current short-window d0+d1+d2 rate is running
        above the adaptive loss_gate — suppress trades during losing-digit surges.
        """
        if len(window_digits) < 20:
            return True
        short_loss = sum(1 for d in window_digits if d < 3) / len(window_digits)
        return short_loss <= self.loss_gate

    def summary(self) -> str:
        return (f"over2={self.live_over2_rate:.4f} "
                f"loss={self.live_loss_rate:.4f}(d0={self.live_d0:.3f},"
                f"d1={self.live_d1:.3f},d2={self.live_d2:.3f}) "
                f"min_over2={self.min_over2_prob:.4f} "
                f"loss_gate={self.loss_gate:.4f} "
                f"null_p={self.null_p_value:.2f} "
                f"markov={self.live_markov_stab:.4f}")

    def get_state(self) -> dict:
        return {
            "mk":               [r[:] for r in self._mk],
            "live_over2_rate":  self.live_over2_rate,
            "live_loss_rate":   self.live_loss_rate,
            "live_d0":          self.live_d0,
            "live_d1":          self.live_d1,
            "live_d2":          self.live_d2,
            "live_markov_stab": self.live_markov_stab,
            "live_zscore_rate": self.live_zscore_rate,
            "live_streak_rev":  self.live_streak_rev,
            "live_chi2_p":      self.live_chi2_p,
            "zscore_spikes":    self._zscore_spikes,
            "zscore_checks":    self._zscore_checks,
            "streak_revs":      self._streak_revs,
            "streak_total":     self._streak_total,
        }

    def load_state(self, s: dict):
        self._mk              = s.get("mk", self._mk)
        self.live_over2_rate  = s.get("live_over2_rate",  self.live_over2_rate)
        self.live_loss_rate   = s.get("live_loss_rate",   self.live_loss_rate)
        self.live_d0          = s.get("live_d0",          self.live_d0)
        self.live_d1          = s.get("live_d1",          self.live_d1)
        self.live_d2          = s.get("live_d2",          self.live_d2)
        self.live_markov_stab = s.get("live_markov_stab", self.live_markov_stab)
        self.live_zscore_rate = s.get("live_zscore_rate", self.live_zscore_rate)
        self.live_streak_rev  = s.get("live_streak_rev",  self.live_streak_rev)
        self.live_chi2_p      = s.get("live_chi2_p",      self.live_chi2_p)
        self._zscore_spikes   = s.get("zscore_spikes", 0)
        self._zscore_checks   = s.get("zscore_checks", 0)
        self._streak_revs     = s.get("streak_revs",   0)
        self._streak_total    = s.get("streak_total",  0)
        self._compute_thresholds()


# ─────────────────────────────────────────────────────────────────────────────
# MICROSTRUCTURE ANALYZER  (digit + velocity based)
# ─────────────────────────────────────────────────────────────────────────────
class MicrostructureAnalyzer:

    def __init__(self, cfg: Config, profile: SymbolProfile):
        self.cfg     = cfg
        self.profile = profile
        self._prices  = deque(maxlen=max(cfg.micro_window, cfg.cluster_window, 100))
        self._digits  = deque(maxlen=80)
        self._diffs   = deque(maxlen=cfg.micro_window)
        self._vel_hist= deque(maxlen=cfg.micro_window)
        # Digit-3-split Markov: does being >2 predict next is >2?
        # [prev_over2][curr_over2]  — Laplace prior
        o2 = profile.live_over2_rate
        self._over2_mk = [[o2*10, (1-o2)*10],    # prev=over2
                          [(1-o2)*10, o2*10]]     # prev=under3  (note: symmetric assumption)
        self._over2_prev: Optional[int] = None
        # Parity Markov (for fusion bias signal)
        ee, oe = BOOTSTRAP[profile.symbol]["E_to_E"], BOOTSTRAP[profile.symbol]["O_to_E"]
        self._mk = [[ee*10,(1-ee)*10],[oe*10,(1-oe)*10]]
        self._par_prev: Optional[int] = None

    def push(self, price: float) -> dict:
        digit    = int(round(price * 100)) % 10
        parity   = digit % 2
        is_over2 = 1 if digit >= 3 else 0

        self._prices.append(price)
        self._digits.append(digit)

        prices = list(self._prices)
        f = {}
        if len(prices) >= 2:
            diff = abs(prices[-1] - prices[-2])
            self._diffs.append(diff); self._vel_hist.append(diff)

        if len(self._vel_hist) >= 10:
            vel  = list(self._vel_hist)
            mu   = np.mean(vel); sig = np.std(vel) + 1e-12
            vz   = float((vel[-1]-mu)/sig)
            f["tick_velocity_z"] = vz
            diffs = list(self._diffs); mid = len(diffs)//2
            f["tick_acceleration_z"] = float(
                (np.mean(diffs[mid:])-np.mean(diffs[:mid]) if mid else 0) / (sig+1e-12))
        else:
            f["tick_velocity_z"] = f["tick_acceleration_z"] = 0.0

        f["volatility_burst"] = f["tick_velocity_z"]

        # Over2 Markov bias — P(next is over2 | current is over2) - base_rate
        if self._over2_prev is not None:
            self._over2_mk[self._over2_prev][is_over2] += 1.0
        self._over2_prev = is_over2

        row_o2 = self._over2_mk[1 if is_over2 else 0]
        tot    = row_o2[0] + row_o2[1]
        live_p_next_over2 = float(row_o2[1]/tot) if tot else self.profile.live_over2_rate
        f["over2_markov_p"]    = live_p_next_over2
        f["over2_markov_bias"] = live_p_next_over2 - self.profile.live_over2_rate

        # Parity Markov (secondary signal)
        if self._par_prev is not None:
            self._mk[self._par_prev][parity] += 1.0
        self._par_prev = parity
        row_p = self._mk[self._par_prev if self._par_prev is not None else 0]
        tot_p = row_p[0]+row_p[1]
        f["markov_even_bias"] = float((row_p[0]/tot_p)-0.5) if tot_p else 0.0
        f["markov_odd_bias"]  = float((row_p[1]/tot_p)-0.5) if tot_p else 0.0

        # Short-window loss rate gate
        short = list(self._digits)[-30:] if len(self._digits) >= 30 else list(self._digits)
        f["short_loss_rate"]  = sum(1 for d in short if d < 3) / max(len(short),1)
        f["loss_gate_ok"]     = f["short_loss_rate"] <= self.profile.loss_gate

        # Cluster density
        if len(self._digits) >= cfg.cluster_window if (cfg:=self.cfg) else False:
            counts = np.bincount(list(self._digits)[-self.cfg.cluster_window:],
                                 minlength=10).astype(float)
            exp    = self.cfg.cluster_window / 10.0
            f["cluster_density"] = float(np.clip(
                np.sum((counts-exp)**2/(exp+1e-9)) / (self.cfg.cluster_window*9), 0, 1))
        else:
            f["cluster_density"] = 0.0

        f["over2_rate_short"] = sum(1 for d in short if d >= 3) / max(len(short),1)
        return f

    def get_state(self) -> dict:
        return {"over2_mk": [r[:] for r in self._over2_mk],
                "mk": [r[:] for r in self._mk]}

    def load_state(self, s: dict):
        self._over2_mk = s.get("over2_mk", self._over2_mk)
        self._mk       = s.get("mk",       self._mk)


# ─────────────────────────────────────────────────────────────────────────────
# ENTROPY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EntropyEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._digits = deque(maxlen=max(cfg.entropy_window, cfg.null_hyp_window, 150))

    def push(self, digit: int) -> dict:
        self._digits.append(digit)
        out = {"shannon":1.0,"permutation":1.0,"uniformity":1.0,
               "composite":1.0,"tradeable":False}
        if len(self._digits) < self.cfg.entropy_window: return out
        w      = list(self._digits)[-self.cfg.entropy_window:]
        sh     = self._shannon(w)
        pe     = self._perm_entropy(w, self.cfg.perm_entropy_order)
        unif   = self._uniformity_p(w)
        comp   = float(np.clip(0.45*sh + 0.35*pe + 0.20*(1.0-unif), 0, 1))
        out.update({"shannon":round(sh,4),"permutation":round(pe,4),
                    "uniformity":round(unif,4),"composite":round(comp,4),
                    "tradeable": comp < self.cfg.entropy_threshold})
        return out

    @staticmethod
    def _shannon(d):
        c=np.bincount(d,minlength=10).astype(float); p=c[c>0]/c.sum()
        return float(-np.sum(p*np.log2(p))/np.log2(10))

    @staticmethod
    def _perm_entropy(d, order):
        if len(d)<order+1: return 1.0
        pats=Counter(tuple(sorted(range(order),key=lambda j:d[i:i+order][j]))
                     for i in range(len(d)-order+1))
        total=sum(pats.values()); probs=[v/total for v in pats.values()]
        h=-sum(p*math.log2(p) for p in probs if p>0)
        mh=math.log2(math.factorial(order))
        return float(h/mh) if mh>0 else 1.0

    @staticmethod
    def _uniformity_p(d):
        c=np.bincount(d,minlength=10).astype(float)
        _,p=scipy_stats.chisquare(c,np.full(10,len(d)/10.0))
        return float(p)


# ─────────────────────────────────────────────────────────────────────────────
# RL AGENT
# ─────────────────────────────────────────────────────────────────────────────
class RLAgent:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._Q=np.zeros((cfg.rl_states,2))
        self._eps=cfg.rl_epsilon_start; self._last_s=self._last_a=None

    def state_index(self, entropy, regime, win_rate, vol_z) -> int:
        e=min(int(entropy*4),3); r=min(regime,4)
        wr=min(int(win_rate*4),3); vz=min(int((np.clip(vol_z,-3,3)+3)/1.5),3)
        return int(np.clip(e*16+r*3+wr+vz,0,self.cfg.rl_states-1))

    def act(self, state) -> Tuple[int,float]:
        a=(random.choice([0,1]) if random.random()<self._eps
           else int(np.argmax(self._Q[state])))
        q=self._Q[state]; qr=max(abs(q.max()-q.min()),1e-9)
        self._last_s=state; self._last_a=a
        return a, float(np.clip((q[a]-q.min())/qr,0,1))

    def update(self, reward, next_s):
        if self._last_s is None: return
        td=(reward+self.cfg.rl_gamma*np.max(self._Q[next_s])
            -self._Q[self._last_s][self._last_a])
        self._Q[self._last_s][self._last_a]+=self.cfg.rl_alpha*td
        self._eps=max(self.cfg.rl_epsilon_min,self._eps*self.cfg.rl_epsilon_decay)

    @property
    def epsilon(self): return self._eps
    def get_state(self): return {"Q":self._Q.tolist(),"eps":self._eps}
    def load_state(self,s): self._Q=np.array(s["Q"]); self._eps=s["eps"]


# ─────────────────────────────────────────────────────────────────────────────
# DIGIT NET  (predicts P(next digit >= 3))
# ─────────────────────────────────────────────────────────────────────────────
class DigitNet:
    def __init__(self, cfg: Config):
        self.cfg=cfg; w=cfg.nn_input_window; h=cfg.nn_hidden; F=4
        rng=np.random.default_rng(42)
        self.W_conv=rng.normal(0,0.1,(h,F,3)); self.b_conv=np.zeros(h)
        pool_out=max((w-2)//2,1); gru_in=h*pool_out
        self.W_gru=rng.normal(0,0.1,(h,gru_in)); self.b_gru=np.zeros(h)
        self.W_att=rng.normal(0,0.1,(1,h));  self.b_att=np.zeros(1)
        # Output: [p_over2, noise, stability]
        self.W_out=rng.normal(0,0.1,(3,h)); self.b_out=np.zeros(3)
        self._buf_X=[]; self._buf_y=[]

    @staticmethod
    def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
    @staticmethod
    def _relu(x): return np.maximum(0,x)

    def _forward(self, X):
        T,F=X.shape; k=3; ol=T-k+1
        cnn=np.zeros((self.cfg.nn_hidden,ol))
        for t in range(ol):
            cnn[:,t]=self._relu(np.einsum('hij,ij->h',self.W_conv,X[t:t+k,:].T)+self.b_conv)
        pl=max(ol//2,1)
        pooled=np.array([cnn[:,i*2:i*2+2].max(axis=1) for i in range(pl)]).T
        flat=pooled.flatten(); gi=self.W_gru.shape[1]
        flat=flat[:gi] if len(flat)>=gi else np.pad(flat,(0,gi-len(flat)))
        hg=self._relu(self.W_gru@flat+self.b_gru)
        ha=hg*self._sig(self.W_att@hg+self.b_att)[0]
        return self._sig(self.W_out@ha+self.b_out),ha

    def predict(self, X) -> dict:
        out,_=self._forward(X)
        return {"p_over2":float(out[0]),"noise":float(out[1]),"stability":float(out[2])}

    def record(self, X, y):
        self._buf_X.append(X.copy()); self._buf_y.append(y.copy())
        if len(self._buf_X)>self.cfg.nn_batch*4: self._buf_X.pop(0); self._buf_y.pop(0)
        if len(self._buf_X)>=self.cfg.nn_batch: self._train_step()

    def _train_step(self):
        idxs=random.sample(range(len(self._buf_X)),min(self.cfg.nn_batch,len(self._buf_X)))
        gW=np.zeros_like(self.W_out); gb=np.zeros_like(self.b_out)
        for i in idxs:
            out,h=self._forward(self._buf_X[i]); err=out-self._buf_y[i]
            gW+=np.outer(err,h); gb+=err
        n=len(idxs); self.W_out-=self.cfg.nn_lr*gW/n; self.b_out-=self.cfg.nn_lr*gb/n

    def get_state(self):
        return {k:getattr(self,k).tolist()
                for k in("W_conv","b_conv","W_gru","b_gru","W_att","b_att","W_out","b_out")}
    def load_state(self,s):
        for k,v in s.items(): setattr(self,k,np.array(v))


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class RegimeDetector:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._prices=deque(maxlen=cfg.regime_window)
        self._vol_h=deque(maxlen=cfg.regime_window)

    def push(self, price, entropy_score) -> Tuple[int,float]:
        self._prices.append(price); prices=list(self._prices)
        if len(prices)<15: return REGIME_STABLE,0.50
        diffs=np.diff(prices); vol=float(np.std(diffs)); self._vol_h.append(vol)
        ac=(float(np.corrcoef(diffs[:-1],diffs[1:])[0,1]) if len(diffs)>=10 else 0.0)
        vol_z=(float((vol-np.mean(list(self._vol_h)))/(np.std(list(self._vol_h))+1e-12))
               if len(self._vol_h)>=10 else 0.0)
        if   vol_z>2.0: return REGIME_VOL_EXPAND,min(0.5+vol_z*0.1,1.0)
        elif entropy_score<0.12: return REGIME_CHAOTIC,0.70
        elif ac>0.30:   return REGIME_TRENDING,min(0.5+ac,1.0)
        elif ac<-0.30:  return REGIME_REVERTING,min(0.5+abs(ac),1.0)
        else:           return REGIME_STABLE,max(0.5,entropy_score)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATOR  (Platt scaling on p_over2)
# ─────────────────────────────────────────────────────────────────────────────
class Calibrator:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._A=1.0; self._B=0.0
        self._hist=deque(maxlen=cfg.cal_window); self._since=0

    def calibrate(self, p) -> float:
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
# NULL HYPOTHESIS TESTER
# ─────────────────────────────────────────────────────────────────────────────
class NullHypothesisTester:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._digits=deque(maxlen=cfg.null_hyp_window)

    def push(self, digit): self._digits.append(digit)

    def test(self, null_p_value: float) -> Tuple[bool,float,str]:
        digits=list(self._digits)
        if len(digits)<self.cfg.null_hyp_window: return False,1.0,"insufficient"
        counts=np.bincount(digits,minlength=10).astype(float)
        _,p_chi=scipy_stats.chisquare(counts,np.full(10,len(digits)/10.0))
        # Over2-specific runs test
        binary=[1 if d>=3 else 0 for d in digits]
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
# ADAPTIVE LEARNER
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveLearner:
    FEATURES=["entropy","over2_markov","loss_gate","vol_burst","nn_over2","regime_stability"]

    def __init__(self):
        self._weights={f:1.0 for f in self.FEATURES}
        self._scores={f:deque(maxlen=50) for f in self.FEATURES}
        self._outcomes=deque(maxlen=100); self._drift=False

    def record(self, preds, won):
        self._outcomes.append(1 if won else 0)
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
                log.warning("[DRIFT] acc<45% — penalising weights")
                for f in self._weights: self._weights[f]=max(0.1,self._weights[f]*0.85)

    def weight(self,f): return self._weights.get(f,1.0)
    @property
    def drift(self): return self._drift
    @property
    def recent_accuracy(self): return float(np.mean(self._outcomes)) if self._outcomes else 0.0
    def get_state(self): return {"weights":dict(self._weights),
                                  "scores":{k:list(v) for k,v in self._scores.items()}}
    def load_state(self,s):
        self._weights=s.get("weights",{f:1.0 for f in self.FEATURES})
        for f,v in s.get("scores",{}).items():
            if f in self._scores: self._scores[f]=deque(v,maxlen=50)


# ─────────────────────────────────────────────────────────────────────────────
# FUSION RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FusionResult:
    final_confidence: float
    entropy_score:    float
    rl_confidence:    float
    neural_p_over2:   float   # calibrated P(next digit >= 3)
    over2_markov_p:   float   # live Markov P(next is over2)
    regime_stability: float
    volatility_score: float
    regime:           int
    regime_name:      str
    null_rejected:    bool
    null_p:           float
    tradeable:        bool
    block_reason:     str
    # Threshold snapshot
    snap_min_over2:   float = 0.0
    snap_loss_gate:   float = 0.0
    snap_null_p:      float = 0.0
    snap_short_loss:  float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE FUSION
# DIGITOVER2 specific: single direction (always over2), so fusion is about
# CONFIDENCE that this is a good entry — not about picking a side.
# ─────────────────────────────────────────────────────────────────────────────
def fuse(cfg: Config,
         profile:     SymbolProfile,
         entropy:     dict,
         rl_conf:     float,
         nn_pred:     dict,
         regime_id:   int,
         regime_conf: float,
         micro:       dict,
         null_rej:    bool,
         null_p:      float,
         learner:     AdaptiveLearner,
         cal:         Calibrator) -> FusionResult:

    snap_min_over2  = profile.min_over2_prob
    snap_loss_gate  = profile.loss_gate
    snap_null_p     = profile.null_p_value
    snap_short_loss = micro.get("short_loss_rate", 0.0)

    entropy_score = float(1.0 - entropy["composite"])
    vol_z         = micro.get("volatility_burst", 0.0)
    vol_score     = float(np.clip(1.0 - abs(vol_z)/3.0, 0, 1))

    over2_markov_p   = micro.get("over2_markov_p",  profile.live_over2_rate)
    over2_markov_bias= micro.get("over2_markov_bias", 0.0)
    trans_score      = float(np.clip(over2_markov_bias / profile.live_over2_rate + 0.5, 0, 1))

    p_over2_cal = cal.calibrate(nn_pred["p_over2"])
    neural_conf = float(np.clip((p_over2_cal - 0.5) * 2.0, 0, 1))

    w_e = cfg.w_entropy    * learner.weight("entropy")
    w_r = cfg.w_rl         * learner.weight("nn_over2")
    w_n = cfg.w_neural     * learner.weight("nn_over2")
    w_g = cfg.w_regime
    w_t = cfg.w_transition * learner.weight("over2_markov")
    w_v = cfg.w_volatility * learner.weight("vol_burst")
    tot = w_e+w_r+w_n+w_g+w_t+w_v or 1.0

    conf = (w_e*entropy_score + w_r*rl_conf + w_n*neural_conf
            + w_g*regime_conf + w_t*trans_score + w_v*vol_score) / tot
    if learner.drift: conf *= 0.70
    conf = float(np.clip(conf, 0, 1))

    # ── GATES ─────────────────────────────────────────────────────────────────
    block = []

    # Entropy gate
    if not entropy["tradeable"]:
        block.append(f"entropy={entropy['composite']:.3f}>={cfg.entropy_threshold}")

    # Null hypothesis gate (uses live adaptive null_p_value)
    if not null_rej:
        block.append(f"null_p={null_p:.3f}>={snap_null_p:.2f}")

    # Regime stability gate
    if regime_conf < cfg.min_regime_stability:
        block.append(f"regime={regime_conf:.3f}<{cfg.min_regime_stability}")

    # Final confidence gate
    if conf < cfg.min_final_conf:
        block.append(f"conf={conf:.3f}<{cfg.min_final_conf}")

    # Neural P(over2) gate — calibrated probability must be above min_over2_prob
    if p_over2_cal < snap_min_over2:
        block.append(f"p_over2={p_over2_cal:.4f}<{snap_min_over2:.4f}")

    # Short-window loss rate gate — suppress if losing digits are surging
    if not micro.get("loss_gate_ok", True):
        block.append(f"loss_surge={snap_short_loss:.3f}>{snap_loss_gate:.3f}")

    # Markov over2 gate — live Markov must not predict loss is more likely
    if over2_markov_p < profile.live_over2_rate - profile.markov_margin:
        block.append(f"markov_over2={over2_markov_p:.4f}<{profile.live_over2_rate-profile.markov_margin:.4f}")

    # Drift gate
    if learner.drift:
        block.append("drift")

    return FusionResult(
        final_confidence = round(conf,4),
        entropy_score    = round(entropy_score,4),
        rl_confidence    = round(rl_conf,4),
        neural_p_over2   = round(p_over2_cal,4),
        over2_markov_p   = round(over2_markov_p,4),
        regime_stability = round(regime_conf,4),
        volatility_score = round(vol_score,4),
        regime           = regime_id,
        regime_name      = REGIME_NAMES[regime_id],
        null_rejected    = null_rej,
        null_p           = round(null_p,4),
        tradeable        = len(block)==0,
        block_reason     = " | ".join(block) if block else "ok",
        snap_min_over2   = snap_min_over2,
        snap_loss_gate   = snap_loss_gate,
        snap_null_p      = snap_null_p,
        snap_short_loss  = snap_short_loss,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._mart_step=0; self._consec_losses=0
        self._in_trade=False; self._paused=False; self._pause_reason=""
        self._start_balance=None; self._daily_pnl=0.0; self._cooldown=0

    def set_balance(self,b):
        if self._start_balance is None: self._start_balance=b

    @property
    def current_stake(self):
        return round(self.cfg.base_stake*
                     (self.cfg.martingale_factor**min(self._mart_step,self.cfg.martingale_steps)),2)

    def tick(self):
        if self._cooldown>0: self._cooldown-=1

    def can_trade(self, balance) -> Tuple[bool,str]:
        if self._in_trade:  return False,"in_trade"
        if self._paused:    return False,f"paused:{self._pause_reason}"
        if self._cooldown>0: return False,f"cooldown:{self._cooldown}t"
        if self._consec_losses>=self.cfg.max_consecutive_losses:
            self._paused=True; self._pause_reason=f"{self._consec_losses}_losses"
            return False,f"paused:{self._pause_reason}"
        if self._start_balance:
            if self._daily_pnl<-(self._start_balance*self.cfg.max_daily_loss_pct):
                self._paused=True; self._pause_reason="daily_loss_cap"
                return False,"paused:daily_loss_cap"
        if balance>0 and balance<self.cfg.base_stake*self.cfg.balance_guard_mult:
            return False,f"balance_low:{balance:.2f}"
        return True,"ok"

    def on_open(self): self._in_trade=True

    def on_close(self, won, profit):
        self._in_trade=False; self._daily_pnl+=profit
        if won:
            self._mart_step=0; self._consec_losses=0; self._cooldown=0
        else:
            self._consec_losses+=1
            self._mart_step=min(self._mart_step+1,self.cfg.martingale_steps)
            self._cooldown=self.cfg.loss_cooldown_ticks

    def release_lock(self): self._in_trade=False
    def reset(self):
        self._paused=False; self._consec_losses=0
        self._mart_step=0; self._cooldown=0


# ─────────────────────────────────────────────────────────────────────────────
# TRADE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
class History:
    COLS=["ts","symbol","tick","contract_id","stake",
          "final_confidence","neural_p_over2","over2_markov_p",
          "regime","entropy_score","snap_min_over2","snap_loss_gate",
          "snap_short_loss","snap_null_p",
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
        if not done: return {"n":0,"win_rate":0.0,"pnl":0.0}
        wins=[r for r in done if r.get("won") in (True,"True")]
        pnl=sum(float(r.get("profit",0) or 0) for r in done)
        return {"n":len(done),"win_rate":len(wins)/len(done),"pnl":round(pnl,4)}

    def stats_by_symbol(self):
        result={}
        for sym in SYMBOLS:
            rows=[r for r in self._rows
                  if r.get("symbol")==sym and r.get("won")!=""]
            if not rows: result[sym]={"n":0,"win_rate":0.0,"pnl":0.0}; continue
            wins=[r for r in rows if r.get("won") in (True,"True")]
            result[sym]={"n":len(rows),"win_rate":len(wins)/len(rows),
                         "pnl":round(sum(float(r.get("profit",0) or 0) for r in rows),4)}
        return result


# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class SymbolEngine:

    def __init__(self, symbol: str, cfg: Config):
        self.symbol  = symbol
        self.cfg     = cfg
        self.profile = SymbolProfile(symbol, cfg)
        self.micro   = MicrostructureAnalyzer(cfg, self.profile)
        self.entropy = EntropyEngine(cfg)
        self.rl      = RLAgent(cfg)
        self.nn      = DigitNet(cfg)
        self.regime  = RegimeDetector(cfg)
        self.cal     = Calibrator(cfg)
        self.null_t  = NullHypothesisTester(cfg)
        self.learner = AdaptiveLearner()
        self.risk    = RiskManager(cfg)

        self._feat_buf  = deque(maxlen=cfg.nn_input_window)
        self._recent_wr = deque(maxlen=20)
        self._tick      = 0
        self._skip      = Counter()
        self._last_skip_log  = 0.0
        self._last_state_log = 0.0
        self._ticks_after_warmup = 0

        self.last_fusion: Optional[FusionResult] = None

    def state_path(self):
        return os.path.join(self.cfg.state_dir, f"{self.symbol}_over2_state.pkl")

    def save_state(self):
        state={"version":5,"symbol":self.symbol,
               "saved_at":datetime.utcnow().isoformat(),
               "rl":self.rl.get_state(),"nn":self.nn.get_state(),
               "cal":self.cal.get_state(),"learner":self.learner.get_state(),
               "micro":self.micro.get_state(),"profile":self.profile.get_state()}
        path=self.state_path(); tmp=path+".tmp"
        with open(tmp,"wb") as f: pickle.dump(state,f,protocol=4)
        os.replace(tmp,path)

    def load_state(self):
        path=self.state_path()
        if not os.path.exists(path):
            log.info(f"[{self.symbol}] No state file — using bootstrap priors")
            return
        try:
            with open(path,"rb") as f: s=pickle.load(f)
            self.rl.load_state(s["rl"]); self.nn.load_state(s["nn"])
            self.cal.load_state(s["cal"]); self.learner.load_state(s["learner"])
            self.micro.load_state(s["micro"])
            if "profile" in s: self.profile.load_state(s["profile"])
            log.info(f"[{self.symbol}] State loaded ({s.get('saved_at','?')}) | "
                     f"{self.profile.summary()}")
        except Exception as e:
            log.warning(f"[{self.symbol}] State load failed: {e}")

    def on_tick(self, price: float) -> Optional[FusionResult]:
        self._tick+=1; self.risk.tick()
        digit    = int(round(price*100))%10
        is_over2 = 1 if digit>=3 else 0

        # Feed live collector first — may trigger adaptation
        self.profile.push(price)

        micro_f = self.micro.push(price)
        ent     = self.entropy.push(digit)
        self.null_t.push(digit)

        # Neural net feature vector: [digit/9, is_over2, velocity_z_norm, loss_rate]
        vz = micro_f.get("tick_velocity_z", 0.0)
        nn_vec = np.array([digit/9.0, float(is_over2),
                           float(np.clip(vz/3.0, -1, 1)),
                           self.profile.live_loss_rate], dtype=np.float32)
        self._feat_buf.append(nn_vec)

        if self._tick < self.cfg.warmup_ticks:
            if self._tick % 30 == 0:
                log.info(f"[{self.symbol}] Warmup {self._tick}/{self.cfg.warmup_ticks} | "
                         f"{self.profile.summary()}")
            return None
        if len(self._feat_buf) < self.cfg.nn_input_window:
            return None

        self._ticks_after_warmup += 1
        regime_id, regime_conf = self.regime.push(price, 1.0-ent["composite"])
        X       = np.stack(list(self._feat_buf), axis=0)
        nn_pred = self.nn.predict(X)
        null_rej, null_p, _ = self.null_t.test(self.profile.null_p_value)

        wr    = float(np.mean(self._recent_wr)) if self._recent_wr else 0.5
        rl_s  = self.rl.state_index(ent["composite"], regime_id, wr,
                                     micro_f.get("volatility_burst",0.0))
        rl_a, rl_c = self.rl.act(rl_s)

        fusion = fuse(self.cfg, self.profile, ent,
                      rl_c if rl_a==1 else 0.0,
                      nn_pred, regime_id, regime_conf,
                      micro_f, null_rej, null_p,
                      self.learner, self.cal)
        self.last_fusion = fusion

        # Periodic state log
        now = time.time()
        if now - self._last_state_log > 20:
            log.info(f"[{self.symbol} t={self._tick}] "
                     f"regime={fusion.regime_name}({fusion.regime_stability:.2f}) "
                     f"ent={fusion.entropy_score:.3f} conf={fusion.final_confidence:.3f} "
                     f"p_over2={fusion.neural_p_over2:.4f} "
                     f"markov_over2={fusion.over2_markov_p:.4f} "
                     f"null={'REJ' if fusion.null_rejected else 'fail'}(p={fusion.null_p:.3f}) "
                     f"short_loss={fusion.snap_short_loss:.3f} "
                     f"block=[{fusion.block_reason[:60]}]")
            self._last_state_log = now

        if self._ticks_after_warmup % self.cfg.skip_summary_every == 0:
            self._log_skip_summary()

        if rl_a==0:
            self._skip["rl_idle"]+=1; return None
        if not fusion.tradeable:
            key=fusion.block_reason.split("|")[0].strip()[:35]
            self._skip[key]+=1
            if now-self._last_skip_log>self.cfg.skip_log_interval:
                self._last_skip_log=now
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
        reward=profit/stake
        next_s=self.rl.state_index(0.5,0,float(np.mean(self._recent_wr)),0.0)
        self.rl.update(reward,next_s)

        if len(self._feat_buf)==self.cfg.nn_input_window:
            X=np.stack(list(self._feat_buf),axis=0)
            # Target: p_over2=1 if won, 0 if lost
            y=np.array([1.0 if won else 0.0,
                        0.0 if won else 1.0,
                        1.0 if won else 0.3],dtype=np.float32)
            self.nn.record(X,y)

        self.cal.record(fusion.neural_p_over2, won)
        self.learner.record({"nn_over2":fusion.neural_p_over2>0.5,
                             "over2_markov":fusion.over2_markov_p>self.profile.live_over2_rate,
                             "vol_burst":fusion.volatility_score>0.5}, won)
        self.save_state()

    def _log_skip_summary(self):
        total=sum(self._skip.values())
        if not total: return
        s=" | ".join(f"{k}:{v}({v/total*100:.0f}%)"
                     for k,v in self._skip.most_common(5))
        log.info(f"[{self.symbol} SKIPS t={self._ticks_after_warmup}] {s}")


# ─────────────────────────────────────────────────────────────────────────────
# DERIV WEBSOCKET CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class DerivClient:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self._ws=None; self._rid=0
        self._pending:  Dict[int,asyncio.Future]={}
        self._tick_cbs: Dict[str,Callable]={}
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
        await self._send({"ticks":symbol,"subscribe":1,"req_id":self._next()})

    async def buy(self, symbol, stake) -> Optional[dict]:
        r=await self._rpc({"buy":1,"price":str(stake),"parameters":{
            "amount":str(stake),"basis":"stake",
            "contract_type":"DIGITOVER",
            "barrier":self.cfg.barrier,          # "2" → DIGITOVER 2
            "currency":self.cfg.currency,
            "duration":self.cfg.duration,
            "duration_unit":self.cfg.duration_unit,
            "symbol":symbol}})
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
            log.error(f"[{symbol}] dispatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BOT
# ─────────────────────────────────────────────────────────────────────────────
class Bot:
    def __init__(self, cfg: Config):
        self.cfg=cfg; self.client=DerivClient(cfg)
        self.history=History(cfg.history_file); self._alive=True
        self.engines={sym:SymbolEngine(sym,cfg) for sym in cfg.symbols}

    async def run(self):
        for eng in self.engines.values(): eng.load_state()
        retry=5
        while self._alive:
            try:
                log.info("Connecting...")
                await self.client.connect()
                await self.client.auth()
                for eng in self.engines.values():
                    eng.risk.set_balance(self.client.balance)
                for sym in self.cfg.symbols:
                    await self.client.subscribe_ticks(sym,self._make_cb(sym))
                    await asyncio.sleep(0.1)
                log.info(f"Live on {self.cfg.symbols} | DIGITOVER 2 | adaptive")
                retry=5
                while self._alive and self.client.connected:
                    await asyncio.sleep(1)
                if self._alive:
                    log.warning("Disconnected — reconnect in 5s")
                    await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Error: {e} — retry in {retry}s")
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
            eng._skip[reason.split(":")[0]]+=1; return
        await self._execute(symbol,eng,fusion)

    async def _execute(self, symbol, eng: SymbolEngine, fusion: FusionResult):
        stake=max(min(eng.risk.current_stake,
                      round(self.client.balance*self.cfg.max_balance_pct,2)),
                  self.cfg.base_stake)
        log.info(
            f"[{symbol}] TRADE DIGITOVER2 ${stake:.2f} "
            f"step={eng.risk._mart_step}/{self.cfg.martingale_steps} "
            f"conf={fusion.final_confidence:.3f} "
            f"p_over2={fusion.neural_p_over2:.4f} "
            f"markov={fusion.over2_markov_p:.4f} "
            f"loss_live={fusion.snap_short_loss:.3f}/<={fusion.snap_loss_gate:.3f} "
            f"regime={fusion.regime_name} bal=${self.client.balance:.2f}"
        )
        rl_s=eng.get_rl_state(fusion); eng.risk.on_open()
        result=await self.client.buy(symbol, stake)
        if not result:
            eng.risk.release_lock(); return
        cid=result.get("contract_id"); buy_price=float(result.get("buy_price",stake))
        self.history.add({
            "ts":datetime.utcnow().isoformat(),"symbol":symbol,
            "tick":eng._tick,"contract_id":cid,"stake":buy_price,
            "final_confidence":fusion.final_confidence,
            "neural_p_over2":fusion.neural_p_over2,
            "over2_markov_p":fusion.over2_markov_p,
            "regime":fusion.regime_name,"entropy_score":fusion.entropy_score,
            "snap_min_over2":fusion.snap_min_over2,
            "snap_loss_gate":fusion.snap_loss_gate,
            "snap_short_loss":fusion.snap_short_loss,
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
                    profit=(float(ap) if ap is not None
                            else float(sp)-buy_price if sp else 0.0)
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
            f"profit={profit:+.4f} bal=${self.client.balance:.2f} | "
            f"ALL WR={st['win_rate']:.1%} n={st['n']} PnL={st['pnl']:+.4f} | " +
            " ".join(f"{s} WR={bs.get(s,{}).get('win_rate',0):.1%}"
                     f"({bs.get(s,{}).get('n',0)})" for s in SYMBOLS)
        )

    def shutdown(self):
        self._alive=False
        st=self.history.stats
        log.info(f"Shutdown | WR={st['win_rate']:.1%} n={st['n']} PnL={st['pnl']:+.4f}")
        for sym,eng in self.engines.items():
            log.info(f"  [{sym}] ticks={eng._tick} | {eng.profile.summary()}")


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
                    "contract":"DIGITOVER 2","symbols":bot.cfg.symbols,
                    "trades":st["n"],"win_rate":round(st["win_rate"],4),
                    "pnl":st["pnl"],"balance":bot.client.balance,
                    "engines":{
                        sym:{
                            "ticks":eng._tick,
                            "profile":{
                                "over2_rate":round(eng.profile.live_over2_rate,4),
                                "loss_rate": round(eng.profile.live_loss_rate,4),
                                "d0":round(eng.profile.live_d0,4),
                                "d1":round(eng.profile.live_d1,4),
                                "d2":round(eng.profile.live_d2,4),
                                "markov_stab":round(eng.profile.live_markov_stab,4),
                            },
                            "thresholds":{
                                "min_over2":eng.profile.min_over2_prob,
                                "loss_gate":eng.profile.loss_gate,
                                "null_p":   eng.profile.null_p_value,
                                "markov_margin":eng.profile.markov_margin,
                                "zscore_gate":eng.profile.zscore_gate,
                            },
                            "trades":bs.get(sym,{}).get("n",0),
                            "win_rate":round(bs.get(sym,{}).get("win_rate",0),4),
                            "pnl":bs.get(sym,{}).get("pnl",0),
                            "epsilon":round(eng.rl.epsilon,4),
                            "martingale":eng.risk._mart_step,
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
                       f"<td class='pos'>{p.live_over2_rate:.4f}</td>"
                       f"<td>{p.live_loss_rate:.4f}</td>"
                       f"<td>{p.live_d0:.3f} / {p.live_d1:.3f} / {p.live_d2:.3f}</td>"
                       f"<td>{p.live_markov_stab:.4f}</td>"
                       f"<td>{p.min_over2_prob:.4f}</td>"
                       f"<td>{p.loss_gate:.4f}</td>"
                       f"<td>{p.null_p_value:.2f}</td>"
                       f"<td>{eng._tick}</td>"
                       f"<td class='{'pos' if ss.get('win_rate',0)>=0.513 else 'neg'}'>"
                       f"{ss.get('win_rate',0):.1%}({ss.get('n',0)})</td>"
                       f"<td class='{'pos' if pnl>=0 else 'neg'}'>${pnl:+.4f}</td>"
                       f"<td class='{'neg' if eng.risk._paused else 'pos'}'>"
                       f"{'PAUSED' if eng.risk._paused else 'OK'}</td></tr>")

            html=f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>DIGITOVER2 Adaptive Bot</title>
<style>
body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:2rem;}}
h1{{color:#58a6ff;}} h3{{color:#58a6ff;margin-top:1.5rem;}}
.sub{{color:#8b949e;font-size:0.85rem;margin-bottom:1.5rem;}}
table{{border-collapse:collapse;width:100%;margin-bottom:1rem;}}
td,th{{padding:0.35rem 0.7rem;border:1px solid #21262d;font-size:0.85rem;}}
th{{background:#161b22;color:#8b949e;font-weight:normal;}}
.pos{{color:#3fb950;}} .neg{{color:#f85149;}} .neu{{color:#d29922;}}
</style></head><body>
<h1>DIGITOVER 2 — Adaptive Bot</h1>
<div class="sub">
  1HZ100V + JD25 + 1HZ25V | Selected from 191,739 real ticks |
  Self-calibrating every {bot.cfg.adapt_every} ticks | Refreshes 10s<br>
  Breakeven: 51.28% | Theoretical base rate: 70% | Bootstrap edge: +1.6% to +6.0%
</div>
<h3>Live Adaptive State</h3>
<table>
<tr><th>Symbol</th><th>Live over2</th><th>Live loss</th><th>d0/d1/d2</th>
<th>Markov stab</th><th>min_over2</th><th>loss_gate</th>
<th>null_p</th><th>Ticks</th><th>Win rate</th><th>P&L</th><th>Status</th></tr>
{rows}
</table>
<h3>Bootstrap reference (from 191,739 real ticks)</h3>
<table>
<tr><th>Symbol</th><th>Observed over2</th><th>Edge vs 70%</th>
<th>d0</th><th>d1</th><th>d2</th><th>Markov stab</th><th>Ticks</th></tr>
<tr><td>1HZ100V</td><td class="pos">71.60%</td><td class="pos">+1.60%</td>
<td>10.8%</td><td class="pos">5.6%</td><td>12.0%</td><td>0.09633</td><td>14,799</td></tr>
<tr><td>JD25</td><td class="pos">76.00%</td><td class="pos">+6.00%</td>
<td>7.6%</td><td>10.4%</td><td class="pos">6.0%</td><td>0.06830</td><td>16,146</td></tr>
<tr><td>1HZ25V</td><td class="pos">74.00%</td><td class="pos">+4.00%</td>
<td>8.4%</td><td>8.8%</td><td>8.8%</td><td>0.08186</td><td>14,801</td></tr>
</table>
<h3>Combined Stats</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total trades</td><td><strong>{st['n']}</strong></td></tr>
<tr><td>Win rate</td>
<td class="{'pos' if st['win_rate']>=0.513 else 'neg'}">
<strong>{st['win_rate']:.1%}</strong>
<span style="color:#8b949e"> (breakeven 51.28%)</span></td></tr>
<tr><td>P&L</td>
<td class="{'pos' if st['pnl']>=0 else 'neg'}"><strong>${st['pnl']:+.4f}</strong></td></tr>
<tr><td>Balance</td><td><strong>${bot.client.balance:.2f}</strong></td></tr>
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
    log.info("DIGITOVER 2 Adaptive Bot")
    log.info("Symbols  : 1HZ100V (primary) + JD25 + 1HZ25V")
    log.info("           Selected from 191,739 real ticks — all 15 symbols compared")
    log.info("Contract : DIGITOVER 2 (wins if last digit ∈ {3,4,5,6,7,8,9})")
    log.info("Bootstrap: 1HZ100V over2=71.6%  JD25 over2=76.0%  1HZ25V over2=74.0%")
    log.info("Note     : R_10/R_25/R_100 NOT used — over2 rates 65.6%–67.6% (<70%)")
    log.info(f"Stake    : " + " -> ".join(
        f"${cfg.base_stake*cfg.martingale_factor**s:.2f}"
        for s in range(cfg.martingale_steps+1)) + " -> halt")
    log.info(f"Adapt    : every {cfg.adapt_every} ticks per symbol")
    log.info("="*72)

    _start_health_server(bot)
    await bot.run()


if __name__=="__main__":
    asyncio.run(live(Config()))
