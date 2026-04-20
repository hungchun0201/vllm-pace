"""ASTERIA: Bayesian Online Tool Latency Predictor.

Uses Normal-Inverse-Gamma (NIG) conjugate priors to model log-normal
execution time distributions per tool context. Thompson Sampling provides
exploration-exploitation tradeoff for TTL selection.

Theory:
- NIG posterior updates are O(1) closed-form (conjugate to Gaussian likelihood)
- Posterior predictive is Student's t -> log-Student-t on actual times
- Thompson Sampling Bayesian regret: O(sqrt(T log K)) [Russo & Van Roy 2016]
- Sequential density estimation: minimax-optimal ln(T) + O(1) [Takeuchi & Barron 1997]

References:
- Murphy, "Conjugate Bayesian Analysis of the Gaussian Distribution" (UBC 2007)
- Russo & Van Roy, "An Information-Theoretic Analysis of TS" (JMLR 2016)
- Agrawal & Goyal, "Near-Optimal Regret Bounds for TS" (JACM 2017)
"""

import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# Lazy scipy import for environments without it
_scipy_stats = None


def _get_scipy_stats():
    global _scipy_stats
    if _scipy_stats is None:
        from scipy import stats
        _scipy_stats = stats
    return _scipy_stats


class TTLPolicy(Enum):
    """TTL selection strategy."""
    QUANTILE = "quantile"         # Deterministic: TTL = posterior quantile
    THOMPSON = "thompson"         # Stochastic: TTL from Thompson Sampling
    UPPER_CREDIBLE = "upper_ci"   # Conservative: TTL = upper 95% CI


@dataclass(frozen=True)
class NIGPosterior:
    """Normal-Inverse-Gamma posterior for log-normal execution times.

    Models Y = log(exec_time) ~ N(mu, sigma^2) with both mu and sigma^2
    unknown. The NIG prior is:
        mu | sigma^2 ~ N(mu_0, sigma^2 / lambda_0)
        sigma^2 ~ Inv-Gamma(alpha_0, beta_0)

    All updates are O(1) closed-form.
    """
    mu: float = 0.0       # posterior mean of log(exec_time)
    lam: float = 1.0      # precision scaling (number of pseudo-observations)
    alpha: float = 1.0    # shape of inverse-gamma (> 0)
    beta: float = 1.0     # scale of inverse-gamma (> 0)

    def update(self, x: float) -> "NIGPosterior":
        """Sequential update with a single observation x = log(exec_time).

        Returns a new NIGPosterior (immutable update).
        """
        new_lam = self.lam + 1.0
        new_mu = (self.lam * self.mu + x) / new_lam
        new_alpha = self.alpha + 0.5
        new_beta = (self.beta
                    + self.lam * (x - self.mu) ** 2 / (2.0 * new_lam))
        return NIGPosterior(
            mu=new_mu, lam=new_lam, alpha=new_alpha, beta=new_beta,
        )

    def batch_update(self, observations: List[float]) -> "NIGPosterior":
        """Batch update with multiple observations.

        More numerically stable than sequential updates for large batches.
        """
        if not observations:
            return self
        n = len(observations)
        x_bar = sum(observations) / n
        s_sq = (sum((x - x_bar) ** 2 for x in observations) / n
                if n > 1 else 0.0)

        new_lam = self.lam + n
        new_mu = (self.lam * self.mu + n * x_bar) / new_lam
        new_alpha = self.alpha + n / 2.0
        new_beta = (self.beta
                    + n * s_sq / 2.0
                    + n * self.lam * (x_bar - self.mu) ** 2
                    / (2.0 * new_lam))
        return NIGPosterior(
            mu=new_mu, lam=new_lam, alpha=new_alpha, beta=new_beta,
        )

    @property
    def predictive_df(self) -> float:
        """Degrees of freedom of posterior predictive (Student's t)."""
        return 2.0 * self.alpha

    @property
    def predictive_loc(self) -> float:
        """Location of posterior predictive (Student's t)."""
        return self.mu

    @property
    def predictive_scale(self) -> float:
        """Scale of posterior predictive (Student's t)."""
        return math.sqrt(self.beta * (self.lam + 1.0)
                         / (self.alpha * self.lam))

    @property
    def predictive_mean(self) -> float:
        """E[log(exec_time)] under posterior predictive.

        Only defined when alpha > 0.5 (df > 1).
        """
        return self.mu

    @property
    def predictive_variance(self) -> float:
        """Var[log(exec_time)] under posterior predictive.

        Only defined when alpha > 1 (df > 2).
        """
        if self.alpha <= 1.0:
            return float("inf")
        return (self.beta * (self.lam + 1.0)
                / (self.alpha * self.lam)
                * self.alpha / (self.alpha - 1.0))

    def quantile(self, p: float) -> float:
        """p-th quantile of posterior predictive in log-space.

        Returns the quantile of the Student's t posterior predictive.
        To get actual exec_time quantile, exponentiate the result.
        """
        stats = _get_scipy_stats()
        return float(stats.t.ppf(
            p, df=self.predictive_df,
            loc=self.predictive_loc,
            scale=self.predictive_scale,
        ))

    def exec_time_quantile(self, p: float) -> float:
        """p-th quantile of predicted execution time (in seconds)."""
        return math.exp(self.quantile(p))

    def sample_predictive(self) -> float:
        """Draw one sample from posterior predictive (for Thompson Sampling).

        Samples (mu, sigma^2) from NIG posterior, then draws x ~ N(mu, sigma^2).
        Returns log(exec_time).
        """
        import random
        stats = _get_scipy_stats()

        # Sample sigma^2 ~ Inv-Gamma(alpha, beta)
        # = 1 / Gamma(alpha, 1/beta)
        sigma2 = 1.0 / random.gammavariate(self.alpha, 1.0 / self.beta)

        # Sample mu ~ N(self.mu, sigma^2 / self.lam)
        mu_sample = random.gauss(self.mu, math.sqrt(sigma2 / self.lam))

        # Sample x ~ N(mu_sample, sigma^2)
        return random.gauss(mu_sample, math.sqrt(sigma2))


def _context_key_from_command(full_command: str) -> str:
    """Map a bash command to a context key for posterior grouping.

    More granular than Continuum's tool-name-only grouping but avoids
    overfitting to exact command strings. Groups by:
    - tool name (first word)
    - argument pattern (flags, target file extensions, subcommand)

    Examples:
        "python -c 'print(1)'"  -> "python:-c"
        "python train.py"       -> "python:.py"
        "python -m pytest"      -> "python:-m:pytest"
        "ls -la /tmp"           -> "ls"
        "make build"            -> "make:build"
        "git status"            -> "git:status"
        "pip install requests"  -> "pip:install"
    """
    if not full_command or not full_command.strip():
        return "__empty__"

    words = full_command.strip().split()
    if not words:
        return "__empty__"

    tool = words[0]

    # For python: distinguish by execution mode
    if tool == "python" or tool == "python3":
        if "-c" in words:
            return "python:-c"
        if "-m" in words:
            idx = words.index("-m")
            module = words[idx + 1] if idx + 1 < len(words) else "unknown"
            return f"python:-m:{module}"
        # Check target file extension pattern
        for w in words[1:]:
            if w.endswith(".py"):
                # Group by whether it looks like a test
                if any(kw in w.lower() for kw in ("test", "check", "lint")):
                    return "python:test_script"
                if any(kw in w.lower() for kw in
                       ("train", "eval", "bench", "setup")):
                    return "python:heavy_script"
                return "python:script"
        return "python:unknown"

    # For git: group by subcommand
    if tool == "git" and len(words) > 1:
        return f"git:{words[1]}"

    # For pip/npm/apt: group by subcommand
    if tool in ("pip", "pip3", "npm", "yarn", "apt", "apt-get", "conda"):
        sub = words[1] if len(words) > 1 else "unknown"
        return f"{tool}:{sub}"

    # For make/cmake: group by target
    if tool in ("make", "cmake"):
        target = words[1] if len(words) > 1 else "default"
        return f"{tool}:{target}"

    # For pytest: just "pytest"
    if tool == "pytest":
        return "pytest"

    # For docker: group by subcommand
    if tool == "docker":
        sub = words[1] if len(words) > 1 else "unknown"
        return f"docker:{sub}"

    # Default: just tool name (same as Continuum for simple tools)
    return tool


@dataclass
class RegretEntry:
    """Single regret observation for tracking."""
    context_key: str
    chosen_ttl: float
    actual_exec_time: float
    holding_cost: float   # max(0, ttl - actual) * memory_cost_rate
    miss_cost: float      # max(0, actual - ttl) * recompute_cost_rate


class BayesianToolPredictor:
    """Bayesian online predictor for tool execution time.

    For each context key (derived from tool name + argument pattern),
    maintains a NIG posterior over log(execution_time). Uses Thompson
    Sampling or posterior quantiles to select TTL values.

    Three-tier cold-start hierarchy:
    1. count < cold_start_threshold: use global prior
    2. count < 2 * cold_start_threshold: use tool-level posterior
    3. count >= 2 * cold_start_threshold: use context-specific posterior
    """

    # Thresholds for FAST/SLOW classification (in seconds)
    FAST_THRESHOLD = 1.0    # commands < 1s are FAST
    SLOW_THRESHOLD = 5.0    # commands > 5s are SLOW
    CONFIDENCE_THRESHOLD = 0.85  # confidence needed for FAST/SLOW call

    def __init__(
        self,
        prior: Optional[NIGPosterior] = None,
        cold_start_threshold: int = 5,
        ttl_policy: TTLPolicy = TTLPolicy.THOMPSON,
        window_size: int = 200,
        ttl_quantile: float = 0.9,
        memory_cost_rate: float = 1.0,
        recompute_cost_rate: float = 5.0,
    ):
        self.prior = prior or NIGPosterior(
            mu=0.0, lam=1.0, alpha=1.0, beta=1.0,
        )
        self.cold_start_threshold = cold_start_threshold
        self.ttl_policy = ttl_policy
        self.window_size = window_size
        self.ttl_quantile = ttl_quantile
        self.memory_cost_rate = memory_cost_rate
        self.recompute_cost_rate = recompute_cost_rate

        # Per-context posteriors
        self.posteriors: Dict[str, NIGPosterior] = {}
        # Per-tool posteriors (aggregated across contexts)
        self.tool_posteriors: Dict[str, NIGPosterior] = {}
        # Global posterior (all observations)
        self.global_posterior: NIGPosterior = NIGPosterior(
            mu=self.prior.mu, lam=self.prior.lam,
            alpha=self.prior.alpha, beta=self.prior.beta,
        )

        # Observation counts
        self.context_counts: Dict[str, int] = {}
        self.tool_counts: Dict[str, int] = {}
        self.total_count: int = 0

        # Sliding window for non-stationarity
        self.recent_observations: Dict[str, deque] = {}

        # Regret tracking
        self.regret_log: List[RegretEntry] = []

    def observe(self, full_command: str, exec_time: float) -> None:
        """Update posteriors with a new observation.

        Args:
            full_command: The bash command that was executed.
            exec_time: Actual execution time in seconds.
        """
        if exec_time <= 0:
            return

        context_key = _context_key_from_command(full_command)
        tool_name = self._tool_from_key(context_key)
        log_t = math.log(max(exec_time, 1e-9))

        # Update context-specific posterior
        posterior = self.posteriors.get(context_key, self.prior)
        self.posteriors[context_key] = posterior.update(log_t)
        self.context_counts[context_key] = (
            self.context_counts.get(context_key, 0) + 1
        )

        # Update tool-level posterior
        tool_post = self.tool_posteriors.get(tool_name, self.prior)
        self.tool_posteriors[tool_name] = tool_post.update(log_t)
        self.tool_counts[tool_name] = (
            self.tool_counts.get(tool_name, 0) + 1
        )

        # Update global posterior
        self.global_posterior = self.global_posterior.update(log_t)
        self.total_count += 1

        # Maintain sliding window
        if context_key not in self.recent_observations:
            self.recent_observations[context_key] = deque(
                maxlen=self.window_size,
            )
        window = self.recent_observations[context_key]
        window.append(log_t)

        # Recompute from window if full (handles non-stationarity)
        if len(window) >= self.window_size:
            self.posteriors[context_key] = self.prior.batch_update(
                list(window),
            )

    def predict_ttl(
        self,
        full_command: str,
        continuum_ttl: float = 0.0,
    ) -> Tuple["ToolSpeed", float]:
        """Predict tool speed class and suggested TTL.

        Uses the appropriate posterior (context/tool/global) based on
        cold-start hierarchy, then classifies and computes TTL.

        Args:
            full_command: The bash command to predict for.
            continuum_ttl: Continuum's CDF-based TTL as ultimate fallback.

        Returns:
            (ToolSpeed, suggested_ttl)
        """
        from vllm.v1.core.context_predictor import ToolSpeed

        context_key = _context_key_from_command(full_command)
        posterior = self._get_posterior(context_key)
        count = self._get_count(context_key)

        # Not enough data even at global level -> fall back to Continuum
        if count < 2:
            return ToolSpeed.UNCERTAIN, continuum_ttl

        # Classify based on posterior predictive
        try:
            p_fast = self._prob_below(posterior, math.log(self.FAST_THRESHOLD))
            p_slow = 1.0 - self._prob_below(
                posterior, math.log(self.SLOW_THRESHOLD),
            )

            if p_fast > self.CONFIDENCE_THRESHOLD:
                ttl = self._compute_ttl(posterior)
                return ToolSpeed.FAST, max(ttl, 0.5)

            if p_slow > self.CONFIDENCE_THRESHOLD:
                return ToolSpeed.SLOW, 0.0

            # UNCERTAIN: use TTL from posterior
            ttl = self._compute_ttl(posterior)
            return ToolSpeed.UNCERTAIN, max(ttl, continuum_ttl)

        except Exception:
            return ToolSpeed.UNCERTAIN, continuum_ttl

    def thompson_sample_ttl(self, full_command: str) -> float:
        """Thompson Sampling: sample exec_time from posterior, use as TTL.

        Provides exploration-exploitation tradeoff with sublinear regret.
        """
        context_key = _context_key_from_command(full_command)
        posterior = self._get_posterior(context_key)

        log_sample = posterior.sample_predictive()
        return max(math.exp(log_sample), 0.1)

    def log_regret(
        self,
        full_command: str,
        chosen_ttl: float,
        actual_exec_time: float,
    ) -> float:
        """Track regret for a single TTL decision.

        Returns the instantaneous regret (cost of mismatch).
        """
        context_key = _context_key_from_command(full_command)

        holding = max(0.0, chosen_ttl - actual_exec_time)
        miss = max(0.0, actual_exec_time - chosen_ttl)

        holding_cost = holding * self.memory_cost_rate
        miss_cost = miss * self.recompute_cost_rate

        entry = RegretEntry(
            context_key=context_key,
            chosen_ttl=chosen_ttl,
            actual_exec_time=actual_exec_time,
            holding_cost=holding_cost,
            miss_cost=miss_cost,
        )
        self.regret_log.append(entry)
        return holding_cost + miss_cost

    @property
    def cumulative_regret(self) -> float:
        """Total regret accumulated so far."""
        return sum(e.holding_cost + e.miss_cost for e in self.regret_log)

    def get_posterior_summary(self, full_command: str) -> dict:
        """Get posterior statistics for a command (for debugging/logging)."""
        context_key = _context_key_from_command(full_command)
        posterior = self._get_posterior(context_key)
        count = self._get_count(context_key)

        return {
            "context_key": context_key,
            "observation_count": count,
            "posterior_mu": posterior.mu,
            "posterior_lam": posterior.lam,
            "posterior_alpha": posterior.alpha,
            "posterior_beta": posterior.beta,
            "predictive_mean_log": posterior.predictive_mean,
            "predictive_mean_sec": math.exp(posterior.predictive_mean),
            "predictive_variance": posterior.predictive_variance,
            "cold_start_tier": self._cold_start_tier(context_key),
        }

    # ---- Private helpers ----

    def _get_posterior(self, context_key: str) -> NIGPosterior:
        """Get the appropriate posterior based on cold-start hierarchy."""
        tier = self._cold_start_tier(context_key)

        if tier == "context":
            return self.posteriors[context_key]
        elif tier == "tool":
            tool = self._tool_from_key(context_key)
            return self.tool_posteriors.get(tool, self.global_posterior)
        elif tier == "global":
            return self.global_posterior
        else:
            return self.prior

    def _get_count(self, context_key: str) -> int:
        """Get observation count at the effective tier."""
        tier = self._cold_start_tier(context_key)

        if tier == "context":
            return self.context_counts.get(context_key, 0)
        elif tier == "tool":
            tool = self._tool_from_key(context_key)
            return self.tool_counts.get(tool, 0)
        elif tier == "global":
            return self.total_count
        return 0

    def _cold_start_tier(self, context_key: str) -> str:
        """Determine which tier of the cold-start hierarchy to use."""
        ctx_count = self.context_counts.get(context_key, 0)

        if ctx_count >= 2 * self.cold_start_threshold:
            return "context"

        tool = self._tool_from_key(context_key)
        tool_count = self.tool_counts.get(tool, 0)

        if tool_count >= self.cold_start_threshold:
            return "tool"

        if self.total_count >= 2:
            return "global"

        return "prior"

    def _compute_ttl(self, posterior: NIGPosterior) -> float:
        """Compute TTL based on the active policy."""
        if self.ttl_policy == TTLPolicy.THOMPSON:
            log_sample = posterior.sample_predictive()
            return max(math.exp(log_sample), 0.1)

        elif self.ttl_policy == TTLPolicy.UPPER_CREDIBLE:
            log_q = posterior.quantile(0.95)
            return max(math.exp(log_q), 0.1)

        else:  # QUANTILE
            log_q = posterior.quantile(self.ttl_quantile)
            return max(math.exp(log_q), 0.1)

    def _prob_below(self, posterior: NIGPosterior, threshold: float) -> float:
        """P(log(exec_time) < threshold) under posterior predictive."""
        stats = _get_scipy_stats()
        return float(stats.t.cdf(
            threshold,
            df=posterior.predictive_df,
            loc=posterior.predictive_loc,
            scale=posterior.predictive_scale,
        ))

    @staticmethod
    def _tool_from_key(context_key: str) -> str:
        """Extract tool name from context key."""
        return context_key.split(":")[0]
