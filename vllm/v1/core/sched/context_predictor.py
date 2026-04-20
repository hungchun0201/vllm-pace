"""ContextServe: Context-Aware Tool Latency Predictor.

Replaces Continuum's tool-name-only CDF with a hybrid predictor
that uses full bash command arguments for Fast/Slow/Uncertain
three-way classification.

Three-layer prediction strategy:
1. Rule-based fast path (zero latency, covers most common cases)
2. XGBoost model (< 1ms inference, covers long-tail)
3. Fallback to Continuum CDF (guarantees worst-case = SOTA)
"""
import hashlib
import os
import re
from enum import Enum
from typing import Optional, Tuple


def _stable_hash(s: str, mod: int = 100) -> int:
    """Deterministic hash that is consistent across Python processes.

    Python's built-in hash() is randomized by PYTHONHASHSEED,
    which would cause training/inference feature mismatch.
    """
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

# Lazy import to avoid torch dependency at module level
_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        try:
            from vllm.logger import init_logger
            _logger = init_logger(__name__)
        except ImportError:
            import logging
            _logger = logging.getLogger(__name__)
    return _logger


class ToolSpeed(Enum):
    FAST = "fast"
    SLOW = "slow"
    UNCERTAIN = "uncertain"


class ContextAwarePredictor:
    """Hybrid context-aware tool latency predictor.

    Prediction layers:
    1. Rule-based fast path (zero latency, covers common cases)
    2. Bayesian online learning (NIG + Thompson Sampling, < 1ms)
       OR XGBoost model (< 1ms, covers long-tail)
    3. Fallback to Continuum CDF (guarantees worst-case >= SOTA)
    """

    # Tools that are almost certainly fast (exec_time < 0.5s)
    DEFINITELY_FAST = frozenset({
        "ls", "cat", "head", "tail", "echo", "pwd", "cd", "which",
        "type", "wc", "basename", "dirname", "readlink", "stat",
        "true", "false", "date", "whoami", "hostname", "uname",
        "env", "export", "set", "unset", "printf", "touch",
        "mkdir", "rmdir", "mv", "cp", "rm", "ln", "chmod", "chown",
    })

    # Multi-word prefixes that are almost certainly slow (exec_time > 5s)
    DEFINITELY_SLOW_PREFIXES = (
        "make", "cmake", "cargo build", "go build", "mvn", "gradle",
        "docker build", "docker pull",
        "apt install", "apt-get install",
        "pip install", "npm install", "yarn install", "conda install",
    )

    # Regex patterns for slow commands
    SLOW_PATTERNS = [
        re.compile(r"pytest\s+(?!.*-x)(?!.*--co)(?!.*--collect-only)"),
        re.compile(r"python\s+\S*(?:train|eval|benchmark|setup\.py)"),
        re.compile(r"pip\s+install"),
        re.compile(r"git\s+clone"),
        re.compile(r"wget\s"),
        re.compile(r"curl\s.*-[oO]"),
    ]

    # Regex patterns for fast commands
    FAST_PATTERNS = [
        re.compile(r"git\s+(?:status|log|diff|show|branch|rev-parse|remote)"),
        re.compile(r"python\s+-c\s"),
        re.compile(r"find\s+.*-maxdepth\s+[12]\s"),
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        fast_threshold: float = 0.2,
        slow_threshold: float = 0.8,
        use_bayesian: bool = True,
    ):
        self.fast_threshold = fast_threshold
        self.slow_threshold = slow_threshold
        self.xgb_model = None
        self.bayesian_predictor = None

        # Layer 2a: Bayesian online learning (preferred)
        if use_bayesian:
            try:
                from vllm.v1.core.bayesian_predictor import (
                    BayesianToolPredictor, TTLPolicy,
                )
                self.bayesian_predictor = BayesianToolPredictor(
                    cold_start_threshold=5,
                    ttl_policy=TTLPolicy.QUANTILE,
                    ttl_quantile=0.9,
                    window_size=200,
                )
                _get_logger().info("Initialized Bayesian TTL predictor (NIG + TS)")
            except Exception as e:
                _get_logger().warning(f"Failed to init Bayesian predictor: {e}")

        # Layer 2b: XGBoost fallback (if Bayesian unavailable or model provided)
        if model_path and os.path.exists(model_path):
            try:
                import xgboost as xgb
                self.xgb_model = xgb.Booster({"nthread": 1})
                self.xgb_model.load_model(model_path)
                _get_logger().info(
                    f"Loaded XGBoost predictor from {model_path}"
                )
            except Exception as e:
                _get_logger().warning(
                    f"Failed to load XGBoost model: {e}"
                )

    def predict(
        self,
        full_command: Optional[str],
        continuum_cdf_ttl: float,
    ) -> Tuple[ToolSpeed, float]:
        """Predict tool speed and return suggested pin duration.

        Args:
            full_command: Complete bash command string.
            continuum_cdf_ttl: Continuum's CDF-based TTL as fallback.

        Returns:
            (ToolSpeed, suggested_ttl):
              FAST:      (FAST, positive_pin_duration)  -> Strict Pin
              SLOW:      (SLOW, 0.0)                    -> Proactive Swap-out
              UNCERTAIN: (UNCERTAIN, continuum_cdf_ttl)  -> Fallback
        """
        if not full_command or not full_command.strip():
            return ToolSpeed.UNCERTAIN, continuum_cdf_ttl

        words = full_command.strip().split()
        if not words:
            return ToolSpeed.UNCERTAIN, continuum_cdf_ttl

        tool_name = words[0]

        # === Layer 1: Deterministic rules (zero latency) ===

        # Check definitely fast tools
        if tool_name in self.DEFINITELY_FAST:
            return ToolSpeed.FAST, 3.0

        # Check slow prefix patterns (multi-word matches)
        for slow_prefix in self.DEFINITELY_SLOW_PREFIXES:
            if full_command.strip().startswith(slow_prefix):
                return ToolSpeed.SLOW, 0.0

        # Check regex slow patterns
        for pattern in self.SLOW_PATTERNS:
            if pattern.search(full_command):
                return ToolSpeed.SLOW, 0.0

        # Check regex fast patterns
        for pattern in self.FAST_PATTERNS:
            if pattern.search(full_command):
                return ToolSpeed.FAST, 3.0

        # === Layer 2a: Bayesian online learning (< 1ms) ===
        if self.bayesian_predictor is not None:
            try:
                speed, ttl = self.bayesian_predictor.predict_ttl(
                    full_command, continuum_cdf_ttl,
                )
                if speed != ToolSpeed.UNCERTAIN:
                    return speed, ttl
            except Exception as e:
                _get_logger().debug(f"Bayesian prediction failed: {e}")

        # === Layer 2b: XGBoost model (< 1ms, fallback) ===
        if self.xgb_model is not None:
            try:
                import xgboost as xgb
                features = self._extract_features(full_command)
                dmat = xgb.DMatrix([features])
                prob_slow = float(self.xgb_model.predict(dmat)[0])

                if prob_slow > self.slow_threshold:
                    return ToolSpeed.SLOW, 0.0
                elif prob_slow < self.fast_threshold:
                    return ToolSpeed.FAST, 3.0
                else:
                    return ToolSpeed.UNCERTAIN, continuum_cdf_ttl
            except Exception as e:
                _get_logger().debug(f"XGBoost prediction failed: {e}")

        # === Layer 3: Graceful Fallback -> Continuum CDF ===
        return ToolSpeed.UNCERTAIN, continuum_cdf_ttl

    def _extract_features(self, full_command: str) -> list:
        """Extract feature vector from a bash command.

        Feature vector must be consistent with
        scripts/extract_features.py for training compatibility.
        """
        words = full_command.split()
        tool_name = words[0] if words else ""

        # Use word-set for accurate flag detection (avoids substring false positives)
        word_set = set(words)

        return [
            _stable_hash(tool_name),                                     # 0: tool_name_encoded
            len(words) - 1,                                           # 1: arg_count
            len(full_command),                                        # 2: command_length
            int("|" in full_command),                                 # 3: has_pipe
            int(">" in full_command),                                 # 4: has_redirect
            int(";" in full_command),                                 # 5: has_semicolon
            int("&&" in full_command),                                # 6: has_and
            int(bool(re.search(                                       # 7: has_loop
                r'\b(?:while|for|until)\b', full_command))),
            int("$(" in full_command or "`" in full_command),         # 8: has_subshell
            int(bool(word_set & {"-r", "-R", "--recursive"})),        # 9: has_recursive
            int("-v" in word_set),                                    # 10: has_verbose
            int("-f" in word_set),                                    # 11: has_force
            int(tool_name in {"ls", "cat", "head", "tail", "wc",     # 12: is_read_only
                              "echo", "pwd", "which", "type"}),
            int(tool_name in {"grep", "find", "ag", "rg", "locate"}),# 13: is_search
            int(tool_name in {"sed", "awk", "patch", "ed"}),         # 14: is_edit
            int(tool_name in {"make", "cmake", "cargo",              # 15: is_build
                              "go", "gcc", "g++", "javac"}),
            int(tool_name in {"pytest", "python", "node",            # 16: is_test_runtime
                              "java", "cargo"}),
            int(tool_name == "git"),                                  # 17: is_git
            int(tool_name in {"curl", "wget", "ssh", "scp"}),        # 18: is_network
            int(tool_name in {"pip", "npm", "apt", "conda", "yarn"}),# 19: is_install
            int(tool_name == "python" and any(                        # 20: python_is_test
                kw in full_command
                for kw in ["test", "pytest", "unittest"])),
            int(tool_name == "python" and "-c" in word_set),          # 21: python_is_inline
            int(tool_name == "python" and "-m" in word_set),          # 22: python_is_module
            int(any("test" in w.lower() for w in words[1:])),         # 23: target_has_test
            int(any(w.endswith(".py") for w in words[1:])),           # 24: target_has_py
        ]

    def observe_execution(self, full_command: str, exec_time: float) -> None:
        """Feed real execution time back to Bayesian predictor for online learning.

        Called when a tool call completes and we know the actual exec_time.
        This enables the Bayesian posterior to update and improve future TTL
        predictions. No-op if Bayesian predictor is not active.
        """
        if self.bayesian_predictor is not None and exec_time > 0:
            self.bayesian_predictor.observe(full_command, exec_time)
