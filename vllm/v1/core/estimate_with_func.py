# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal ToolCallEstimator port for Continuum.

Ported from vllm-continuum/vllm/v1/core/estimate_with_func.py. Stripped the
ASTERIA / ContextServe predictor branches; keeps only the Continuum pin-TTL
decision path:

  * `set_up_pin(request)` returns `FIXED_THRESHOLD_CONTINUUM` (2.0s) if the
    predicted tool-call execution time is ≤ threshold, else 0 (no pin).
  * Execution time is a rolling mean of past gaps between `request_finished`
    and `request_arrives` for the same job's tool call.

Without this port, the Agentic_KVCache_management scheduler had a hardcoded
10-second pin on every finished non-last-step request, causing excessive
VRAM hold under JPS=10 and a spurious +17% slowdown vs FCFS on H100 80GB.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Optional

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request


def _import_get_tokenizer():
    """Lazy-import get_tokenizer from whichever module provides it in this vLLM version."""
    try:
        from vllm.tokenizers import get_tokenizer  # vLLM 0.19+
        return get_tokenizer
    except ImportError:
        pass
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer  # vLLM 0.10.x / older
        return get_tokenizer
    except ImportError:
        return None

logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds


class ToolCallParser:
    """Extract the first tool-call command from an LLM-generated bash block.

    Matches the mini-swe-agent parsing convention:
        ```bash
        <command> <args>
        ```
    Returns the command token (first word) or None if no single bash block.
    """

    _PATTERN = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

    def parse(self, text: str) -> Optional[str]:
        actions = self._PATTERN.findall(text)
        if len(actions) != 1:
            return None
        words = actions[0].strip().split()
        return words[0] if words else None


class ToolCallEstimator:
    """Rolling-mean predictor for tool-call execution times, per function name.

    Use by the Continuum scheduler to decide whether pinning a finished
    request's KV cache is worthwhile. If the tool call historically runs
    longer than the pin threshold, the next turn won't arrive in time to
    reuse the pin — so we skip pinning and free blocks immediately.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        model_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
        parser: Optional[ToolCallParser] = None,
    ) -> None:
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}
        self.job_to_history: dict[str, list[dict]] = {}

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            get_tokenizer = _import_get_tokenizer()
            if get_tokenizer is None:
                logger.warning("No get_tokenizer available; this_func_call parsing disabled.")
                self.tokenizer = None
            else:
                try:
                    # vLLM versions differ in keyword argument name:
                    # 0.19+: tokenizer_name, 0.10.x: tokenizer_name_or_path
                    try:
                        self.tokenizer = get_tokenizer(
                            tokenizer_name=model_name,
                            tokenizer_mode=tokenizer_mode,
                            trust_remote_code=trust_remote_code,
                            revision=tokenizer_revision,
                        )
                    except TypeError:
                        # Older signature variant — try positional + minimal kwargs.
                        self.tokenizer = get_tokenizer(model_name)
                    logger.info(f"ToolCallEstimator tokenizer loaded: {model_name}")
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
                    self.tokenizer = None
        else:
            self.tokenizer = None

        self.parser = parser if parser is not None else ToolCallParser()

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        return self.func_call_to_exec_time.get(func)

    def update_func_call_exec_time(self, job_id: str) -> None:
        history = self.job_to_history.get(job_id) or []
        if not history:
            return
        last = history[-1]
        if "departure_time" not in last or "func_call" not in last:
            return
        func = last["func_call"]
        if func is None:
            return
        exec_time = time.time() - last["departure_time"]
        self.record_func_call_to_exec_time.setdefault(func, []).append(exec_time)
        samples = self.record_func_call_to_exec_time[func]
        self.func_call_to_exec_time[func] = sum(samples) / len(samples)

    # --------------------------------------------------------------
    # Scheduler-facing API
    # --------------------------------------------------------------
    def set_up_pin(self, request: "Request") -> float:
        """Return pin TTL seconds for this finished request (0 = no pin)."""
        if getattr(request, "this_func_call", None) is None:
            return 0.0
        exec_time = self.get_func_call_exec_time(request.this_func_call) or 0.0
        if exec_time > FIXED_THRESHOLD_CONTINUUM:
            return 0.0
        return FIXED_THRESHOLD_CONTINUUM

    def request_arrives(self, request: "Request") -> None:
        jid = getattr(request, "job_id", None)
        if jid is None:
            return
        if jid not in self.job_to_history:
            self.job_to_history[jid] = [{"arrival_time": request.arrival_time}]
            return
        # Propagate the previous turn's func_call to this request so the
        # queue can use it for prefetch / admission decisions.
        prev = self.job_to_history[jid][-1]
        request.last_func_call = prev.get("func_call")
        self.update_func_call_exec_time(jid)
        self.job_to_history[jid].append({"arrival_time": request.arrival_time})

    def request_finished(self, request: "Request") -> None:
        jid = getattr(request, "job_id", None)
        if jid is None:
            return
        this_func_call: Optional[str] = None
        if self.tokenizer is not None and len(request.output_token_ids) > 0:
            try:
                output_text = self.tokenizer.decode(
                    list(request.output_token_ids), skip_special_tokens=True
                )
                this_func_call = self.parser.parse(output_text)
            except Exception as e:  # pragma: no cover
                logger.debug(f"detokenize/parse failed for {request.request_id}: {e}")
        request.this_func_call = this_func_call
        self.job_to_history.setdefault(jid, []).append({
            "departure_time": time.time(),
            "func_call": this_func_call,
        })
