# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-step scheduler queue snapshot trace.

Enabled by ``SCHED_TRACE_PATH`` environment variable. When set, each scheduling
step writes:

* One line to ``<SCHED_TRACE_PATH>.steps.jsonl`` — slim record with req_id lists
  and kv-cache manager state.
* For every observed request, one line to ``<SCHED_TRACE_PATH>.requests.jsonl``
  — full observable state dump (keyed by (step, req_id)).

This split keeps the step file small for fast loading in a visualizer while
preserving the ability to drill into any individual request at any step.
"""

import json
import os
import time
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from vllm.v1.request import Request


_TRACE_PATH: str | None = os.environ.get("SCHED_TRACE_PATH")
_STEPS_FILE: Any = None
_REQS_FILE: Any = None

_POLICY: str = os.environ.get("SCHED_TRACE_POLICY", "unknown")
_MAX_PROMPT_FP: int = int(os.environ.get("SCHED_TRACE_PROMPT_FP", "16"))
_MAX_OUTPUT_TAIL: int = int(os.environ.get("SCHED_TRACE_OUTPUT_TAIL", "32"))
_MAX_STEPS: int = int(os.environ.get("SCHED_TRACE_MAX_STEPS", "2000"))
_EVERY_N: int = max(1, int(os.environ.get("SCHED_TRACE_EVERY_N", "1")))
_TOKENIZER_PATH: str | None = os.environ.get("SCHED_TRACE_TOKENIZER")  # if set, decode snippets
_STEPS_EMITTED: int = 0
_TOKENIZER: Any = None  # lazy-loaded


def _decode(ids):
    """Decode a token-id list to text using the configured tokenizer (if any)."""
    global _TOKENIZER
    if not _TOKENIZER_PATH or not ids:
        return None
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_PATH)
        except Exception:
            _TOKENIZER = False  # mark as failed so we don't retry every call
            return None
    if _TOKENIZER is False:
        return None
    try:
        return _TOKENIZER.decode(ids, skip_special_tokens=False)
    except Exception:
        return None


def enabled() -> bool:
    return _TRACE_PATH is not None


def _open_lazy() -> None:
    global _STEPS_FILE, _REQS_FILE
    if _TRACE_PATH is None:
        return
    if _STEPS_FILE is None:
        _STEPS_FILE = open(f"{_TRACE_PATH}.steps.jsonl", "a", buffering=1)
    if _REQS_FILE is None:
        _REQS_FILE = open(f"{_TRACE_PATH}.requests.jsonl", "a", buffering=1)


def _req_to_dict(req: "Request", step: int, ts: float, queue: str) -> dict:
    """Serialize one Request to a trace dict. Token lists are truncated to
    fingerprints; counts and every scalar attribute are preserved verbatim so
    the HTML viewer can drill into details."""
    prompt_ids = req.prompt_token_ids or []
    try:
        output_ids = list(req._output_token_ids)
    except AttributeError:
        output_ids = []
    return {
        "step": step,
        "ts": ts,
        "queue": queue,  # "running" | "waiting"
        "req_id": req.request_id,
        "job_id": getattr(req, "job_id", None),
        "is_last_step": getattr(req, "is_last_step", False),
        "status": str(req.status),
        "priority": req.priority,
        "arrival_time": req.arrival_time,
        "client_index": req.client_index,
        "max_tokens": req.max_tokens,
        "num_prompt_tokens": req.num_prompt_tokens,
        "num_tokens": req.num_tokens,
        "num_output_tokens": req.num_output_tokens,
        "num_computed_tokens": req.num_computed_tokens,
        "num_cached_tokens": req.num_cached_tokens,
        "num_preemptions": req.num_preemptions,
        "num_external_computed_tokens": req.num_external_computed_tokens,
        "num_nans_in_logits": req.num_nans_in_logits,
        "num_encoder_inputs": req.num_encoder_inputs,
        "is_prefill_chunk": req.is_prefill_chunk,
        "resumable": req.resumable,
        "cache_salt": req.cache_salt,
        "stop_reason": req.stop_reason,
        "kv_transfer_params": req.kv_transfer_params,
        # Fingerprints (first N and last N prompt token ids + last M output)
        "prompt_prefix": prompt_ids[:_MAX_PROMPT_FP],
        "prompt_suffix": prompt_ids[-_MAX_PROMPT_FP:] if len(prompt_ids) > _MAX_PROMPT_FP else [],
        "output_tail": output_ids[-_MAX_OUTPUT_TAIL:],
        # Decoded text (only when SCHED_TRACE_TOKENIZER env is set).
        "prompt_prefix_text": _decode(prompt_ids[:_MAX_PROMPT_FP]),
        "prompt_suffix_text": _decode(prompt_ids[-_MAX_PROMPT_FP:]) if len(prompt_ids) > _MAX_PROMPT_FP else None,
        "output_tail_text": _decode(output_ids[-_MAX_OUTPUT_TAIL:]),
        "spec_tokens_count": len(req.spec_token_ids),
        "block_hashes_count": len(req.block_hashes),
    }


def step_snapshot(
    step: int,
    running: Iterable["Request"],
    waiting: Iterable["Request"],
    free_blocks: int | None = None,
    total_blocks: int | None = None,
    num_pinned: int = 0,
    **extra: Any,
) -> None:
    """Emit one step record + one record per observed request.

    Meant to be called at the top of ``Scheduler.schedule()`` before any
    scheduling mutation happens, so the snapshot reflects exactly what the
    scheduler is about to decide over.
    """
    global _STEPS_EMITTED
    if _TRACE_PATH is None:
        return
    if _STEPS_EMITTED >= _MAX_STEPS:
        return
    if step % _EVERY_N != 0:
        return
    _open_lazy()

    ts = time.time()
    running_list = list(running)
    waiting_list = list(waiting)

    step_rec = {
        "event": "step_snapshot",
        "step": step,
        "ts": ts,
        "policy": _POLICY,
        "free_blocks": free_blocks,
        "total_blocks": total_blocks,
        "num_running": len(running_list),
        "num_waiting": len(waiting_list),
        "num_pinned": num_pinned,
        "running_req_ids": [r.request_id for r in running_list],
        "waiting_req_ids": [r.request_id for r in waiting_list],
        "running_job_ids": [getattr(r, "job_id", None) for r in running_list],
        "waiting_job_ids": [getattr(r, "job_id", None) for r in waiting_list],
        **extra,
    }
    _STEPS_FILE.write(json.dumps(step_rec, default=str) + "\n")

    for r in running_list:
        _REQS_FILE.write(json.dumps(_req_to_dict(r, step, ts, "running"), default=str) + "\n")
    for r in waiting_list:
        _REQS_FILE.write(json.dumps(_req_to_dict(r, step, ts, "waiting"), default=str) + "\n")
    _STEPS_EMITTED += 1


def prefix_cache_event(
    step: int,
    req_id: str,
    job_id: Optional[str],
    local_hit_tokens: int,
    external_hit_tokens: int,
    total_prompt_tokens: int,
    num_tokens: int,
    connector_name: Optional[str] = None,
    load_kv_async: bool = False,
    **extra: Any,
) -> None:
    """Emit one record per first-schedule prefix-cache lookup.

    Captures the split between vLLM's local prefix-cache hit
    (kv_cache_manager.get_computed_blocks) and the connector's external
    contribution (CPUOffload / LMCache / etc).

    Called once per request the first time it gets scheduled (when
    request.num_computed_tokens transitions from 0 to >0).
    """
    if _TRACE_PATH is None:
        return
    if _STEPS_EMITTED >= _MAX_STEPS:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({
        "event": "prefix_cache_lookup",
        "step": step,
        "ts": time.time(),
        "policy": _POLICY,
        "req_id": req_id,
        "job_id": job_id,
        "local_hit_tokens": local_hit_tokens,
        "external_hit_tokens": external_hit_tokens,
        "total_hit_tokens": local_hit_tokens + external_hit_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "num_tokens": num_tokens,
        "connector": connector_name,
        "load_kv_async": load_kv_async,
        **extra,
    }, default=str) + "\n")


def step_decision(
    step: int,
    scheduled_new_req_ids: list[str],
    scheduled_resumed_req_ids: list[str],
    scheduled_running_req_ids: list[str],
    preempted_req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    **extra: Any,
) -> None:
    """Emit post-schedule decision classification.

    Categories mirror how scheduler.schedule() actually partitions requests:
      * new       — admitted from waiting for the first time
      * resumed   — re-admitted after preemption / remote-kv wait
      * running   — already running, granted more tokens this step
      * preempted — demoted from running to waiting this step
      * skipped_running / skipped_waiting are inferred client-side by
        set-difference against the step_snapshot queues.
    """
    if _TRACE_PATH is None:
        return
    if step % _EVERY_N != 0:
        return
    if _STEPS_EMITTED >= _MAX_STEPS:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({
        "event": "step_decision",
        "step": step,
        "ts": time.time(),
        "policy": _POLICY,
        "scheduled_new_req_ids": list(scheduled_new_req_ids),
        "scheduled_resumed_req_ids": list(scheduled_resumed_req_ids),
        "scheduled_running_req_ids": list(scheduled_running_req_ids),
        "preempted_req_ids": list(preempted_req_ids),
        "num_scheduled_tokens": dict(num_scheduled_tokens),
        **extra,
    }, default=str) + "\n")


# ---------------------------------------------------------------------------
# Legacy helpers retained so existing call sites (if any) keep compiling.
# They are no-ops unless SCHED_TRACE_PATH is set; they still write to the
# steps file under the "event" discriminator so they are not lost.
# ---------------------------------------------------------------------------

def step_begin(**kwargs: Any) -> None:
    if _TRACE_PATH is None:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({"event": "step_begin", "ts": time.time(), **kwargs}, default=str) + "\n")


def step_end(**kwargs: Any) -> None:
    if _TRACE_PATH is None:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({"event": "step_end", "ts": time.time(), **kwargs}, default=str) + "\n")


def request_done(**kwargs: Any) -> None:
    if _TRACE_PATH is None:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({"event": "request_done", "ts": time.time(), **kwargs}, default=str) + "\n")


def prefix_cache_hit(**kwargs: Any) -> None:
    if _TRACE_PATH is None:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({"event": "prefix_cache_hit", "ts": time.time(), **kwargs}, default=str) + "\n")


def allocate(**kwargs: Any) -> None:
    if _TRACE_PATH is None:
        return
    _open_lazy()
    _STEPS_FILE.write(json.dumps({"event": "allocate", "ts": time.time(), **kwargs}, default=str) + "\n")
