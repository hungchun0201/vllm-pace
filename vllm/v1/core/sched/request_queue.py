# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    CONTINUUM = "continuum"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class ContinuumRequestQueue(deque[Request], RequestQueue):
    """A job-aware FCFS queue for the Continuum scheduling policy.

    Continuum pins KV-cache blocks across multi-turn agent tool-call gaps.
    Within a scheduling step the queue prefers requests whose ``job_id`` is
    already pinned in VRAM (so their blocks stay warm).  Among unpinned jobs
    the request whose job was first seen is returned next (job-level FCFS).

    The scheduler calls ``update_pinned_state()`` once per step to tell the
    queue which job_ids are currently pinned.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pinned_job_ids: set[str] = set()
        self.job_id_first_entry_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Pinning state (updated by Scheduler each step)
    # ------------------------------------------------------------------

    def update_pinned_state(self, pinned_job_ids: set[str]) -> None:
        """Refresh the set of job_ids whose blocks are pinned in VRAM."""
        self._pinned_job_ids = pinned_job_ids

    # ------------------------------------------------------------------
    # RequestQueue interface
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add a request; record first-entry time for its job."""
        job_id = getattr(request, "job_id", None)
        if job_id is not None and job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[job_id] = time.time()
        self.append(request)

    def pop_request(self) -> Request:
        """Pop the highest-priority request according to Continuum order."""
        if not self:
            raise IndexError("pop from an empty queue")
        req = self.peek_request()
        self.remove(req)
        return req

    def peek_request(self) -> Request:
        """Return (without removing) the next request to be scheduled.

        Priority order:
        1. Requests whose job_id is pinned (earliest pinned job first).
        2. All other requests ordered by job first-entry time (FCFS per job).
        """
        if not self:
            raise IndexError("peek from an empty queue")

        # Collect pinned candidates
        pinned: list[Request] = [
            r for r in self if getattr(r, "job_id", None) in self._pinned_job_ids
        ]
        if pinned:
            # Return the request belonging to the earliest-pinned job.
            return min(
                pinned,
                key=lambda r: self.job_id_first_entry_time.get(
                    getattr(r, "job_id", ""), float("inf")
                ),
            )

        # No pinned candidates — standard job-level FCFS.
        return min(
            self,
            key=lambda r: self.job_id_first_entry_time.get(
                getattr(r, "job_id", ""), float("inf")
            ),
        )

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the deque."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue."""
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered = [r for r in self if r not in requests_to_remove]
        self.clear()
        self.extend(filtered)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue in deque order."""
        return super().__iter__()


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.CONTINUUM:
        return ContinuumRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
