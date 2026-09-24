"""FINAL IDEA 5.3--6.2: read-only core trials and committed first reads."""

import heapq
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .problem import Placement, UnitGraph


class FirstReadIndex:
    def __init__(self, capacity):
        self.capacity = capacity
        self.earliest = {}
        self.events = []
        self.position = {}
        self.times = []
        self.cumulative = [0]

    def contains_at(self, tensor, issue):
        if tensor not in self.position:
            return False
        index = self.position[tensor]
        stop = bisect_right(self.times, issue)
        return index < stop and self.cumulative[stop] - self.cumulative[index] <= self.capacity

    def commit(self, reads):
        changed = False
        for finish, tensor, size in reads:
            if not 0 < size <= self.capacity:
                continue
            if tensor in self.earliest:
                old = self.earliest[tensor]
                if old <= finish:
                    continue
                self.events.pop(bisect_left(self.events, (old, tensor, size)))
            self.earliest[tensor] = finish
            self.events.insert(bisect_right(self.events, (finish, tensor, size)), (finish, tensor, size))
            changed = True
        if changed:
            self.times = []
            self.position = {}
            self.cumulative = [0]
            for index, (finish, tensor, size) in enumerate(self.events):
                self.times.append(finish)
                self.position[tensor] = index
                self.cumulative.append(self.cumulative[-1] + size)


@dataclass
class Trial:
    core: int
    ready: int
    read_clock: int
    completion: int
    score: int
    ddr_cycles: int
    cache_cycles: int
    new_reads: list[tuple[int, int, int]]
    remote_writes: list[int]
    hit_bytes: int
    events: list[dict]

    def priority(self):
        return self.score, self.completion, self.ddr_cycles, self.core


@dataclass
class ConstructionState:
    assigned: dict
    finished: dict
    compute_clock: list[list[int]]
    read_clock: list[int]
    consumer_cores: list[set[int]]
    available: list[dict[int, int]]
    first_reads: FirstReadIndex

    @classmethod
    def create(cls, cores, tensors, capacity):
        return cls(
            {},
            {},
            [[0, 0] for _ in range(cores)],
            [0] * cores,
            [set() for _ in range(tensors)],
            [{} for _ in range(cores)],
            FirstReadIndex(capacity),
        )


def trial_placement(problem, graph, unit, core, state, hardware, cache_aware, weight):
    release = max(
        (
            state.finished[p] + (hardware.copy_wait if state.assigned[p] != core else 0)
            for p in graph.predecessors[unit]
        ),
        default=0,
    )
    clock = max(state.read_clock[core], release)
    ready = clock
    ddr = 0
    cached = 0
    hit_bytes = 0
    reads = []
    writes = []
    events = []
    for t in sorted(graph.inputs[unit], key=lambda j: problem.tensors[j].identifier):
        tensor = problem.tensors[t]
        producer = graph.owner[tensor.producer] if tensor.producer is not None else None
        local = producer is not None and state.assigned[producer] == core
        if local or core in state.consumer_cores[t]:
            ready = max(ready, state.available[core][t])
            events.append(dict(tensor=tensor.identifier, path="resident", available=state.available[core][t]))
            continue
        source_release = 0
        if producer is not None:
            source_release = state.finished[producer] + hardware.copy_cycles(tensor.size) + hardware.copy_wait
            writes.append(t)
        issue = max(clock, source_release)
        hit = cache_aware and state.first_reads.contains_at(t, issue)
        duration = hardware.copy_cycles(
            tensor.size, hardware.cache_bandwidth if hit else hardware.ddr_bandwidth
        )
        clock = issue + duration
        ready = max(ready, clock)
        reads.append((clock, t, tensor.size))
        if hit:
            cached += duration
            hit_bytes += tensor.size
        else:
            ddr += duration
        events.append(
            dict(
                tensor=tensor.identifier,
                path="L2" if hit else "DDR",
                source_release=source_release,
                issue=issue,
                finish=clock,
            )
        )
    finals = {t for v in graph.units[unit] for t in problem.outputs[v] if problem.tensors[t].final_output}
    ddr += sum(hardware.copy_cycles(problem.tensors[t].size) for t in writes)
    ddr += sum(hardware.copy_cycles(problem.tensors[t].size) for t in finals)
    completion = max(
        [ready + graph.span[unit]]
        + [
            max(state.compute_clock[core][p], ready) + work
            for p, work in enumerate(graph.work[unit])
            if work > 0
        ]
    )
    return Trial(
        core,
        ready,
        clock,
        completion,
        completion + weight * (ddr + cached),
        ddr,
        cached,
        reads,
        writes,
        hit_bytes,
        events,
    )


def price_assignment(problem, units, cores, hardware, cache_aware, weight):
    graph = UnitGraph.build(problem, units)
    rank = [0] * len(units)
    for u in reversed(graph.order):
        rank[u] = max(*graph.work[u], graph.span[u]) + max((rank[v] for v in graph.successors[u]), default=0)
    degree = [len(p) for p in graph.predecessors]
    ready = [(-rank[u], u) for u in graph.order if degree[u] == 0]
    heapq.heapify(ready)
    state = ConstructionState.create(cores, len(problem.tensors), hardware.cache_capacity)
    queues = [[] for _ in range(cores)]
    boundary = 0
    hit_bytes = 0
    predicted_hits = 0
    while ready:
        _, u = heapq.heappop(ready)
        winner = min(
            (
                trial_placement(problem, graph, u, c, state, hardware, cache_aware, weight)
                for c in range(cores)
            ),
            key=Trial.priority,
        )
        c = winner.core
        state.assigned[u] = c
        state.finished[u] = winner.completion
        queues[c].append(u)
        state.read_clock[c] = winner.read_clock
        for pipe, work in enumerate(graph.work[u]):
            if work > 0:
                state.compute_clock[c][pipe] = max(state.compute_clock[c][pipe], winner.ready) + work
        for finish, t, size in winner.new_reads:
            state.consumer_cores[t].add(c)
            state.available[c][t] = finish
            boundary += size
        for v in graph.units[u]:
            for t in problem.outputs[v]:
                state.available[c][t] = winner.completion
        finals = {t for v in graph.units[u] for t in problem.outputs[v] if problem.tensors[t].final_output}
        boundary += problem.byte_sum(finals) + problem.byte_sum(winner.remote_writes)
        state.first_reads.commit(winner.new_reads)
        hit_bytes += winner.hit_bytes
        predicted_hits += sum(event["path"] == "L2" for event in winner.events)
        for v in sorted(graph.successors[u]):
            degree[v] -= 1
            if degree[v] == 0:
                heapq.heappush(ready, (-rank[v], v))
    result = Placement(graph.units, tuple(tuple(q) for q in queues))
    return result, dict(
        cache_aware=cache_aware,
        weight=weight,
        boundary_bytes_before_spill=boundary,
        predicted_hit_bytes=hit_bytes,
        predicted_hit_count=predicted_hits,
    )
