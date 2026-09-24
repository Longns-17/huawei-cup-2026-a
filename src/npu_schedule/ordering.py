"""Reorder fixed assignments within legal ready bands."""

import heapq
import math
from collections import Counter, defaultdict

from .problem import Placement, UnitGraph


def band_graph(problem, placement, width):
    graph = UnitGraph.build(problem, placement.units)
    for queue in placement.queues:
        bands = [queue[start : start + width] for start in range(0, len(queue), width)]
        for left, right in zip(bands, bands[1:]):
            for a in left:
                for b in right:
                    graph.successors[a].add(b)
                    graph.predecessors[b].add(a)
    return graph


def release_priority(problem, placement, width):
    graph = band_graph(problem, placement, width)
    owners = placement.owners()
    indices = {u: i for q in placement.queues for i, u in enumerate(q)}
    all_inputs = [set().union(*(problem.inputs[v] for v in part)) for part in graph.units]
    touches = [
        all_inputs[u] | set().union(*(problem.outputs[v] for v in part)) for u, part in enumerate(graph.units)
    ]
    remaining = Counter((owners[u], t) for u, ts in enumerate(all_inputs) for t in ts)
    resident = [set() for _ in placement.queues]
    output = [[] for _ in placement.queues]
    degree = [len(p) for p in graph.predecessors]
    ready = {u for u, d in enumerate(degree) if d == 0}
    while ready:

        def priority(u):
            c = owners[u]
            allocation = problem.byte_sum(touches[u] - resident[c])
            release = problem.byte_sum(t for t in all_inputs[u] if remaining[c, t] == 1)
            return allocation - release, indices[u], c, u

        u = min(ready, key=priority)
        ready.remove(u)
        c = owners[u]
        output[c].append(u)
        resident[c].update(touches[u])
        for t in all_inputs[u]:
            remaining[c, t] -= 1
        resident[c].difference_update(t for t in touches[u] if remaining[c, t] == 0)
        for v in graph.successors[u]:
            degree[v] -= 1
            if degree[v] == 0:
                ready.add(v)
    return Placement(graph.units, tuple(tuple(q) for q in output)), dict(window=width)


def cache_stagger(problem, placement, width, capacity):
    graph = band_graph(problem, placement, width)
    owners = placement.owners()
    external = []
    reading_cores = defaultdict(set)
    for u, part in enumerate(graph.units):
        ts = {
            t
            for v in part
            for t in problem.inputs[v]
            if problem.tensors[t].producer is None
            or owners[graph.owner[problem.tensors[t].producer]] != owners[u]
        }
        external.append(ts)
        for t in ts:
            reading_cores[t].add(owners[u])
    shared = sorted(
        (t for t, cs in reading_cores.items() if len(cs) > 1 and 0 < problem.tensors[t].size <= capacity),
        key=lambda t: (-(len(reading_cores[t]) - 1) * problem.tensors[t].size, problem.tensors[t].identifier),
    )
    if not shared:
        return placement, dict(window=width, shared_tensors=0)
    rank = {t: i for i, t in enumerate(shared)}
    stride = math.ceil(len(shared) / len(placement.queues))
    indices = {u: i for q in placement.queues for i, u in enumerate(q)}
    priority = [
        (
            min(((rank[t] - owners[u] * stride) % len(shared) for t in ts if t in rank), default=len(shared)),
            indices[u],
            owners[u],
            u,
        )
        for u, ts in enumerate(external)
    ]
    degree = [len(p) for p in graph.predecessors]
    ready = [priority[u] for u, d in enumerate(degree) if d == 0]
    heapq.heapify(ready)
    output = [[] for _ in placement.queues]
    while ready:
        *_, u = heapq.heappop(ready)
        output[owners[u]].append(u)
        for v in graph.successors[u]:
            degree[v] -= 1
            if degree[v] == 0:
                heapq.heappush(ready, priority[v])
    return Placement(graph.units, tuple(tuple(q) for q in output)), dict(
        window=width, shared_tensors=len(shared)
    )
