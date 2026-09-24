"""Translate the official Op/Tensor input into a dense compute DAG."""

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def topological_order(predecessors, successors):
    degree = [len(p) for p in predecessors]
    frontier = [v for v, d in enumerate(degree) if d == 0]
    heapq.heapify(frontier)
    sequence = []
    while frontier:
        v = heapq.heappop(frontier)
        sequence.append(v)
        for w in sorted(successors[v]):
            degree[w] -= 1
            if degree[w] == 0:
                heapq.heappush(frontier, w)
    if len(sequence) != len(degree):
        raise ValueError("The dependency graph contains a cycle")
    return sequence


@dataclass(frozen=True)
class Tensor:
    identifier: int
    size: int
    region: str
    producer: int | None
    consumers: frozenset[int]
    final_output: bool


@dataclass
class Problem:
    name: str
    raw: dict
    identifiers: tuple[int, ...]
    duration: tuple[int, ...]
    pipe: tuple[int, ...]
    tensors: tuple[Tensor, ...]
    inputs: tuple[frozenset[int], ...]
    outputs: tuple[frozenset[int], ...]
    predecessors: list[set[int]]
    successors: list[set[int]]
    order: list[int]
    position: dict[int, int]
    components: list[tuple[int, ...]]

    @classmethod
    def read(cls, name):
        return cls.from_dict(name, json.loads((RAW / f"{name}.json").read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, name, raw):
        raw_ops = {o["id"]: o for o in raw["ops"]}
        identifiers = tuple(sorted(o["id"] for o in raw["ops"] if o["op"] not in ("COPY_IN", "COPY_OUT")))
        index = {identifier: i for i, identifier in enumerate(identifiers)}
        raw_tensors = sorted(raw["tensors"], key=lambda t: t["id"])
        tensor_index = {t["id"]: i for i, t in enumerate(raw_tensors)}
        producers, consumers = defaultdict(list), defaultdict(list)
        for edge in raw["edges"]:
            u, v = edge["source"], edge["target"]
            if u in raw_ops:
                producers[v].append(u)
            else:
                consumers[u].append(v)
        tensors = []
        inputs = [set() for _ in identifiers]
        outputs = [set() for _ in identifiers]
        pred = [set() for _ in identifiers]
        succ = [set() for _ in identifiers]
        for t in raw_tensors:
            tid = t["id"]
            compute_producers = [index[u] for u in producers[tid] if u in index]
            producer = compute_producers[0] if compute_producers else None
            readers = frozenset(index[u] for u in consumers[tid] if u in index)
            final = producer is not None and (
                not readers or any(raw_ops[u]["op"] == "COPY_OUT" for u in consumers[tid])
            )
            tensor = Tensor(tid, t["size"], t["pos"], producer, readers, final)
            tensors.append(tensor)
            j = tensor_index[tid]
            for v in readers:
                inputs[v].add(j)
            if producer is not None:
                outputs[producer].add(j)
                for v in readers:
                    pred[v].add(producer)
                    succ[producer].add(v)
        order = topological_order(pred, succ)
        position = {v: i for i, v in enumerate(order)}
        unvisited = set(range(len(identifiers)))
        components = []
        while unvisited:
            root = min(unvisited)
            stack, part = [root], []
            unvisited.remove(root)
            while stack:
                v = stack.pop()
                part.append(v)
                for w in sorted((pred[v] | succ[v]) & unvisited):
                    unvisited.remove(w)
                    stack.append(w)
            components.append(tuple(sorted(part, key=position.__getitem__)))
        return cls(
            name,
            raw,
            identifiers,
            tuple(max(1, raw_ops[u]["cycles"]) for u in identifiers),
            tuple({"PIPE_M": 0, "PIPE_V": 1}[raw_ops[u]["pipe"]] for u in identifiers),
            tuple(tensors),
            tuple(map(frozenset, inputs)),
            tuple(map(frozenset, outputs)),
            pred,
            succ,
            order,
            position,
            components,
        )

    def work(self, nodes):
        values = [0, 0]
        for v in nodes:
            values[self.pipe[v]] += self.duration[v]
        return tuple(values)

    def longest_path(self, nodes):
        members = set(nodes)
        length = {}
        for v in sorted(nodes, key=self.position.__getitem__):
            length[v] = self.duration[v] + max((length[u] for u in self.predecessors[v] & members), default=0)
        return max(length.values(), default=0)

    def compute_lower_bound(self, cores):
        return max(max(self.work(self.order)) / cores, self.longest_path(self.order))

    def byte_sum(self, tensors):
        return sum(self.tensors[t].size for t in tensors)


@dataclass(frozen=True)
class Placement:
    units: tuple[tuple[int, ...], ...]
    queues: tuple[tuple[int, ...], ...]

    def owners(self):
        return {unit: core for core, queue in enumerate(self.queues) for unit in queue}

    def add_idle_cores(self, total):
        return Placement(self.units, self.queues + ((),) * (total - len(self.queues)))

    def submission(self, problem):
        return {
            "node_to_subgraph": {
                str(problem.identifiers[v]): u for u, part in enumerate(self.units) for v in part
            },
            "core_schedules": [list(q) for q in self.queues],
        }

    def structure(self):
        queues = list(self.queues)
        while len(queues) > 1 and queues[-1] == ():
            queues.pop()
        return tuple(tuple(tuple(sorted(self.units[u])) for u in q) for q in queues)


@dataclass
class UnitGraph:
    units: tuple[tuple[int, ...], ...]
    owner: dict[int, int]
    predecessors: list[set[int]]
    successors: list[set[int]]
    edge_tensors: dict[tuple[int, int], set[int]]
    order: list[int]
    inputs: list[frozenset[int]]
    outputs: list[frozenset[int]]
    work: list[tuple[int, int]]
    span: list[int]

    @classmethod
    def build(cls, problem, units):
        units = tuple(tuple(u) for u in units)
        owner = {v: i for i, part in enumerate(units) for v in part}
        pred, succ = [set() for _ in units], [set() for _ in units]
        boundary = defaultdict(set)
        inputs, outputs = [], []
        for i, part in enumerate(units):
            members = set(part)
            incoming = set().union(*(problem.inputs[v] for v in part))
            outgoing = set().union(*(problem.outputs[v] for v in part))
            inputs.append(frozenset(t for t in incoming if problem.tensors[t].producer not in members))
            outputs.append(
                frozenset(
                    t
                    for t in outgoing
                    if problem.tensors[t].final_output or problem.tensors[t].consumers - members
                )
            )
        for t, tensor in enumerate(problem.tensors):
            if tensor.producer is not None:
                a = owner[tensor.producer]
                for b in {owner[v] for v in tensor.consumers} - {a}:
                    pred[b].add(a)
                    succ[a].add(b)
                    boundary[a, b].add(t)
        return cls(
            units,
            owner,
            pred,
            succ,
            dict(boundary),
            topological_order(pred, succ),
            inputs,
            outputs,
            [problem.work(part) for part in units],
            [problem.longest_path(part) for part in units],
        )
