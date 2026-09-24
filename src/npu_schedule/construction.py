"""Component/chain seeds, HEFT insertion, and scenario-A coarsening."""

import heapq

from .calendar import TaskCalendar
from .problem import Placement, UnitGraph


def chain_partition(problem, component):
    remaining = set(component)
    chains = []
    for start in component:
        if start in remaining:
            chain = [start]
            remaining.remove(start)
            current = start
            while len(problem.successors[current]) == 1:
                following = next(iter(problem.successors[current]))
                if len(problem.predecessors[following]) != 1:
                    break
                chain.append(following)
                remaining.remove(following)
                current = following
            chains.append(tuple(chain))
    return chains


def insertion(calendar, release, duration, gap):
    """Earliest idle interval with both neighbouring Task gaps accounted for."""
    start = release
    for position, (left, right, unit) in enumerate(calendar):
        if start + duration + gap <= left:
            return start, position
        start = max(start, right + gap)
    return start, len(calendar)


def heft(problem, units, cores, scene, hardware, task_limit=None):
    graph = UnitGraph.build(problem, units)
    # One proxy tick is 1 / DDR-bandwidth cycles; finite clocks stay exact integers.
    scale = hardware.ddr_bandwidth
    duration = []
    for u in range(len(units)):
        compute = max(*graph.work[u], graph.span[u])
        if scene == 1:
            transfer = scale * (
                sum(hardware.copy_cycles(problem.tensors[t].size) for t in graph.inputs[u])
                + sum(hardware.copy_cycles(problem.tensors[t].size) for t in graph.outputs[u])
            )
        else:
            transfer = problem.byte_sum(graph.inputs[u] | graph.outputs[u])
        duration.append(max(compute * scale, transfer))
    communication = {
        edge: (
            hardware.task_remote_wait * scale
            if scene == 1
            else hardware.copy_wait * scale + 2 * problem.byte_sum(ts)
        )
        for edge, ts in graph.edge_tensors.items()
    }
    rank = [0] * len(units)
    for u in reversed(graph.order):
        rank[u] = duration[u] * cores + max(
            (rank[v] + communication[u, v] * (cores - 1) for v in graph.successors[u]), default=0
        )
    pending = [len(p) for p in graph.predecessors]
    ready = [(-rank[u], u) for u in graph.order if pending[u] == 0]
    heapq.heapify(ready)
    gap = hardware.task_local_wait * scale if scene == 1 else 0
    calendars = [TaskCalendar(gap) for _ in range(cores)]
    assigned = {}
    finished = {}
    while ready:
        _, u = heapq.heappop(ready)
        choices = []
        for c in range(cores):
            if task_limit is not None and len(calendars[c].entries) == task_limit:
                continue
            release = max(
                (
                    finished[p] + (gap if assigned[p] == c else communication[p, u])
                    for p in graph.predecessors[u]
                ),
                default=0,
            )
            start, interval = calendars[c].earliest(release, duration[u])
            choices.append((start + duration[u], start, c, interval))
        end, start, c, interval = min(choices)
        calendars[c].commit(start, end, u, interval)
        assigned[u] = c
        finished[u] = end
        for v in sorted(graph.successors[u]):
            pending[v] -= 1
            if pending[v] == 0:
                heapq.heappush(ready, (-rank[v], v))
    return Placement(graph.units, tuple(calendar.order() for calendar in calendars))


def seed(problem, cores, scene, alpha, repair, hardware, residency):
    threshold = max(problem.work(problem.order)) / (alpha * cores)
    units = []
    intact = set()
    for component in problem.components:
        if max(problem.work(component)) <= threshold:
            intact.add(len(units))
            units.append(component)
        else:
            units.extend(chain_partition(problem, component))
    placement = heft(problem, units, cores, scene, hardware)
    details = dict(alpha=alpha, repaired_components=0)
    if repair:
        paths = [tuple(v for u in queue for v in units[u]) for queue in placement.queues]
        peaks = residency.compute(paths)
        opened = {
            u
            for c, queue in enumerate(placement.queues)
            if any(peaks[c][r] > limit for r, limit in hardware.capacities.items())
            for u in queue
            if u in intact
        }
        details["serial_visit_peaks"] = peaks
        details["repaired_components"] = len(opened)
        refined = []
        for u, part in enumerate(units):
            refined.extend(chain_partition(problem, part) if u in opened else [part])
        if opened:
            placement = heft(problem, refined, cores, scene, hardware)
    return placement, details, intact


def combine_intact(placement, intact):
    units = []
    queues = []
    for queue in placement.queues:
        complete = [u for u in queue if u in intact]
        combined = tuple(v for u in complete for v in placement.units[u])
        emitted = False
        new_queue = []
        for u in queue:
            if u in intact:
                if emitted:
                    continue
                nodes = combined
                emitted = True
            else:
                nodes = placement.units[u]
            new_queue.append(len(units))
            units.append(nodes)
        queues.append(tuple(new_queue))
    return Placement(tuple(units), tuple(queues))


def boundary_bytes(problem, nodes):
    members = set(nodes)
    incoming = {t for v in nodes for t in problem.inputs[v] if problem.tensors[t].producer not in members}
    outgoing = {
        t
        for v in nodes
        for t in problem.outputs[v]
        if problem.tensors[t].final_output or problem.tensors[t].consumers - members
    }
    return problem.byte_sum(incoming) + problem.byte_sum(outgoing)


def merge_adjacent(problem, placement, hardware):
    units = {u: part for u, part in enumerate(placement.units)}
    queues = [list(q) for q in placement.queues]
    quotient = UnitGraph.build(problem, placement.units)
    succ = {u: set(s) for u, s in enumerate(quotient.successors)}
    pred = {u: set(p) for u, p in enumerate(quotient.predecessors)}
    owners = placement.owners()
    work = {u: problem.work(part) for u, part in units.items()}
    traffic = {u: boundary_bytes(problem, part) for u, part in units.items()}
    for q in queues:
        for a, b in zip(q, q[1:]):
            succ[a].add(b)
            pred[b].add(a)
    merges = 0
    for c, q in enumerate(queues):
        i = 0
        while i + 1 < len(q):
            a, b = q[i], q[i + 1]
            combined_work = tuple(x + y for x, y in zip(work[a], work[b]))
            if max(combined_work) > 1000:
                i += 1
                continue
            joined = units[a] + units[b]
            new_traffic = boundary_bytes(problem, joined)
            penalty = max(work[b]) if any(owners[v] != c for v in succ[a] - {b}) else 0
            saving = (
                hardware.task_local_wait + (traffic[a] + traffic[b] - new_traffic) / hardware.ddr_bandwidth
            )
            if saving <= penalty:
                i += 1
                continue
            reachable = set(succ[a] - {b})
            frontier = list(reachable)
            while frontier:
                v = frontier.pop()
                for w in succ[v] - reachable:
                    reachable.add(w)
                    frontier.append(w)
            if b in reachable:
                i += 1
                continue
            incoming = (pred[a] | pred[b]) - {a, b}
            outgoing = (succ[a] | succ[b]) - {a, b}
            for v in incoming:
                succ[v].difference_update((a, b))
                succ[v].add(a)
            for v in outgoing:
                pred[v].difference_update((a, b))
                pred[v].add(a)
            pred[a] = incoming
            succ[a] = outgoing
            pred[b] = set()
            succ[b] = set()
            units[a] = joined
            work[a] = combined_work
            traffic[a] = new_traffic
            q.pop(i + 1)
            merges += 1
    remaining = [u for q in queues for u in q]
    remap = {u: i for i, u in enumerate(remaining)}
    output = Placement(tuple(units[u] for u in remaining), tuple(tuple(remap[u] for u in q) for q in queues))
    return output, merges


def local_coarsening(problem, cores, hardware, residency):
    fine, _, intact = seed(problem, cores, 1, 2, False, hardware, residency)
    folded = combine_intact(fine, intact)
    merged, count = merge_adjacent(problem, folded, hardware)
    result = heft(problem, merged.units, cores, 1, hardware) if count else merged
    return result, dict(adjacent_merges=count, units_before=len(fine.units), units_after=len(result.units))


def budget_coarsening(problem, cores, percentage, hardware, residency, topology="kahn"):
    initial, _, _ = seed(problem, cores, 1, 2, False, hardware, residency)
    quotient = UnitGraph.build(problem, initial.units)
    order = quotient.order
    if topology == "dfs":
        visited = set()
        post = []
        for start in range(len(initial.units)):
            stack = [(start, False)]
            while stack:
                u, expanded = stack.pop()
                if expanded:
                    post.append(u)
                elif u not in visited:
                    visited.add(u)
                    stack.append((u, True))
                    stack.extend((v, False) for v in sorted(quotient.successors[u], reverse=True))
        order = list(reversed(post))
    bound_numerator = max(max(problem.work(problem.order)), problem.longest_path(problem.order) * cores)
    per_core = 1 + bound_numerator * percentage // (100 * cores * hardware.task_local_wait)
    target = min(len(initial.units), cores * per_core)
    weights = [sum(problem.duration[v] for v in part) for part in initial.units]
    total = sum(weights)
    cumulative = 0
    previous = -1
    units = []
    for u in order:
        section = min(target - 1, cumulative * target // total)
        if section != previous:
            units.append([])
            previous = section
        units[-1].extend(initial.units[u])
        cumulative += weights[u]
    placement = heft(problem, units, cores, 1, hardware, task_limit=per_core)
    return placement, dict(
        percentage=percentage,
        task_limit=per_core,
        target_units=target,
        actual_units=len(units),
        topology=topology,
    )
