"""Independently count transfer domains and compute the IDEA lower bounds."""

from .problem import UnitGraph


def boundary_bytes(problem, placement, scene):
    graph = UnitGraph.build(problem, placement.units)
    unit_core = placement.owners()
    domain = graph.owner if scene == 1 else {v: unit_core[u] for v, u in graph.owner.items()}
    total = 0
    for tensor in problem.tensors:
        receivers = {domain[v] for v in tensor.consumers}
        if tensor.producer is None:
            total += tensor.size * len(receivers)
        else:
            remote = receivers - {domain[tensor.producer]}
            copies = (
                len(remote) + int(bool(remote) or tensor.final_output)
                if scene == 1
                else 2 * len(remote) + int(tensor.final_output)
            )
            total += tensor.size * copies
    return total


def summarize(problem, placement, scene, hardware, result):
    movement = result["data_movement_bytes"]
    original = movement["original_graph_copy_bytes"]
    scheduled = original + movement["added_copy_bytes"]
    hits = result["cache_stats"]["hit_bytes"] if scene == 3 else 0
    hit_rate = result["cache_stats"]["hit_rate"] if scene == 3 else 0.0
    global_bound = max(problem.compute_lower_bound(len(placement.queues)), original / hardware.ddr_bandwidth)
    core_work = max(
        max(problem.work(v for u in queue for v in placement.units[u])) for queue in placement.queues
    )
    task_bound = 0
    if scene == 1:
        graph = UnitGraph.build(problem, placement.units)
        task_work = [
            max(
                *graph.work[u],
                graph.span[u],
                sum(hardware.copy_cycles(problem.tensors[t].size) for t in graph.inputs[u])
                + sum(hardware.copy_cycles(problem.tensors[t].size) for t in graph.outputs[u]),
            )
            for u in range(len(graph.units))
        ]
        task_bound = max(
            sum(task_work[u] for u in q) + hardware.task_local_wait * max(0, len(q) - 1)
            for q in placement.queues
        )
    plan_bound = max(
        global_bound,
        core_work,
        (scheduled - hits) / hardware.ddr_bandwidth,
        hits / hardware.cache_bandwidth,
        task_bound,
    )
    counted = boundary_bytes(problem, placement, scene)
    # These are scientific invariants, evaluated on every reported result.
    assert counted + movement["spill_added_copy_bytes"] == scheduled, (problem.name, scene, counted, movement)
    assert plan_bound <= result["makespan"] + 1e-8, (problem.name, scene, plan_bound, result["makespan"])
    return dict(
        makespan=result["makespan"],
        added_copy_bytes=movement["added_copy_bytes"],
        original_copy_bytes=original,
        boundary_bytes_before_spill=counted,
        spill_copy_bytes=movement["spill_added_copy_bytes"],
        scheduled_copy_bytes=scheduled,
        ddr_service_bytes=scheduled - hits,
        cache_hit_bytes=hits,
        cache_byte_hit_rate=hit_rate,
        global_lower_bound=global_bound,
        plan_lower_bound=plan_bound,
        gap_to_global_bound=result["makespan"] / global_bound - 1,
        subgraph_count=len(placement.units),
        tasks_per_core=[len(q) if scene == 1 else int(bool(q)) for q in placement.queues],
        memory_peak_by_core=result["memory_peak_by_core"],
    )
