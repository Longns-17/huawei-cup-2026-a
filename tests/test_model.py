from copy import deepcopy
from dataclasses import replace
from itertools import product

import pytest

from npu_schedule.construction import budget_coarsening, insertion, seed
from npu_schedule.metrics import boundary_bytes, summarize
from npu_schedule.official import evaluate
from npu_schedule.pricing import ConstructionState, FirstReadIndex, price_assignment, trial_placement
from npu_schedule.problem import Placement, Problem, UnitGraph, topological_order
from npu_schedule.residency import Residency
from npu_schedule.storage import fingerprint
from npu_schedule.synthetic import make_problem


def test_task_insertion_respects_both_neighbours():
    assert insertion([(0, 200, 0), (500, 800, 1)], 300, 150, 100) == (900, 2)
    assert insertion([(0, 200, 0), (550, 800, 1)], 300, 150, 100) == (300, 1)


@pytest.mark.parametrize("durations,optimal", [([2, 3, 5], 5), ([2, 2, 3], 4)])
@pytest.mark.parametrize("scene", [1, 2, 3])
def test_exhaustive_partition_exact_optimum(durations, optimal, scene, hardware):
    problem = make_problem(durations, ["M"] * 3, [])
    values = []
    for owners in product(range(2), repeat=3):
        units = []
        queues = [[], []]
        for c in range(2):
            group = tuple(v for v in range(3) if owners[v] == c)
            if group:
                queues[c].append(len(units))
                units.append(group)
        placement = Placement(tuple(units), tuple(tuple(q) for q in queues))
        values.append(evaluate(problem, placement, scene, hardware)["makespan"])
    assert min(values) == optimal


@pytest.mark.parametrize("queues,expected_q2", [(((0,), (1, 2)), 300), (((0,), (1,), (2,)), 420)])
def test_transfer_accounting_one_to_many(queues, expected_q2, hardware):
    problem = make_problem(
        [1] * 4, ["V"] * 4, [(0, [1, 2, 3], 60, False)] + [(v, [], 60, True) for v in (1, 2, 3)]
    )
    placement = Placement(((0,), (1, 2), (3,)), queues)
    for scene in (1, 2, 3):
        expected = 360 if scene == 1 else expected_q2
        assert boundary_bytes(problem, placement, scene) == expected
        result = evaluate(problem, placement, scene, hardware)
        assert result["data_movement_bytes"]["scheduled_copy_bytes"] == expected
        summarize(problem, placement, scene, hardware, result)


def test_first_read_at_issue_and_fifo_without_refresh():
    cache = FirstReadIndex(30000)
    cache.commit([(10, 0, 600)])
    assert not cache.contains_at(0, 9)
    assert cache.contains_at(0, 10)
    cache.commit([(15, 0, 600), (20, 1, 30000), (21, 2, 31000)])
    assert not cache.contains_at(0, 20)
    assert cache.contains_at(1, 20)
    assert not cache.contains_at(2, 99)
    assert cache.earliest[0] == 10
    cache.commit([(8, 0, 600)])
    assert cache.contains_at(0, 9)
    wider = FirstReadIndex(40000)
    wider.commit([(10, 0, 600), (20, 1, 30000)])
    assert wider.contains_at(0, 20)


def test_core_trial_is_read_only_and_uses_source_release(hardware):
    # IDEA 5.4 integer example, with the DAG span explicitly fixed to the stated 25.
    problem = make_problem(
        [1, 30, 12],
        ["M", "M", "V"],
        [(None, [1], 600, False), (0, [1, 2], 1200, False), (None, [2], 300, False)],
    )
    graph = UnitGraph.build(problem, ((0,), (1, 2)))
    graph.span[1] = 25
    ids = {t.identifier: i for i, t in enumerate(problem.tensors)}
    t0, t1, t2 = (ids[1000], ids[1001], ids[1002])
    state = ConstructionState.create(2, len(problem.tensors), hardware.cache_capacity)
    state.assigned[0] = 0
    state.finished[0] = 100
    state.compute_clock[1] = [605, 590]
    state.read_clock[1] = 590
    state.consumer_cores[t2].add(1)
    state.available[1][t2] = 580
    state.first_reads.commit([(10, t0, 600), (630, t1, 1200)])
    before = deepcopy(state.__dict__)
    ddr = trial_placement(problem, graph, 1, 1, state, hardware, False, 1)
    cached = trial_placement(problem, graph, 1, 1, state, hardware, True, 1)
    assert [(e["issue"], e["finish"]) for e in ddr.events if e["path"] == "DDR"] == [(600, 610), (620, 640)]
    assert cached.events[0]["finish"] == 603
    assert cached.events[1]["path"] == "DDR"  # 630 completion is after the 620 issue.
    assert ddr.completion == cached.completion == 670
    assert (ddr.ddr_cycles, cached.ddr_cycles, cached.cache_cycles) == (50, 40, 3)
    assert state.assigned == before["assigned"] and state.read_clock == before["read_clock"]
    assert state.first_reads.events == before["first_reads"].events


def test_cache_stagger_has_causal_effect(hardware):
    problem = make_problem(
        [1] * 4,
        ["V"] * 4,
        [(None, [0, 1], 600, False), (None, [2, 3], 30000, False)] + [(v, [], 60, True) for v in range(4)],
    )
    together = Placement(((0,), (1,), (2,), (3,)), ((0, 2), (1, 3)))
    staggered = Placement(together.units, ((0, 2), (3, 1)))
    assert [evaluate(problem, p, 2, hardware)["makespan"] for p in (together, staggered)] == [1025, 1025]
    assert [evaluate(problem, p, 3, hardware)["makespan"] for p in (together, staggered)] == [1025, 1015]


def test_structural_dedup_is_weaker_than_official_reuse():
    problem = make_problem([2, 3], ["M", "V"], [])
    a = Placement(((0,), (1,)), ((0, 1),))
    renamed = Placement(((1,), (0,)), ((1, 0),))
    padded = a.add_idle_cores(5)
    assert a.structure() == renamed.structure() == padded.structure()
    assert fingerprint(a.submission(problem)) != fingerprint(renamed.submission(problem))
    assert fingerprint(a.submission(problem)) == fingerprint(padded.submission(problem))


@pytest.mark.parametrize("case", ["case_019", "case_078"])
def test_equal_bandwidth_gives_identical_pricing_assignment(case, hardware):
    problem = Problem.read(case)
    scan = Residency(problem, "cpu")
    hardware = replace(hardware, cache_bandwidth=hardware.ddr_bandwidth)
    anchor, _, _ = seed(problem, 4, 2, 2, True, hardware, scan)
    a, diag_a = price_assignment(problem, anchor.units, 4, hardware, False, 1)
    b, diag_b = price_assignment(problem, anchor.units, 4, hardware, True, 1)
    assert a == b
    assert diag_a["boundary_bytes_before_spill"] == boundary_bytes(problem, a, 2)
    assert diag_b["boundary_bytes_before_spill"] == boundary_bytes(problem, b, 3)


@pytest.mark.parametrize("cores", [2, 3, 4, 5])
@pytest.mark.parametrize("percentage", [5, 10])
@pytest.mark.parametrize("topology", ["kahn", "dfs"])
def test_budget_coarsening_legal_and_within_task_cap(cores, percentage, topology, hardware):
    problem = Problem.read("case_019")
    placement, diagnostic = budget_coarsening(
        problem, cores, percentage, hardware, Residency(problem, "cpu"), topology
    )
    assert max(map(len, placement.queues)) <= diagnostic["task_limit"]
    assert sorted(v for part in placement.units for v in part) == list(range(len(problem.identifiers)))
    graph = UnitGraph.build(problem, placement.units)
    for queue in placement.queues:
        for u, v in zip(queue, queue[1:]):
            graph.successors[u].add(v)
            graph.predecessors[v].add(u)
    assert len(topological_order(graph.predecessors, graph.successors)) == len(placement.units)
