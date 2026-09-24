import json
from collections import Counter

import numpy as np
import pytest

from npu_schedule.candidates import BUDGET, admitted_proposals
from npu_schedule.official import evaluate
from npu_schedule.problem import Placement, Problem
from npu_schedule.residency import Residency
from npu_schedule.search import solve_prefix
from npu_schedule.synthetic import make_problem
from npu_schedule.verification import check_slot


def test_serial_proxy_is_not_actual_concurrent_memory_peak(hardware):
    problem = make_problem([100, 100], ["M", "V"], [(0, [], 60000, True), (1, [], 60000, True)])
    placement = Placement(((0, 1),), ((0,),))
    proxy = Residency(problem, "cuda").compute([(0, 1)])
    actual = evaluate(problem, placement, 2, hardware)
    assert proxy[0]["UB"] == 60000
    assert actual["memory_peak_by_core"][0]["UB"] == 120000


def visit_reference(problem, sequences):
    """Sequential allocation/free oracle, independent of interval and prefix kernels."""
    results = []
    for sequence in sequences:
        remaining = Counter(t for v in sequence for t in problem.inputs[v])
        resident = set()
        peak = dict(L1=0, UB=0, DDR=0)
        for v in sequence:
            resident.update(problem.inputs[v] | problem.outputs[v])
            for region in peak:
                peak[region] = max(
                    peak[region],
                    sum(problem.tensors[t].size for t in resident if problem.tensors[t].region == region),
                )
            for t in problem.inputs[v]:
                remaining[t] -= 1
            resident.difference_update(t for t in list(resident) if remaining[t] == 0)
        results.append(peak)
    return results


@pytest.mark.parametrize("cores", [1, 2, 5])
def test_cuda_residency_matches_independent_visit_simulation(cores):
    problem = Problem.read("case_019")
    rng = np.random.default_rng(109)
    owners = rng.integers(0, cores, size=len(problem.identifiers))
    paths = [tuple(v for v in problem.order if owners[v] == c) for c in range(cores)]
    expected = visit_reference(problem, paths)
    assert Residency(problem, "cpu").compute(paths) == expected
    assert Residency(problem, "cuda").compute(paths) == expected


@pytest.mark.parametrize("scene", [1, 2, 3])
def test_candidate_admission_and_cpu_gpu_equivalence(scene, hardware):
    problem = Problem.read("case_019")
    observed = []
    for device in ("cpu", "cuda"):
        items = list(admitted_proposals(problem, 4, scene, hardware, Residency(problem, device)))
        admitted = [item for item in items if item["duplicate_of"] is None]
        assert len(admitted) <= BUDGET[scene]
        assert len({item["placement"].structure() for item in admitted}) == len(admitted)
        observed.append([(item["label"], item["placement"]) for item in admitted])
    assert observed[0] == observed[1]


@pytest.mark.parametrize("scene", [1, 2, 3])
def test_official_end_to_end_prefix_and_json_contract(scene, hardware, tmp_path):
    problem = Problem.read("case_019")
    folder = tmp_path / f"q{scene}"
    rows = solve_prefix(problem, scene, 3, hardware, Residency(problem, "cuda"), folder)
    assert rows[0]["makespan"] >= rows[1]["makespan"] >= rows[2]["makespan"]
    search = json.loads((folder / "search.json").read_text())
    assert search["actual_calls_completed"] <= 1 + 2 * BUDGET[scene]
    for k in range(1, 4):
        checked = check_slot(problem, folder / f"k{k}/metrics.json")
        assert checked["makespan"] == rows[k - 1]["makespan"]
        plan = json.loads((folder / f"k{k}/case_019_multicore_res.json").read_text())
        assert set(plan) == {"node_to_subgraph", "core_schedules"}
        assert len(plan["core_schedules"]) == k
        assert {int(i) for i in plan["node_to_subgraph"]} == set(problem.identifiers)
        assert sorted(u for q in plan["core_schedules"] for u in q) == sorted(
            set(plan["node_to_subgraph"].values())
        )
