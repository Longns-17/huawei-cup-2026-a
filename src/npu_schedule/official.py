"""The original evaluator is the only source of reported objective values."""

import sys
from dataclasses import dataclass

from .problem import RAW, ROOT

sys.path.insert(0, str(ROOT / "official/code"))
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from multicore_cut_evaluate_problem_2 import evaluate_scene_b, read_scene_b_config
from multicore_cut_evaluate_problem_3 import evaluate_problem_3, read_cache_config


@dataclass(frozen=True)
class Hardware:
    capacities: dict
    ddr_bandwidth: int
    task_remote_wait: int
    task_local_wait: int
    copy_wait: int
    cache_capacity: int
    cache_bandwidth: int

    @classmethod
    def read(cls):
        path = RAW / "config.txt"
        base = read_evaluation_config(path)
        a = read_scene_a_config(path)
        b = read_scene_b_config(path)
        cache = read_cache_config(path)
        return cls(
            base["capacity"],
            base["bandwidth"],
            a["task_cross_core_wait_cycles"],
            a["task_same_core_wait_cycles"],
            b["cross_core_copy_delay_cycles"],
            cache["cache_capacity_bytes"],
            cache["cache_bandwidth_bytes_per_cycle"],
        )

    def copy_cycles(self, size, bandwidth=None):
        rate = self.ddr_bandwidth if bandwidth is None else bandwidth
        return max(1, (size + rate - 1) // rate)


def evaluate(problem, placement, scene, hardware):
    submission = placement.submission(problem)
    common = dict(bandwidth=hardware.ddr_bandwidth, capacity=hardware.capacities)
    if scene == 1:
        return evaluate_scene_a(
            problem.raw,
            submission,
            **common,
            cross_core_wait=hardware.task_remote_wait,
            same_core_wait=hardware.task_local_wait,
        )
    if scene == 2:
        return evaluate_scene_b(problem.raw, submission, **common, cross_core_copy_delay=hardware.copy_wait)
    if scene == 3:
        return evaluate_problem_3(
            problem.raw,
            submission,
            **common,
            cross_core_copy_delay=hardware.copy_wait,
            cache_capacity_bytes=hardware.cache_capacity,
            cache_bandwidth_bytes_per_cycle=hardware.cache_bandwidth,
        )
    raise ValueError(f"Unknown scene: {scene}")


def objective(result):
    return result["makespan"], result["data_movement_bytes"]["added_copy_bytes"]
