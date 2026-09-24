"""Cold-start, synchronous prefix search with fixed unique candidate budgets."""

from time import perf_counter

from .candidates import BUDGET, admitted_proposals
from .metrics import summarize
from .official import evaluate, objective
from .problem import Placement
from .storage import fingerprint, save_raw, write_json


class Evaluations:
    def __init__(self, problem, scene, hardware, directory):
        self.problem = problem
        self.scene = scene
        self.hardware = hardware
        self.directory = directory
        directory.mkdir()
        self.cache = {}
        self.calls = 0
        self.seconds = 0.0

    def score(self, placement):
        submission = placement.submission(self.problem)
        key = fingerprint(submission)
        reused = key in self.cache
        if not reused:
            self.calls += 1
            started = perf_counter()
            result = evaluate(self.problem, placement, self.scene, self.hardware)
            seconds = perf_counter() - started
            metrics = summarize(self.problem, placement, self.scene, self.hardware, result)
            self.seconds += seconds
            stem = f"{self.calls:03d}_{key[:12]}"
            write_json(self.directory / f"{stem}.plan.json", submission)
            save_raw(self.directory / f"{stem}.official.json.gz", result)
            self.cache[key] = dict(
                metrics=metrics,
                objective=objective(result),
                evaluation_seconds=seconds,
                official_result=f"{self.directory.name}/{stem}.official.json.gz",
                submission=f"{self.directory.name}/{stem}.plan.json",
            )
        return self.cache[key], reused


def solve_prefix(problem, scene, cores, hardware, residency, directory):
    directory.mkdir(parents=True)
    evaluations = Evaluations(problem, scene, hardware, directory / "evaluations")
    baseline = Placement((tuple(problem.order),), ((0,),))
    started = perf_counter()
    base_record, _ = evaluations.score(baseline)
    incumbent = baseline
    incumbent_record = base_record
    rows = []
    events = []
    for k in range(1, cores + 1):
        calls_before = evaluations.calls
        generation_seconds = 0.0
        admitted = 0
        if k == 1:
            winner = "whole_graph"
            event = dict(cores=1, label=winner, status="evaluated", evaluation=base_record["official_result"])
            events.append(event)
        else:
            incumbent = incumbent.add_idle_cores(k)
            winner = (
                "whole_graph" if incumbent_record["objective"] == base_record["objective"] else "inherited"
            )
            events.extend(
                [
                    dict(cores=k, label=label, status="reused", evaluation=record["official_result"])
                    for label, record in (("whole_graph", base_record), ("inherited", incumbent_record))
                ]
            )
            stream = admitted_proposals(problem, k, scene, hardware, residency)
            tick = perf_counter()
            for proposal in stream:
                elapsed = perf_counter() - tick
                generation_seconds += elapsed
                label = proposal["label"]
                placement = proposal["placement"]
                event = dict(
                    cores=k, label=label, generation_seconds=elapsed, diagnostic=proposal["diagnostic"]
                )
                if proposal["duplicate_of"] is not None:
                    event.update(status="duplicate", duplicate_of=proposal["duplicate_of"])
                else:
                    admitted += 1
                    record, reused = evaluations.score(placement)
                    event.update(
                        status="reused" if reused else "evaluated",
                        evaluation=record["official_result"],
                        submission=record["submission"],
                        makespan=record["metrics"]["makespan"],
                        added_copy_bytes=record["metrics"]["added_copy_bytes"],
                    )
                    if record["objective"] < incumbent_record["objective"]:
                        incumbent = placement
                        incumbent_record = record
                        winner = label
                events.append(event)
                tick = perf_counter()
        output = directory / f"k{k}"
        output.mkdir()
        write_json(output / f"{problem.name}_multicore_res.json", incumbent.submission(problem))
        # Recalculate k-dependent bounds without re-running the same official submission.
        metrics = dict(incumbent_record["metrics"])
        metrics["global_lower_bound"] = max(
            problem.compute_lower_bound(k), metrics["original_copy_bytes"] / hardware.ddr_bandwidth
        )
        metrics["gap_to_global_bound"] = metrics["makespan"] / metrics["global_lower_bound"] - 1
        metrics["tasks_per_core"] = [len(q) if scene == 1 else int(bool(q)) for q in incumbent.queues]
        metrics["memory_peak_by_core"] = dict(metrics["memory_peak_by_core"])
        evaluated_cores = len(metrics["memory_peak_by_core"])
        for idle in range(evaluated_cores, k):
            metrics["memory_peak_by_core"][idle] = {region: 0 for region in hardware.capacities}
        metrics.update(
            case=problem.name,
            scene=scene,
            cores=k,
            winner=winner,
            complete=True,
            official_result="../" + incumbent_record["official_result"],
            admitted_candidates=admitted,
            candidate_budget=BUDGET[scene],
            new_official_calls=1 if k == 1 else evaluations.calls - calls_before,
            cumulative_official_calls=evaluations.calls,
            generation_seconds=generation_seconds,
        )
        write_json(output / "metrics.json", metrics)
        rows.append(metrics)
        write_json(directory / "candidate_log.json", events)
        print(
            f"{problem.name} Q{scene} k={k}: {metrics['makespan']} cycles; {winner}; official calls={evaluations.calls}",
            flush=True,
        )
    assert evaluations.calls <= 1 + (cores - 1) * BUDGET[scene]
    assert all(rows[i]["makespan"] <= rows[i - 1]["makespan"] for i in range(1, len(rows)))
    write_json(
        directory / "search.json",
        dict(
            case=problem.name,
            scene=scene,
            complete=True,
            actual_calls_started=evaluations.calls,
            actual_calls_completed=evaluations.calls,
            official_seconds=evaluations.seconds,
            wall_seconds=perf_counter() - started,
            result_slots=cores,
            structural_duplicates=sum(e["status"] == "duplicate" for e in events),
        ),
    )
    return rows
