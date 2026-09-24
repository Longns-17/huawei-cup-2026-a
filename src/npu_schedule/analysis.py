"""Separately charged controls; never overwrite the original search results."""

import os
from dataclasses import replace

from .construction import budget_coarsening, seed
from .experiment import create_run
from .metrics import summarize
from .official import Hardware, objective
from .pricing import price_assignment
from .problem import ROOT, Placement, Problem
from .residency import Residency
from .search import Evaluations
from .storage import fingerprint, load_placement, load_raw, read_json, write_csv, write_json


def populate_from_main(evaluations, source):
    """Reuse only explicitly supplied, same-input official observations of this new project."""
    for path in sorted(source.glob("*.plan.json")):
        placement = load_placement(evaluations.problem, path)
        result_path = path.with_name(path.name.replace(".plan.json", ".official.json.gz"))
        result = load_raw(result_path)
        metrics = summarize(evaluations.problem, placement, evaluations.scene, evaluations.hardware, result)
        evaluations.cache[fingerprint(placement.submission(evaluations.problem))] = dict(
            metrics=metrics,
            objective=objective(result),
            evaluation_seconds=0.0,
            official_result=str(result_path),
            submission=str(path),
        )


def paired(source, name):
    hardware = Hardware.read()
    main = read_json(source / "manifest.json")
    directory = create_run(name, "equal_pool_and_same_submission", dict(source=str(source)))
    current = read_json(directory / "manifest.json")
    official = {
        k: v
        for k, v in current["source_sha256"].items()
        if k.startswith("official") or k.endswith("config.txt")
    }
    assert all(main["source_sha256"][k] == v for k, v in official.items())
    rows = []
    same = []
    calls = 0
    for case in main["arguments"]["cases"]:
        problem = Problem.read(case)
        folder = directory / case
        folder.mkdir()
        evaluators = {q: Evaluations(problem, q, hardware, folder / f"q{q}") for q in (2, 3)}
        pool = {}
        for q in (2, 3):
            populate_from_main(evaluators[q], source / case / f"q{q}/evaluations")
            for path in sorted((source / case / f"q{q}/evaluations").glob("*.plan.json")):
                placement = load_placement(problem, path)
                pool[fingerprint(placement.submission(problem))] = placement
        for k in range(1, main["arguments"]["cores"] + 1):
            path = source / case / f"q1/k{k}/{case}_multicore_res.json"
            placement = load_placement(problem, path)
            pool[fingerprint(placement.submission(problem))] = placement
        results = {q: {} for q in (2, 3)}
        for key, placement in pool.items():
            for q in (2, 3):
                results[q][key] = evaluators[q].score(placement)[0]
        for k in range(1, main["arguments"]["cores"] + 1):
            eligible = [
                key for key, p in pool.items() if max(c for c, queue in enumerate(p.queues) if queue) < k
            ]
            selected = {q: min(eligible, key=lambda key: results[q][key]["objective"]) for q in (2, 3)}
            times = {q: results[q][selected[q]]["metrics"]["makespan"] for q in (2, 3)}
            for q in (2, 3):
                placement = pool[selected[q]]
                # Remove only idle trailing queues; no core relabelling.
                queues = list(placement.queues)
                while len(queues) > k:
                    queues.pop()
                placement = Placement(placement.units, tuple(queues)).add_idle_cores(k)
                output = folder / f"enhanced_q{q}_k{k}.json"
                write_json(output, placement.submission(problem))
                original = read_json(source / case / f"q{q}/k{k}/metrics.json")["makespan"]
                assert times[q] <= original
                best = directory / "best" / case / f"q{q}/k{k}"
                best.mkdir(parents=True)
                write_json(best / f"{case}_multicore_res.json", placement.submission(problem))
                chosen = dict(results[q][selected[q]]["metrics"])
                chosen.update(
                    case=case,
                    scene=q,
                    cores=k,
                    original_makespan=original,
                    selection="equal_candidate_pool",
                    pool_size=len(eligible),
                )
                chosen["global_lower_bound"] = max(
                    problem.compute_lower_bound(k), chosen["original_copy_bytes"] / hardware.ddr_bandwidth
                )
                chosen["gap_to_global_bound"] = chosen["makespan"] / chosen["global_lower_bound"] - 1
                chosen["tasks_per_core"] = [int(bool(queue)) for queue in placement.queues]
                peaks = {int(c): values for c, values in chosen["memory_peak_by_core"].items()}
                chosen["memory_peak_by_core"] = {
                    c: peaks[c] if placement.queues[c] else {r: 0 for r in hardware.capacities}
                    for c in range(k)
                }
                chosen["official_result"] = os.path.relpath(
                    folder / results[q][selected[q]]["official_result"], best
                )
                write_json(best / "metrics.json", chosen)
            best = directory / "best" / case / f"q1/k{k}"
            best.mkdir(parents=True)
            write_json(
                best / f"{case}_multicore_res.json",
                read_json(source / case / f"q1/k{k}/{case}_multicore_res.json"),
            )
            original_q1 = read_json(source / case / f"q1/k{k}/metrics.json")
            original_q1["official_result"] = os.path.relpath(
                source / case / f"q1/k{k}" / original_q1["official_result"], best
            )
            write_json(best / "metrics.json", original_q1)
            rows.append(
                dict(
                    case=case,
                    cores=k,
                    pool_size=len(eligible),
                    no_l2_makespan=times[2],
                    l2_makespan=times[3],
                    equal_pool_speedup=times[2] / times[3],
                )
            )
            winning = load_placement(problem, source / case / f"q3/k{k}/{case}_multicore_res.json")
            key = fingerprint(winning.submission(problem))
            no_l2 = results[2][key]["metrics"]["makespan"]
            with_l2 = results[3][key]["metrics"]["makespan"]
            same.append(
                dict(
                    case=case,
                    cores=k,
                    no_l2_makespan=no_l2,
                    l2_makespan=with_l2,
                    same_submission_speedup=no_l2 / with_l2,
                    plan_sha256=key,
                )
            )
        case_calls = sum(e.calls for e in evaluators.values())
        calls += case_calls
        write_json(
            folder / "accounting.json",
            dict(
                pool_size=len(pool),
                new_official_calls=case_calls,
                new_calls_by_scene={q: e.calls for q, e in evaluators.items()},
            ),
        )
        print(f"{case}: equal pool={len(pool)} submissions, extra official calls={case_calls}", flush=True)
    write_csv(directory / "equal_pool.csv", rows)
    write_csv(directory / "same_submission.csv", same)
    from .statistics import paired_statistics

    paired_statistics(source, directory, rows, same)
    write_json(
        directory / "completed.json",
        dict(
            cases=len(main["arguments"]["cases"]),
            new_official_calls=calls,
            comparison_kinds=[
                "same Q3-selected submission under both hardware scenes",
                "same complete candidate pool, optimized on each side",
            ],
        ),
    )


def resource_scan(source, name, cases):
    selected = (
        [r["case"] for r in read_json(ROOT / "data/derived/resource_sample_20.json")]
        if cases == ["sample20"]
        else [f"case_{int(n):03d}" for n in cases]
    )
    directory = create_run(
        name, "fixed_submission_cache_resources", dict(source=str(source), cases=selected, cores=4)
    )
    hardware = Hardware.read()
    rows = []
    calls = 0
    for case in selected:
        problem = Problem.read(case)
        placement = load_placement(problem, source / case / f"q3/k4/{case}_multicore_res.json")
        folder = directory / case
        folder.mkdir()
        for capacity in (262144, 1048576, 4194304):
            for bandwidth in (60, 250, 500):
                adjusted = replace(hardware, cache_capacity=capacity, cache_bandwidth=bandwidth)
                if (capacity, bandwidth) == (hardware.cache_capacity, hardware.cache_bandwidth):
                    metrics = read_json(source / case / "q3/k4/metrics.json")
                    result = load_raw(source / case / "q3/k4" / metrics["official_result"])
                    record = dict(metrics=summarize(problem, placement, 3, adjusted, result))
                    reused = True
                else:
                    evaluator = Evaluations(problem, 3, adjusted, folder / f"cap{capacity}_bw{bandwidth}")
                    record, reused = evaluator.score(placement)
                    calls += evaluator.calls
                metrics = record["metrics"]
                rows.append(
                    dict(
                        case=case,
                        cores=4,
                        capacity_bytes=capacity,
                        bandwidth=bandwidth,
                        makespan=metrics["makespan"],
                        hit_bytes=metrics["cache_hit_bytes"],
                        byte_hit_rate=metrics["cache_byte_hit_rate"],
                        reused_standard_result=reused,
                        plan_sha256=fingerprint(placement.submission(problem)),
                    )
                )
        print(f"{case}: nine resource settings of one fixed plan completed", flush=True)
    write_csv(directory / "resources.csv", rows)
    from .statistics import resource_statistics

    resource_statistics(directory, rows, hardware)
    write_json(
        directory / "completed.json",
        dict(
            cases=len(selected),
            observations=len(rows),
            new_official_calls=calls,
            interpretation="Fixed submission; no resource-dependent reoptimization",
        ),
    )


def mechanism_ablation(name, cases, cores, device):
    directory = create_run(name, "mechanism_ablation", dict(cases=cases, cores=cores, device=device))
    hardware = Hardware.read()
    rows = []
    for case in cases:
        problem = Problem.read(case)
        scan = Residency(problem, device)
        folder = directory / case
        folder.mkdir()
        base, _, _ = seed(problem, cores, 2, 2, True, hardware, scan)
        variants = []
        for label, cache, weight in (
            ("pipeline_finish", False, 0),
            ("traffic_price", False, 1),
            ("cache_price", True, 1),
        ):
            placement, diagnostic = price_assignment(problem, base.units, cores, hardware, cache, weight)
            variants.append((label, placement, diagnostic))
        for q in (2, 3):
            evaluator = Evaluations(problem, q, hardware, folder / f"q{q}")
            for label, placement, diagnostic in variants:
                record, reused = evaluator.score(placement)
                assert (
                    diagnostic["boundary_bytes_before_spill"]
                    == record["metrics"]["boundary_bytes_before_spill"]
                )
                rows.append(
                    dict(
                        case=case,
                        scene=q,
                        cores=cores,
                        method=label,
                        makespan=record["metrics"]["makespan"],
                        boundary_bytes=record["metrics"]["boundary_bytes_before_spill"],
                        reused=reused,
                        diagnostic=diagnostic,
                    )
                )
        evaluator = Evaluations(problem, 1, hardware, folder / "q1")
        for topology in ("kahn", "dfs"):
            placement, diagnostic = budget_coarsening(problem, cores, 10, hardware, scan, topology)
            record, reused = evaluator.score(placement)
            rows.append(
                dict(
                    case=case,
                    scene=1,
                    cores=cores,
                    method=topology,
                    makespan=record["metrics"]["makespan"],
                    boundary_bytes=record["metrics"]["boundary_bytes_before_spill"],
                    reused=reused,
                    diagnostic=diagnostic,
                )
            )
        write_json(folder / "accelerator.json", scan.report())
        print(f"{case}: pricing and topology controls completed", flush=True)
    write_json(directory / "ablation.json", rows)
    write_csv(directory / "ablation.csv", [{k: v for k, v in r.items() if k != "diagnostic"} for r in rows])
    write_json(
        directory / "completed.json",
        dict(observations=len(rows), new_official_calls=sum(not r["reused"] for r in rows)),
    )
