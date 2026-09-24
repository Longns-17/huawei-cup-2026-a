"""Export scientific figures and directly measured accelerator timings."""

import csv
from statistics import median
from time import perf_counter

import numpy as np

from .problem import ROOT, Problem
from .residency import Residency
from .storage import load_raw, read_json, write_csv, write_json


def timeline(result_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    result = load_raw(result_path)
    pipes = ("PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V")
    colors = dict(
        PIPE_MTE2="#e3a238", PIPE_MTE3="#9370ae", PIPE_M="#32a89c", PIPE_V="#4479b3", CACHE_READ="#d94b86"
    )
    fig, axis = plt.subplots(figsize=(13, 2 + len(result["per_core_timeline"]) * 1.2), layout="constrained")
    ticks = []
    labels = []
    for row, core in enumerate(result["per_core_timeline"]):
        for p, pipe in enumerate(pipes):
            y = row * 5 + p
            ticks.append(y)
            labels.append(f"C{core['core_id']} {pipe[5:]}")
            entries = [o for o in core["ops"] if o["pipe"] == pipe]
            for entry in entries:
                cache = "memory_path" in entry and entry["memory_path"] == "CACHE_READ"
                axis.broken_barh(
                    [(entry["start"], entry["duration"])],
                    (y - 0.35, 0.7),
                    facecolors=colors["CACHE_READ" if cache else pipe],
                    linewidth=0,
                )
    if "cache_events" in result:
        hits = [e for e in result["cache_events"] if e["event"] == "hit"]
        for event in hits:
            axis.axvline(event["time"], color=colors["CACHE_READ"], alpha=0.5, linewidth=0.8)
    axis.set_yticks(ticks, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Official simulated cycles")
    axis.set_title(f"Actual execution timeline | makespan {result['makespan']:,} cycles")
    axis.legend(
        handles=[Patch(color=color, label=key.replace("PIPE_", "")) for key, color in colors.items()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncols=5,
        frameon=False,
    )
    axis.set_xlim(0, result["makespan"] * 1.015)
    axis.grid(axis="x", alpha=0.15)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(output)


def benchmark(case, repeats):
    problem = Problem.read(case)
    rng = np.random.default_rng(20260924)
    owners = rng.integers(0, 5, size=len(problem.identifiers))
    sequences = [tuple(v for v in problem.order if owners[v] == c) for c in range(5)]
    cpu = Residency(problem, "cpu")
    cuda = Residency(problem, "cuda")
    expected = cpu.compute(sequences)
    assert cuda.compute(sequences) == expected
    times = {"cpu": [], "cuda": []}
    for device, engine in (("cpu", cpu), ("cuda", cuda)):
        for _ in range(repeats):
            started = perf_counter()
            observed = engine.compute(sequences)
            elapsed = perf_counter() - started
            assert observed == expected
            times[device].append(elapsed)
    result = dict(
        case=case,
        nodes=len(problem.identifiers),
        tensors=len(problem.tensors),
        cores=5,
        repeats=repeats,
        cpu_median_seconds=median(times["cpu"]),
        gpu_median_seconds=median(times["cuda"]),
        scan_speedup=median(times["cpu"]) / median(times["cuda"]),
        cuda_initialization_seconds=cuda.upload_seconds,
        device=cuda.device_name,
        exactly_equal=True,
        timings=times,
        scope="Serial-residency proxy including per-call host preparation and transfers; excludes official CPU evaluator and one-time CUDA initialization",
    )
    write_json(ROOT / "reports" / f"benchmark_{case}.json", result)
    print({k: v for k, v in result.items() if k != "timings"})


def report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = read_json(directory / "aggregate.json")
    with (directory / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for q in (1, 2, 3):
        points = [r for r in summary["groups"] if r["scene"] == q]
        if points:
            axes[0].plot(
                [r["cores"] for r in points], [r["mean_speedup"] for r in points], marker="o", label=f"Q{q}"
            )
            axes[1].plot(
                [r["cores"] for r in points], [r["mean_makespan"] for r in points], marker="o", label=f"Q{q}"
            )
    for axis in axes:
        axis.set_xlabel("Number of cores")
        axis.set_xticks(range(1, 6))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Mean of per-case speedups")
    axes[1].set_ylabel("Mean makespan (cycles)")
    fig.suptitle(
        f"Fresh official evaluations: {summary['completed_cases']} cases, {summary['completed_slots']} result slots"
    )
    fig.savefig(figures / "performance.png", dpi=180)
    fig.savefig(figures / "performance.pdf")
    plt.close(fig)
    baseline = []
    ablations = []
    for case in sorted({r["case"] for r in rows}):
        for q in sorted({int(r["scene"]) for r in rows if r["case"] == case}):
            events = read_json(directory / case / f"q{q}/candidate_log.json")
            for k in sorted(
                {
                    int(r["cores"])
                    for r in rows
                    if r["case"] == case and int(r["scene"]) == q and int(r["cores"]) > 1
                }
            ):
                actual = read_json(directory / case / f"q{q}/k{k}/metrics.json")
                entries = [e for e in events if e["cores"] == k and e["status"] != "duplicate"]
                for entry in entries:
                    result = load_raw(directory / case / f"q{q}" / entry["evaluation"])
                    entry["objective"] = (
                        result["makespan"],
                        result["data_movement_bytes"]["added_copy_bytes"],
                    )
                first = next(e for e in entries if e["label"] not in ("whole_graph", "inherited"))
                baseline.append(
                    dict(
                        case=case,
                        scene=q,
                        cores=k,
                        seed=first["label"],
                        seed_makespan=first["objective"][0],
                        selected_makespan=actual["makespan"],
                        speedup_over_seed=first["objective"][0] / actual["makespan"],
                    )
                )
                for label in (
                    "cache_price",
                    "traffic_price",
                    "pipeline_finish",
                    "local",
                    "budget_0.10",
                    "budget_0.05",
                ):
                    removed = [e for e in entries if e["label"] == label]
                    if removed:
                        restricted = min(e["objective"] for e in entries if e["label"] != label)
                        ablations.append(
                            dict(
                                case=case,
                                scene=q,
                                cores=k,
                                omitted=label,
                                original_makespan=actual["makespan"],
                                restricted_makespan=restricted[0],
                                saved_calls=sum(e["status"] == "evaluated" for e in removed),
                                interpretation="Delete one admitted candidate at this k; preserve original inherited plan; no replacement",
                            )
                        )
    if baseline:
        write_csv(directory / "seed_comparison.csv", baseline)
    if ablations:
        write_csv(directory / "candidate_deletion.csv", ablations)
    from .statistics import main_statistics

    main_statistics(directory, rows, baseline)
    write_json(
        directory / "report.json",
        dict(
            sample_count=summary["completed_cases"],
            result_slots=summary["completed_slots"],
            actual_official_calls=summary["actual_official_calls"],
            figure="figures/performance.png",
            claim_scope="Only the cases present in this fresh run; no claim of completing all 100 unless count is 100",
        ),
    )
    print(figures / "performance.png")
