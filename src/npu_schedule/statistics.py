"""Updated IDEA 7.5: observed distributions, selection frequencies and fair ratios."""

from statistics import mean, median

from .problem import ROOT
from .storage import read_json, write_csv, write_json


def main_statistics(directory, rows, seed_rows):
    observations = [
        dict(
            case=r["case"],
            scene=int(r["scene"]),
            cores=int(r["cores"]),
            makespan=int(r["makespan"]),
            winner=r["winner"],
        )
        for r in rows
    ]
    by_key = {(r["case"], r["scene"], r["cores"]): r for r in observations}
    for row in observations:
        row["speedup"] = by_key[row["case"], row["scene"], 1]["makespan"] / row["makespan"]
    distributions = []
    for q, k in sorted({(r["scene"], r["cores"]) for r in observations}):
        group = [r for r in observations if (r["scene"], r["cores"]) == (q, k)]
        ratios = [r["speedup"] for r in group]
        worst = min(group, key=lambda r: (r["speedup"], r["case"]))
        distributions.append(
            dict(
                scene=q,
                cores=k,
                sample_count=len(group),
                mean_speedup=mean(ratios),
                median_speedup=median(ratios),
                min_speedup=min(ratios),
                max_speedup=max(ratios),
                worst_case=worst["case"],
                faster=sum(x > 1 for x in ratios),
                equal=sum(x == 1 for x in ratios),
                slower=sum(x < 1 for x in ratios),
                reference="same-scene whole-graph one-core",
            )
        )
    write_csv(directory / "performance_distribution.csv", distributions)
    seed = {(r["case"], r["scene"], r["cores"]): r for r in seed_rows}
    mechanisms = []
    for q, label in sorted({(r["scene"], r["winner"]) for r in observations}):
        group = [r for r in observations if (r["scene"], r["winner"]) == (q, label)]
        comparable = [r for r in group if r["cores"] > 1]
        reductions = [1 - r["makespan"] / seed[r["case"], q, r["cores"]]["seed_makespan"] for r in comparable]
        mechanisms.append(
            dict(
                scene=q,
                mechanism=label,
                selected_slots=len(group),
                scene_slots=sum(r["scene"] == q for r in observations),
                comparable_multicore_slots=len(comparable),
                strictly_faster_than_own_seed=sum(x > 0 for x in reductions),
                mean_time_reduction=mean(reductions) if reductions else "",
                interpretation="Conditional on selection, not independent mechanism-removal ablation",
            )
        )
    write_csv(directory / "mechanism_selection.csv", mechanisms)
    inventory = {r["case"]: r for r in read_json(ROOT / "data/derived/graph_inventory.json")}
    names = sorted({r["case"] for r in observations})
    assignments = {}
    for feature in ("compute_nodes", "critical_path_cycles"):
        ordered = sorted(names, key=lambda name: (inventory[name][feature], name))
        for index, name in enumerate(ordered):
            assignments[name, feature] = min(4, index * 4 // len(ordered) + 1)
    grouped = []
    for feature in ("compute_nodes", "critical_path_cycles", "components"):
        for q, k in sorted({(r["scene"], r["cores"]) for r in observations}):
            sets = {}
            for row in observations:
                if (row["scene"], row["cores"]) != (q, k):
                    continue
                label = (
                    ("single" if inventory[row["case"]]["components"] == 1 else "multiple")
                    if feature == "components"
                    else str(assignments[row["case"], feature])
                )
                sets.setdefault(label, []).append(row["speedup"])
            for label, values in sorted(sets.items()):
                grouped.append(
                    dict(
                        feature=feature,
                        group=label,
                        scene=q,
                        cores=k,
                        sample_count=len(values),
                        mean_speedup=mean(values),
                        median_speedup=median(values),
                        min_speedup=min(values),
                    )
                )
    write_csv(directory / "structure_groups.csv", grouped)
    costs = []
    for case in names:
        execution = read_json(directory / case / "execution.json")
        for q in sorted({r["scene"] for r in observations if r["case"] == case}):
            search = read_json(directory / case / f"q{q}/search.json")
            events = read_json(directory / case / f"q{q}/candidate_log.json")
            costs.append(
                dict(
                    case=case,
                    scene=q,
                    official_calls=search["actual_calls_completed"],
                    official_seconds=search["official_seconds"],
                    scene_wall_seconds=search["wall_seconds"],
                    generation_seconds=sum(
                        e["generation_seconds"] for e in events if "generation_seconds" in e
                    ),
                    evaluated_references=sum(e["status"] != "duplicate" for e in events),
                    reused_references=sum(e["status"] == "reused" for e in events),
                    gpu_batches_case_total=execution["accelerator"]["scan_batches"],
                    gpu_seconds_case_total=execution["accelerator"]["scan_seconds"],
                )
            )
    write_csv(directory / "execution_cost.csv", costs)


def paired_statistics(source, directory, rows, same):
    paired = {(r["case"], r["cores"]): r for r in same}
    summaries = []
    for k in sorted({r["cores"] for r in rows}):
        group = [r for r in rows if r["cores"] == k]
        values = [r["equal_pool_speedup"] for r in group]
        summaries.append(
            dict(
                cores=k,
                sample_count=len(group),
                mean_equal_pool_speedup=mean(values),
                median_equal_pool_speedup=median(values),
                minimum_equal_pool_speedup=min(values),
                mean_same_submission_speedup=mean(
                    paired[r["case"], k]["same_submission_speedup"] for r in group
                ),
            )
        )
    write_csv(directory / "paired_summary.csv", summaries)
    gains = []
    for q, field in ((2, "no_l2_makespan"), (3, "l2_makespan")):
        differences = [
            read_json(source / r["case"] / f"q{q}/k{r['cores']}/metrics.json")["makespan"] - r[field]
            for r in rows
        ]
        gains.append(
            dict(
                scene=q,
                slots=len(differences),
                improved=sum(x > 0 for x in differences),
                equal=sum(x == 0 for x in differences),
                worse=sum(x < 0 for x in differences),
            )
        )
    write_json(directory / "improvement_counts.json", gains)


def resource_statistics(directory, rows, hardware):
    reference = {
        r["case"]: r["makespan"]
        for r in rows
        if (r["capacity_bytes"], r["bandwidth"]) == (hardware.cache_capacity, hardware.cache_bandwidth)
    }
    groups = []
    for capacity, bandwidth in sorted({(r["capacity_bytes"], r["bandwidth"]) for r in rows}):
        ratios = [
            r["makespan"] / reference[r["case"]]
            for r in rows
            if (r["capacity_bytes"], r["bandwidth"]) == (capacity, bandwidth)
        ]
        groups.append(
            dict(
                capacity_bytes=capacity,
                bandwidth=bandwidth,
                sample_count=len(ratios),
                mean_time_ratio=mean(ratios),
                median_time_ratio=median(ratios),
                min_time_ratio=min(ratios),
                max_time_ratio=max(ratios),
            )
        )
    write_csv(directory / "resource_summary.csv", groups)
