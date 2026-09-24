"""Fresh experiments, input inventory, and honest aggregate statistics."""

import hashlib
import importlib.metadata
import platform
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from statistics import mean, median
from time import perf_counter

from .official import Hardware
from .problem import RAW, ROOT, Problem
from .residency import Residency
from .search import solve_prefix
from .storage import write_csv, write_json


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "official/code").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    paths += [
        ROOT / "run.py",
        ROOT / "requirements.txt",
        ROOT / "requirements-lock.txt",
        ROOT / "pyproject.toml",
        ROOT / "华为杯_code.code-workspace",
        ROOT / "开始验证.ps1",
        ROOT / "开始全量实验.ps1",
        ROOT / "更新验证报告.ps1",
        RAW / "config.txt",
        ROOT / "specification/FINAL_IDEA.md",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def create_run(name, kind, arguments):
    directory = ROOT / "experiments" / name
    directory.mkdir()
    hashes = source_hashes()
    with zipfile.ZipFile(directory / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in hashes:
            archive.write(ROOT / relative, relative)
    write_json(
        directory / "manifest.json",
        dict(
            kind=kind,
            created_at=datetime.now().astimezone().isoformat(),
            arguments=arguments,
            source_sha256=hashes,
            hardware=asdict(Hardware.read()),
            input_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob("case_*.json"))
            },
            python=platform.python_version(),
            platform=platform.platform(),
            packages={p: importlib.metadata.version(p) for p in ("numpy", "cupy-cuda13x", "matplotlib")},
            input_provenance=str(ROOT / "specification/input_provenance.json"),
            result_origin="New evaluations of independently extracted official inputs; no previous solver results imported.",
        ),
    )
    return directory


def inventory():
    rows = []
    for path in sorted(RAW.glob("case_*.json")):
        problem = Problem.read(path.stem)
        work = problem.work(problem.order)
        original = sum(
            t.size for t in problem.tensors if (t.producer is None and t.consumers) or t.final_output
        )
        rows.append(
            dict(
                case=problem.name,
                compute_nodes=len(problem.identifiers),
                tensors=len(problem.tensors),
                edges=sum(map(len, problem.successors)),
                components=len(problem.components),
                pipe_m_cycles=work[0],
                pipe_v_cycles=work[1],
                critical_path_cycles=problem.longest_path(problem.order),
                original_boundary_bytes=original,
                io_compute_ratio=(original / 60) / max(work),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    write_csv(ROOT / "data/derived/graph_inventory.csv", rows)
    write_json(ROOT / "data/derived/graph_inventory.json", rows)
    # Fixed before looking at performance: two representatives in each of ten strata.
    selected = []
    for single in (True, False):
        members = sorted(
            (r for r in rows if (r["components"] == 1) == single),
            key=lambda r: (r["compute_nodes"], r["case"]),
        )
        for band in range(5):
            group = members[len(members) * band // 5 : len(members) * (band + 1) // 5]
            for numerator in (1, 3):
                row = group[(len(group) - 1) * numerator // 4]
                selected.append(
                    dict(
                        case=row["case"], single_component=single, size_band=band + 1, quartile=numerator / 4
                    )
                )
    assert len({r["case"] for r in selected}) == 20
    write_json(ROOT / "data/derived/resource_sample_20.json", selected)
    return rows


def case_job(name, scenes, cores, device, directory):
    started = perf_counter()
    problem = Problem.read(name)
    hardware = Hardware.read()
    scan = Residency(problem, device)
    rows = []
    for scene in scenes:
        rows.extend(solve_prefix(problem, scene, cores, hardware, scan, directory / name / f"q{scene}"))
    write_json(
        directory / name / "execution.json",
        dict(case=name, wall_seconds=perf_counter() - started, accelerator=scan.report(), complete=True),
    )
    return rows


def aggregate(directory, rows):
    rows = sorted(rows, key=lambda r: (r["case"], r["scene"], r["cores"]))
    columns = (
        "case",
        "scene",
        "cores",
        "makespan",
        "added_copy_bytes",
        "spill_copy_bytes",
        "cache_hit_bytes",
        "cache_byte_hit_rate",
        "global_lower_bound",
        "plan_lower_bound",
        "gap_to_global_bound",
        "subgraph_count",
        "winner",
        "complete",
        "admitted_candidates",
        "candidate_budget",
        "new_official_calls",
        "generation_seconds",
    )
    write_csv(directory / "results.csv", [{key: r[key] for key in columns} for r in rows])
    table = {(r["case"], r["scene"], r["cores"]): r for r in rows}
    groups = []
    for scene, k in sorted({(r["scene"], r["cores"]) for r in rows}):
        members = [r for r in rows if r["scene"] == scene and r["cores"] == k]
        ratios = [table[r["case"], scene, 1]["makespan"] / r["makespan"] for r in members]
        groups.append(
            dict(
                scene=scene,
                cores=k,
                sample_count=len(members),
                mean_makespan=mean(r["makespan"] for r in members),
                mean_speedup=mean(ratios),
                median_speedup=median(ratios),
                worst_speedup=min(ratios),
            )
        )
    pairs = []
    for name, k in sorted(
        {(r["case"], r["cores"]) for r in rows if r["scene"] == 3 and (r["case"], 2, r["cores"]) in table}
    ):
        a = table[name, 2, k]
        b = table[name, 3, k]
        pairs.append(
            dict(
                case=name,
                cores=k,
                no_l2_makespan=a["makespan"],
                l2_makespan=b["makespan"],
                independent_search_speedup=a["makespan"] / b["makespan"],
                comparison="independently optimized; includes candidate-set differences",
            )
        )
    if pairs:
        write_csv(directory / "l2_pairs.csv", pairs)
    write_json(
        directory / "aggregate.json",
        dict(
            completed_slots=len(rows),
            completed_cases=len({r["case"] for r in rows}),
            actual_official_calls=sum(r["new_official_calls"] for r in rows),
            groups=groups,
            l2_comparison_kind="Independent searches, not an isolated hardware effect",
        ),
    )
    return groups


def batch(name, cases, scenes, cores, device, workers):
    directory = create_run(
        name, "main_search", dict(cases=cases, scenes=scenes, cores=cores, device=device, workers=workers)
    )
    started = perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(case_job, case, scenes, cores, device, directory) for case in cases]
        for future in as_completed(futures):
            rows.extend(future.result())
            aggregate(directory, rows)
    write_json(
        directory / "completed.json",
        dict(wall_seconds=perf_counter() - started, slots=len(rows), cases=len(cases)),
    )
    return directory
