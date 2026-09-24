"""Replace the 10% Task-budget candidate while fixing other observed candidates."""

from statistics import mean

from .analysis import populate_from_main
from .construction import budget_coarsening
from .experiment import create_run
from .official import Hardware, objective
from .problem import ROOT, Problem
from .residency import Residency
from .search import Evaluations
from .storage import load_raw, read_json, write_csv, write_json


def compare(source, name, cases, device):
    selected = (
        [r["case"] for r in read_json(ROOT / "data/derived/graph_inventory.json") if r["components"] == 1]
        if cases == ["single"]
        else [f"case_{int(n):03d}" for n in cases]
    )
    directory = create_run(
        name, "topology_candidate_replacement", dict(source=str(source), cases=selected, device=device)
    )
    hardware = Hardware.read()
    rows = []
    calls = 0
    cores = read_json(source / "manifest.json")["arguments"]["cores"]
    for case in selected:
        problem = Problem.read(case)
        scan = Residency(problem, device)
        folder = directory / case
        folder.mkdir()
        evaluator = Evaluations(problem, 1, hardware, folder / "evaluations")
        populate_from_main(evaluator, source / case / "q1/evaluations")
        log = read_json(source / case / "q1/candidate_log.json")
        for k in range(2, cores + 1):
            entries = {e["label"]: e for e in log if e["cores"] == k}
            original_candidate = entries["budget_0.10"]
            if original_candidate["status"] == "duplicate":
                original_candidate = entries[original_candidate["duplicate_of"]]
            original_value = objective(load_raw(source / case / "q1" / original_candidate["evaluation"]))
            retained = [
                objective(load_raw(source / case / "q1" / e["evaluation"]))
                for e in entries.values()
                if e["label"] != "budget_0.10" and e["status"] != "duplicate"
            ]
            placement, diagnostic = budget_coarsening(problem, k, 10, hardware, scan, "dfs")
            record, reused = evaluator.score(placement)
            replaced = min(retained + [record["objective"]])
            original_search = read_json(source / case / f"q1/k{k}/metrics.json")["makespan"]
            rows.append(
                dict(
                    case=case,
                    cores=k,
                    kahn_candidate=original_value[0],
                    dfs_candidate=record["objective"][0],
                    original_search=original_search,
                    replaced_search=replaced[0],
                    time_ratio=replaced[0] / original_search,
                    reused=reused,
                    scope="Replace only this k candidate; preserve other candidates and original inherited plan",
                )
            )
            write_json(folder / f"k{k}_dfs_submission.json", placement.submission(problem))
        calls += evaluator.calls
        print(f"{case}: topology replacement control completed", flush=True)
    write_csv(directory / "topology.csv", rows)
    write_json(
        directory / "completed.json",
        dict(
            configurations=len(rows),
            new_official_calls=calls,
            improved=sum(r["time_ratio"] < 1 for r in rows),
            equal=sum(r["time_ratio"] == 1 for r in rows),
            worse=sum(r["time_ratio"] > 1 for r in rows),
            mean_time_ratio=mean(r["time_ratio"] for r in rows),
            max_time_ratio=max(r["time_ratio"] for r in rows),
        ),
    )
