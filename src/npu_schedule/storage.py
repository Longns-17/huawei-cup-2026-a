"""Explicit fresh-run artifacts; no historical-result loading or error recovery."""

import csv
import gzip
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(submission):
    queues = list(submission["core_schedules"])
    while len(queues) > 1 and not queues[-1]:
        queues.pop()
    value = dict(node_to_subgraph=submission["node_to_subgraph"], core_schedules=queues)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_raw(path, result):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))


def load_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_placement(problem, path):
    from .problem import Placement

    submission = read_json(path)
    ids = sorted(set(submission["node_to_subgraph"].values()))
    dense = {u: i for i, u in enumerate(ids)}
    parts = [[] for _ in ids]
    for v, identifier in enumerate(problem.identifiers):
        parts[dense[submission["node_to_subgraph"][str(identifier)]]].append(v)
    return Placement(
        tuple(tuple(p) for p in parts),
        tuple(tuple(dense[u] for u in q) for q in submission["core_schedules"]),
    )
