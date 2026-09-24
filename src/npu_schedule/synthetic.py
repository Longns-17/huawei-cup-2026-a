"""Explicit synthetic calibration inputs, kept separate from official contest cases."""

from .experiment import create_run
from .official import Hardware, evaluate
from .problem import ROOT, Placement, Problem
from .storage import save_raw, write_csv, write_json


def make_problem(durations, pipes, tensors):
    """tensors: (producer dense index or None, consumer indices, bytes, final)."""
    ops = [dict(id=100 + i, op="CALC", pipe=f"PIPE_{pipes[i]}", cycles=d) for i, d in enumerate(durations)]
    data = []
    edges = []
    for t, (producer, readers, size, final) in enumerate(tensors):
        tid = 1000 + t
        data.append(dict(id=tid, pos="UB", size=size))
        if producer is None:
            ddr = 3000 + 2 * t
            copy = 2000 + 2 * t
            data.append(dict(id=ddr, pos="DDR", size=size))
            ops.append(dict(id=copy, op="COPY_IN", pipe="PIPE_MTE2", cycles=0))
            edges.extend([dict(source=ddr, target=copy), dict(source=copy, target=tid)])
        else:
            edges.append(dict(source=100 + producer, target=tid))
        edges.extend(dict(source=tid, target=100 + r) for r in readers)
        if final:
            ddr = 3001 + 2 * t
            copy = 2001 + 2 * t
            data.append(dict(id=ddr, pos="DDR", size=size))
            ops.append(dict(id=copy, op="COPY_OUT", pipe="PIPE_MTE3", cycles=0))
            edges.extend([dict(source=tid, target=copy), dict(source=copy, target=ddr)])
    return Problem.from_dict("synthetic", dict(ops=ops, tensors=data, edges=edges))


def calibrate(name):
    directory = create_run(name, "synthetic_calibration", dict(official_case=False))
    inputs = ROOT / "data/derived" / name
    inputs.mkdir()
    cache = make_problem(
        [1] * 4,
        ["V"] * 4,
        [(None, [0, 1], 600, False), (None, [2, 3], 30000, False)] + [(v, [], 60, True) for v in range(4)],
    )
    many = make_problem(
        [1] * 4, ["V"] * 4, [(0, [1, 2, 3], 60, False)] + [(v, [], 60, True) for v in (1, 2, 3)]
    )
    specifications = [
        ("cache_together", cache, Placement(((0,), (1,), (2,), (3,)), ((0, 2), (1, 3)))),
        ("cache_staggered", cache, Placement(((0,), (1,), (2,), (3,)), ((0, 2), (3, 1)))),
        ("one_to_many_two_cores", many, Placement(((0,), (1, 2), (3,)), ((0,), (1, 2)))),
        ("one_to_many_three_cores", many, Placement(((0,), (1, 2), (3,)), ((0,), (1,), (2,)))),
    ]
    hardware = Hardware.read()
    rows = []
    for label, problem, placement in specifications:
        write_json(inputs / f"{label}.json", problem.raw)
        folder = directory / label
        folder.mkdir()
        write_json(folder / "submission.json", placement.submission(problem))
        for scene in (1, 2, 3):
            result = evaluate(problem, placement, scene, hardware)
            save_raw(folder / f"q{scene}.official.json.gz", result)
            rows.append(
                dict(
                    case=label,
                    scene=scene,
                    makespan=result["makespan"],
                    scheduled_copy_bytes=result["data_movement_bytes"]["scheduled_copy_bytes"],
                )
            )
    write_csv(directory / "calibration.csv", rows)
    write_json(
        directory / "completed.json",
        dict(synthetic=True, observations=len(rows), new_official_calls=len(rows)),
    )
    print(directory)
