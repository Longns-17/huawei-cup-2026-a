import argparse
from pathlib import Path

from .problem import RAW


def main():
    parser = argparse.ArgumentParser(
        description="FINAL IDEA independent implementation: new code and new experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser(
        "validate", help="Rerun tests, CUDA benchmark, six-case experiments and evidence report"
    )
    validation.add_argument("--name", required=True)
    validation.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    commands.add_parser(
        "verify", help="Verify saved latest artifacts and regenerate the Chinese validation report"
    )
    calibration = commands.add_parser(
        "calibrate", help="Generate new labelled synthetic examples and evaluate them"
    )
    calibration.add_argument("--name", required=True)
    timeline_command = commands.add_parser(
        "timeline", help="Plot actual operation and cache events from an official result"
    )
    timeline_command.add_argument("result", type=Path)
    timeline_command.add_argument("--output", type=Path, required=True)
    commands.add_parser(
        "inventory", help="Recompute all 100 graph descriptors and preselect the resource scan sample"
    )
    run = commands.add_parser(
        "solve", help="Cold-start prefix search; every output is generated from official inputs"
    )
    run.add_argument("--name", required=True, help="New experiment directory name (must not exist)")
    run.add_argument(
        "--cases", nargs="+", default=["all"], help="Official case numbers, e.g. 006 019 078 090, or all"
    )
    run.add_argument("--scenes", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    run.add_argument("--cores", type=int, choices=range(1, 6), default=5)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    report = commands.add_parser("report", help="Plots and tables from completed fresh-run artifacts")
    report.add_argument("run", type=Path)
    paired = commands.add_parser(
        "paired", help="Same-submission cache control and full equal-pool evaluation"
    )
    paired.add_argument("run", type=Path)
    paired.add_argument("--name", required=True)
    resource = commands.add_parser("resources", help="Nine fixed-plan cache configurations per selected case")
    resource.add_argument("run", type=Path)
    resource.add_argument("--name", required=True)
    resource.add_argument("--cases", nargs="+", required=True, help="Official case numbers, or sample20")
    ablation = commands.add_parser("ablation", help="Three pricing methods and Kahn/DFS task-budget controls")
    ablation.add_argument("--name", required=True)
    ablation.add_argument("--cases", nargs="+", default=["078", "090"])
    ablation.add_argument("--cores", type=int, choices=range(2, 6), default=4)
    ablation.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    topology = commands.add_parser(
        "topology", help="Replace only the 10% candidate and compare the full observed candidate set"
    )
    topology.add_argument("run", type=Path)
    topology.add_argument("--name", required=True)
    topology.add_argument("--cases", nargs="+", default=["single"])
    topology.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    benchmark = commands.add_parser(
        "benchmark", help="Measure CPU and actual GPU residency scans on identical legal sequences"
    )
    benchmark.add_argument("--case", default="014")
    benchmark.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.command == "validate":
        from .verification import validate

        validate(args.name, args.workers)
    elif args.command == "verify":
        from .verification import verify

        verify()
    elif args.command == "topology":
        from .topology_control import compare

        compare(args.run.resolve(), args.name, args.cases, args.device)
    elif args.command == "timeline":
        from .report import timeline

        timeline(args.result.resolve(), args.output.resolve())
    elif args.command == "calibrate":
        from .synthetic import calibrate

        calibrate(args.name)
    elif args.command == "inventory":
        from .experiment import inventory

        print(f"Wrote fresh descriptors for {len(inventory())} official cases.")
    elif args.command == "solve":
        from .experiment import batch

        cases = (
            [p.stem for p in sorted(RAW.glob("case_*.json"))]
            if args.cases == ["all"]
            else [f"case_{int(n):03d}" for n in args.cases]
        )
        print(batch(args.name, cases, args.scenes, args.cores, args.device, args.workers))
    elif args.command == "report":
        from .report import report

        report(args.run.resolve())
    elif args.command == "paired":
        from .analysis import paired

        paired(args.run.resolve(), args.name)
    elif args.command == "resources":
        from .analysis import resource_scan

        resource_scan(args.run.resolve(), args.name, args.cases)
    elif args.command == "ablation":
        from .analysis import mechanism_ablation

        mechanism_ablation(args.name, [f"case_{int(n):03d}" for n in args.cases], args.cores, args.device)
    elif args.command == "benchmark":
        from .report import benchmark

        benchmark(f"case_{int(args.case):03d}", args.repeats)
