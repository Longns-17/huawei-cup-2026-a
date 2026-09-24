"""Reproducible validation and evidence-based Chinese report generation.

Failures propagate. There is no retry, CPU fallback, skipped failed candidate,
or substitution of historical results for the current validation run.
"""

import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from .candidates import BUDGET
from .experiment import source_hashes
from .problem import ROOT, Problem
from .storage import fingerprint, load_raw, read_json, write_json


def check_slot(problem, metrics_path):
    metrics = read_json(metrics_path)
    plan = read_json(metrics_path.parent / f"{problem.name}_multicore_res.json")
    assert metrics["case"] == problem.name
    assert set(plan) == {"node_to_subgraph", "core_schedules"}
    assert len(plan["core_schedules"]) == metrics["cores"]
    assert set(plan["node_to_subgraph"]) == {str(u) for u in problem.identifiers}
    assigned = [u for queue in plan["core_schedules"] for u in queue]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(plan["node_to_subgraph"].values())
    assert len(metrics["tasks_per_core"]) == metrics["cores"]
    assert len(metrics["memory_peak_by_core"]) == metrics["cores"]
    official_path = (metrics_path.parent / metrics["official_result"]).resolve()
    stored_plan = official_path.with_name(official_path.name.replace(".official.json.gz", ".plan.json"))
    assert fingerprint(plan) == fingerprint(read_json(stored_plan)), str(metrics_path)
    official = load_raw(official_path)
    assert (metrics["makespan"], metrics["added_copy_bytes"]) == (
        official["makespan"],
        official["data_movement_bytes"]["added_copy_bytes"],
    ), str(metrics_path)
    return metrics


def test_receipt(path):
    suites = ET.parse(path).getroot().findall("testsuite")
    counts = {
        key: sum(int(s.attrib[key]) for s in suites) for key in ("tests", "failures", "errors", "skipped")
    }
    assert counts["tests"] > 0
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    counts["seconds"] = sum(float(s.attrib["time"]) for s in suites)
    return counts


def verify_saved_sources(manifest, context, current_sources, snapshot):
    """Keep the experiment receipt frozen when only this reporting module changes."""
    recorded = manifest["source_sha256"]
    assert context["source_sha256"] == recorded
    assert set(recorded) == set(current_sources)
    with ZipFile(snapshot) as archive:
        for relative, digest in recorded.items():
            normalized = relative.replace("\\", "/")
            assert hashlib.sha256(archive.read(normalized)).hexdigest() == digest, relative
            if normalized != "src/npu_schedule/verification.py":
                assert current_sources[relative] == digest, relative
    return recorded


def verify():
    latest = read_json(ROOT / "experiments/LATEST.json")
    provenance = read_json(ROOT / "specification/input_provenance.json")
    idea_hash = hashlib.sha256((ROOT / "specification/FINAL_IDEA.md").read_bytes()).hexdigest()
    assert idea_hash == latest["idea_sha256"] == provenance["idea_sha256"]
    assert hashlib.sha256(Path(provenance["idea_source"]).read_bytes()).hexdigest() == idea_hash
    for item in provenance["files"]:
        assert hashlib.sha256((ROOT / item["local_path"]).read_bytes()).hexdigest() == item["sha256"]
    current_sources = source_hashes()
    context = read_json(ROOT / "checks/test_context.json")
    tests = test_receipt(ROOT / "checks/tests.xml")
    main = ROOT / "experiments" / latest["main"]
    paired = ROOT / "experiments" / latest["paired"]
    manifest = read_json(main / "manifest.json")
    recorded_sources = verify_saved_sources(manifest, context, current_sources, main / "source_snapshot.zip")
    project_status = read_json(ROOT / "project_status/full_run_status.json")
    for relative, digest in project_status["evidence_sha256"].items():
        evidence = ROOT / "project_status/full_run_evidence" / relative
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest, relative
    cases = manifest["arguments"]["cases"]
    scenes = manifest["arguments"]["scenes"]
    cores = manifest["arguments"]["cores"]
    expected = len(cases) * len(scenes) * cores
    rows = []
    for case in cases:
        problem = Problem.read(case)
        execution = read_json(main / case / "execution.json")
        assert execution["complete"] and execution["accelerator"]["device"] == "cuda"
        assert execution["accelerator"]["scan_batches"] > 0
        for scene in scenes:
            search = read_json(main / case / f"q{scene}/search.json")
            assert search["complete"]
            assert search["actual_calls_started"] == search["actual_calls_completed"]
            assert search["actual_calls_completed"] <= 1 + (cores - 1) * BUDGET[scene]
            values = []
            for k in range(1, cores + 1):
                metrics = check_slot(problem, main / case / f"q{scene}/k{k}/metrics.json")
                assert metrics["complete"]
                assert metrics["admitted_candidates"] <= BUDGET[scene]
                rows.append(metrics)
                values.append((metrics["makespan"], metrics["added_copy_bytes"]))
                enhanced = check_slot(problem, paired / "best" / case / f"q{scene}/k{k}/metrics.json")
                assert (enhanced["makespan"], enhanced["added_copy_bytes"]) <= values[-1]
            assert all(a >= b for a, b in zip(values, values[1:]))
    assert len(rows) == expected
    aggregate = read_json(main / "aggregate.json")
    assert aggregate["completed_slots"] == expected
    assert aggregate["actual_official_calls"] == sum(r["new_official_calls"] for r in rows)
    summaries = {
        key: read_json(ROOT / "experiments" / latest[key] / "completed.json")
        for key in ("main", "paired", "resources", "ablation", "calibration", "topology")
    }
    benchmark = read_json(ROOT / "reports/benchmark_case_014.json")
    assert benchmark["exactly_equal"]
    audit = dict(
        generated_at=datetime.now().astimezone().isoformat(),
        workspace=str(ROOT),
        idea_sha256=idea_hash,
        official_files_verified=len(provenance["files"]),
        official_cases=len(list((ROOT / "data/raw").glob("case_*.json"))),
        test_result=tests,
        scope="Current _code validation batch; the project's separate full run is recorded in project_full_run",
        source_sha256=recorded_sources,
        report_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verified_slots={"main": expected, "enhanced": expected},
        main_official_calls=aggregate["actual_official_calls"],
        experiments=summaries,
        full_100_case_search_completed=(len(cases) == 100 and set(scenes) == {1, 2, 3} and cores == 5),
        project_full_run=project_status,
        gpu_benchmark=benchmark,
        latest=latest,
    )
    write_json(ROOT / "checks/delivery_audit.json", audit)
    write_report(audit, cases, rows)
    print(ROOT / "checks/验证报告.md", flush=True)
    return audit


def write_report(audit, cases, rows):
    benchmark = audit["gpu_benchmark"]
    tests = audit["test_result"]
    experiments = audit["experiments"]
    latest = audit["latest"]
    q5 = ["| 算例 | 问题一 | 问题二 | 问题三 |", "|---|---:|---:|---:|"]
    table = {(r["case"], r["scene"], r["cores"]): r for r in rows}
    for case in cases:
        q5.append(
            "| " + case + " | " + " | ".join(str(table[case, q, 5]["makespan"]) for q in (1, 2, 3)) + " |"
        )
    project = audit["project_full_run"]
    full_status = (
        f"项目正式全量已完成：{project['cases']} 例、三问、1—5 核共 "
        f"{project['main_slots']} 个配置，来自 A题_Idea迭代/results/FINAL_delivery。"
        f"本工作区当前验证批次完成 {len(cases)} 例、{audit['verified_slots']['main']} 个配置；"
        "两批使用各自对应的代码与实验记录。"
    )
    markdown = f"""# 华为杯 A 题：最新 FINAL 代码与验证报告

生成时间：{audit["generated_at"]}。正式工作区：`{ROOT}`。

依据 FINAL 的 SHA-256：`{audit["idea_sha256"]}`。求解源码与本批次的运行清单一致，原始源码快照和测试收据保留；报告生成模块的当前哈希另行记录。原始官方文件校验通过。本报告由 `run.py verify` 从真实记录生成。

**{full_status}**

项目全量、当前独立验证和论文取数入口见 [项目状态与论文数据来源](../project_status/项目状态与论文数据来源.md)。下表及 GPU 测速属于 `_code` 批次；项目全量的完成状态不能由这里的样本数推断。`delivery_audit.json` 中 `full_100_case_search_completed` 仅指当前工作区批次，项目状态在 `project_full_run`。

## 本次检查

| 项目 | 实际结果 | 证据 |
|---|---|---|
| 官方材料 | {audit["official_files_verified"]} 个文件、{audit["official_cases"]} 个正式输入，SHA-256 相符 | [来源记录](../specification/input_provenance.json) |
| 自动测试 | {tests["tests"]} 项通过；失败、错误、跳过均为 0；{tests["seconds"]:.2f} 秒 | [测试记录](tests.xml) |
| 主求解 | {len(cases)} 例 × 3 问 × 1—5 核，{audit["verified_slots"]["main"]} 个槽位；{audit["main_official_calls"]} 次官方调用 | [逐配置结果](../experiments/{latest["main"]}/results.csv) |
| 增强提交 | {audit["verified_slots"]["enhanced"]} 份，提交指纹和目标值均能追溯到官方记录 | [共同池汇总](../experiments/{latest["paired"]}/equal_pool.csv) |
| 同方案 / 共同池 | 新增 {experiments["paired"]["new_official_calls"]} 次官方评价 | [同方案对照](../experiments/{latest["paired"]}/same_submission.csv) |
| 固定方案资源扫描 | {experiments["resources"]["observations"]} 个观测，新增 {experiments["resources"]["new_official_calls"]} 次评价 | [资源表](../experiments/{latest["resources"]}/resources.csv) |
| 定价和拓扑消融 | {experiments["ablation"]["observations"]} 个观测 | [消融表](../experiments/{latest["ablation"]}/ablation.csv) |
| 拓扑候选替换 | {experiments["topology"]["configurations"]} 个配置 | [替换表](../experiments/{latest["topology"]}/topology.csv) |
| 人工校准 | {experiments["calibration"]["observations"]} 个官方观测，与正式输入分开 | [校准表](../experiments/{latest["calibration"]}/calibration.csv) |

覆盖检查包括：计算操作恰好覆盖一次、子图恰好分配一次、核心列表数、4/4/7 准入预算、严格字典序继承、结果文件与提交指纹一致、主结果及增强结果引用的真实官方分数。GPU 测试包含与独立逐步访问模拟的对照；串行驻留代理和官方并发内存峰值分别测试。

## 五核主结果

单位为官方周期。以下是逐例验证，不是 100 例平均成绩。

{chr(10).join(q5)}

## GPU 本次实测

设备：**{benchmark["device"]}**。在正式 014 图上，固定随机种子构造五条合法核心访问序列，CPU 与 GPU 的驻留字节结果完全相同；重复 {benchmark["repeats"]} 次取中位数。

| CPU 扫描中位数 | GPU 扫描中位数 | 扫描模块速度比 | CUDA 初始开销 |
|---:|---:|---:|---:|
| {benchmark["cpu_median_seconds"] * 1000:.3f} ms | {benchmark["gpu_median_seconds"] * 1000:.3f} ms | {benchmark["scan_speedup"]:.2f}× | {benchmark["cuda_initialization_seconds"]:.3f} s |

计时包含每次主机索引准备与数据传输；CUDA 初始化单列。**这个比值只描述驻留扫描，不能当作整个求解器的加速比。** 官方评分仍在 CPU 上执行。原始计时见 [GPU 记录](../reports/benchmark_case_014.json)。

## 代码与入口

当前 `_code` 使用独立编写的 `src/npu_schedule` 实现。旧目录已完整归档，来源见 [部署记录](../specification/implementation_origin.json)。这次没有导入旧实验数据。

保留整图与低核数继承这两个模型规定的比较项。没有吞错、失败重试、跳过失败候选、自动 CPU 回退或兜底解；错误直接暴露。官方原始合法性检查与验证断言照常执行。

- [使用说明](../README.md)
- [重新验证并生成报告](../开始验证.ps1)
- [全量实验入口](../开始全量实验.ps1)
- [机器可读核验记录](delivery_audit.json)

`开始验证.ps1` 会创建新的独立验证批次；`run.py verify` 检查已保存记录并更新报告；`开始全量实验.ps1` 用本工作区实现另起一批 100 例全量。论文已有正式全量依据，无须为了填补“未完成”而重复启动。
"""
    (ROOT / "checks/验证报告.md").write_text(markdown, encoding="utf-8")


def validate(name, workers):
    """Run the bounded development experiment suite, never the 100-case full job."""
    from .analysis import mechanism_ablation, paired, resource_scan
    from .experiment import batch, inventory
    from .report import benchmark, report, timeline
    from .synthetic import calibrate
    from .topology_control import compare

    started = datetime.now().astimezone().isoformat()
    subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "pytest", "-q", "--junitxml=checks/tests.xml"],
        cwd=ROOT,
        check=True,
    )
    write_json(ROOT / "checks/test_context.json", dict(started_at=started, source_sha256=source_hashes()))
    benchmark("case_014", 15)
    inventory()
    cases = [f"case_{n}" for n in ("006", "019", "064", "069", "078", "090")]
    main = batch(name, cases, [1, 2, 3], 5, "cuda", workers)
    report(main)
    paired(main, name + "_pairs")
    resource_scan(main, name + "_resources", ["078", "090"])
    mechanism_ablation(name + "_ablation", ["case_064", "case_069", "case_078", "case_090"], 4, "cuda")
    compare(main, name + "_topology", ["064", "069"], "cuda")
    calibrate(name + "_calibration")
    timeline(
        ROOT / "experiments" / (name + "_calibration") / "cache_staggered/q3.official.json.gz",
        ROOT / "reports/cache_stagger_timeline.png",
    )
    write_json(
        ROOT / "experiments/LATEST.json",
        dict(
            idea_sha256=hashlib.sha256((ROOT / "specification/FINAL_IDEA.md").read_bytes()).hexdigest(),
            main=name,
            paired=name + "_pairs",
            resources=name + "_resources",
            ablation=name + "_ablation",
            topology=name + "_topology",
            calibration=name + "_calibration",
        ),
    )
    verify()
