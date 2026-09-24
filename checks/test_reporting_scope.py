"""Regression checks for experiment provenance and project/workspace scope."""

import hashlib
from zipfile import ZipFile

import pytest

from npu_schedule import verification
from npu_schedule.problem import ROOT
from npu_schedule.storage import read_json


def snapshot(tmp_path):
    original = {
        "src/npu_schedule/verification.py": b"old report",
        "src/npu_schedule/search.py": b"original solver",
    }
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in original.items()}
    path = tmp_path / "source_snapshot.zip"
    with ZipFile(path, "w") as archive:
        for name, data in original.items():
            archive.writestr(name, data)
    return path, hashes


def test_report_edit_preserves_original_experiment_receipt(tmp_path):
    path, recorded = snapshot(tmp_path)
    current = recorded | {"src/npu_schedule/verification.py": hashlib.sha256(b"new report").hexdigest()}
    result = verification.verify_saved_sources(
        {"source_sha256": recorded}, {"source_sha256": recorded}, current, path
    )
    assert result == recorded
    assert result["src/npu_schedule/verification.py"] != current["src/npu_schedule/verification.py"]


def test_solver_change_cannot_reuse_old_validation_receipt(tmp_path):
    path, recorded = snapshot(tmp_path)
    changed = recorded | {"src/npu_schedule/search.py": hashlib.sha256(b"different solver").hexdigest()}
    with pytest.raises(AssertionError, match="search.py"):
        verification.verify_saved_sources(
            {"source_sha256": recorded}, {"source_sha256": recorded}, changed, path
        )


def test_changed_original_snapshot_is_rejected(tmp_path):
    path, recorded = snapshot(tmp_path)
    with ZipFile(path, "w") as archive:
        archive.writestr("src/npu_schedule/verification.py", b"changed original record")
        archive.writestr("src/npu_schedule/search.py", b"original solver")
    with pytest.raises(AssertionError, match="verification.py"):
        verification.verify_saved_sources(
            {"source_sha256": recorded}, {"source_sha256": recorded}, recorded, path
        )


def test_report_distinguishes_full_project_and_small_validation(monkeypatch, tmp_path):
    audit = read_json(ROOT / "checks/delivery_audit.json")
    audit["project_full_run"] = read_json(ROOT / "project_status/full_run_status.json")
    main = ROOT / "experiments" / audit["latest"]["main"]
    manifest = read_json(main / "manifest.json")
    cases = manifest["arguments"]["cases"]
    rows = [read_json(main / case / f"q{q}/k5/metrics.json") for case in cases for q in (1, 2, 3)]
    (tmp_path / "checks").mkdir()
    monkeypatch.setattr(verification, "ROOT", tmp_path)
    verification.write_report(audit, cases, rows)
    report = (tmp_path / "checks/验证报告.md").read_text(encoding="utf-8")
    assert "项目正式全量已完成：100 例" in report
    assert "1500 个配置" in report
    assert "本工作区当前验证批次完成 6 例、90 个配置" in report
    assert "100 例正式全量尚未完成" not in report
    assert audit["full_100_case_search_completed"] is False
    assert audit["project_full_run"]["completed"] is True
