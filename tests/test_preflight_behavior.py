from __future__ import annotations

import sys
from pathlib import Path

from exp_0304 import preflight_material_gen as preflight


def _touch(path: Path, *, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        path.write_text(text, encoding="utf-8")
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def test_preflight_ready_for_real_run_with_optional_checkpoint_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    wavtok_root = _touch(tmp_path / "wavtok_root")
    wavtok_config = _touch(tmp_path / "wavtok_config.yaml")
    wavtok_ckpt = _touch(tmp_path / "wavtok.ckpt")
    train_cache = _touch(tmp_path / "train_cache.pt")
    val_cache = _touch(tmp_path / "val_cache.pt")
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        preflight,
        "query_gpu",
        lambda _: {
            "status": "ok",
            "index": 1,
            "name": "Fake GPU",
            "memory_total_mib": 12000,
            "memory_used_mib": 2000,
            "memory_free_mib": 10000,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_material_gen.py",
            "--wavtok_root",
            str(wavtok_root),
            "--wavtok_config",
            str(wavtok_config),
            "--wavtok_ckpt",
            str(wavtok_ckpt),
            "--train_cache",
            str(train_cache),
            "--val_cache",
            str(val_cache),
            "--encoder_ckpt",
            str(tmp_path / "missing_encoder.pt"),
            "--report_path",
            str(report_path),
        ],
    )

    rc = preflight.main()
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["status"] == "ready_for_real_run"
    assert report["fallback_mode"] == "pretrained_wavtokenizer"
    assert report["recommended_next_action"] == "launch_short_or_full_run"
    assert report["missing_assets"] == []


def test_preflight_blocks_when_required_asset_is_missing(monkeypatch, tmp_path: Path) -> None:
    wavtok_root = _touch(tmp_path / "wavtok_root")
    wavtok_config = _touch(tmp_path / "wavtok_config.yaml")
    wavtok_ckpt = _touch(tmp_path / "wavtok.ckpt")
    val_cache = _touch(tmp_path / "val_cache.pt")
    encoder_ckpt = _touch(tmp_path / "encoder.pt")
    report_path = tmp_path / "blocked_report.json"

    monkeypatch.setattr(
        preflight,
        "query_gpu",
        lambda _: {
            "status": "ok",
            "index": 1,
            "name": "Fake GPU",
            "memory_total_mib": 12000,
            "memory_used_mib": 1000,
            "memory_free_mib": 11000,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_material_gen.py",
            "--wavtok_root",
            str(wavtok_root),
            "--wavtok_config",
            str(wavtok_config),
            "--wavtok_ckpt",
            str(wavtok_ckpt),
            "--train_cache",
            str(tmp_path / "missing_train_cache.pt"),
            "--val_cache",
            str(val_cache),
            "--encoder_ckpt",
            str(encoder_ckpt),
            "--report_path",
            str(report_path),
        ],
    )

    rc = preflight.main()
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))

    assert rc == 2
    assert report["status"] == "blocked"
    assert report["recommended_next_action"] == "bind_missing_assets"
    assert report["missing_assets"] == ["train_cache"]
