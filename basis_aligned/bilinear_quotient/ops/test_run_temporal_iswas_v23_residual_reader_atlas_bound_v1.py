import json

import run_temporal_iswas_v23_residual_reader_atlas_bound_v1 as target


def test_dry_run_is_model_free_and_reports_pending_binding(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    monkeypatch.setattr(
        target.binding, "create_binding",
        lambda *, dry_run: {"status": "awaiting_result", "created": False},
    )
    monkeypatch.setattr(
        target.atlas, "main",
        lambda: (_ for _ in ()).throw(AssertionError("atlas executed during dry run")),
    )
    target.main()
    report = json.loads(capsys.readouterr().out)
    assert report["conditional_launcher"] is True
    assert report["binding_status"]["status"] == "awaiting_result"
    assert report["model_loaded"] is False


def test_live_path_binds_before_delegating(monkeypatch):
    monkeypatch.delenv("BQLIB_DRYRUN", raising=False)
    monkeypatch.delenv("BQLIB_NO_MODEL", raising=False)
    calls = []
    monkeypatch.setattr(
        target.binding, "create_binding",
        lambda *, dry_run: calls.append(("bind", dry_run)) or {"status": "bound"},
    )
    monkeypatch.setattr(target.atlas, "main", lambda: calls.append(("atlas", None)))
    target.main()
    assert calls == [("bind", False), ("atlas", None)]


def test_live_path_fails_closed_without_license(monkeypatch):
    monkeypatch.delenv("BQLIB_DRYRUN", raising=False)
    monkeypatch.delenv("BQLIB_NO_MODEL", raising=False)
    monkeypatch.setattr(
        target.binding, "create_binding",
        lambda *, dry_run: {"status": "awaiting_result", "created": False},
    )
    monkeypatch.setattr(
        target.atlas, "main",
        lambda: (_ for _ in ()).throw(AssertionError("unlicensed atlas executed")),
    )
    try:
        target.main()
    except RuntimeError as error:
        assert "not licensed" in str(error)
    else:
        raise AssertionError("unlicensed launcher did not fail closed")
