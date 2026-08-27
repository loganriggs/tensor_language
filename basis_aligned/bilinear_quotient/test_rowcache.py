import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("rowcache.py")
SPEC = importlib.util.spec_from_file_location("fineweb_rowcache", PATH)
ROWCACHE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROWCACHE)


class FakeEncoder:
    def __init__(self, documents):
        self.documents = documents

    def encode_ordinary(self, text):
        return self.documents[text]


def reference_rows(documents, fw, n, skip):
    """Literal small-data form of census_lib.fineweb_rows."""
    seen = {tuple(fw[r, :32].tolist()) for r in range(fw.shape[0])}
    out = []
    sk = 0
    for text in documents:
        if sk < skip:
            sk += 1
            continue
        tokens = documents[text]
        for start in range(0, len(tokens) - ROWCACHE.T_LEN, ROWCACHE.T_LEN):
            row = tokens[start:start + ROWCACHE.T_LEN]
            if tuple(row[:32]) in seen:
                continue
            out.append(row)
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return torch.tensor(out, dtype=torch.long)


def make_documents():
    def seq(base, length):
        return list(range(base, base + length))

    documents = {
        "short": seq(1000, 500),
        "dedup_then_good": seq(2000, 1027),
        "one": seq(4000, 514),
        "two": seq(5000, 1027),
        "tail": seq(8000, 514),
        "last": seq(9000, 514),
    }
    # The first chunk of dedup_then_good is excluded; its second chunk remains.
    fw = torch.tensor([documents["dedup_then_good"][:32]], dtype=torch.long)
    return documents, fw


def install_fake_modules(monkeypatch, documents, fw):
    fake_cl = types.SimpleNamespace(FW=fw, enc=lambda: FakeEncoder(documents))
    fake_datasets = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: ({"text": text} for text in documents)
    )
    monkeypatch.setitem(sys.modules, "census_lib", fake_cl)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)


def test_multi_is_identical_to_independent_reference_calls(monkeypatch, tmp_path):
    documents, fw = make_documents()
    install_fake_modules(monkeypatch, documents, fw)
    monkeypatch.setattr(ROWCACHE, "CACHE", str(tmp_path))
    specs = [(2, 0), (2, 2), (1, 4)]

    got = ROWCACHE.multi(specs)

    for spec in specs:
        expected = reference_rows(documents, fw, *spec)
        assert torch.equal(got[spec], expected)
        assert tuple(got[spec].shape) == (spec[0], ROWCACHE.T_LEN)
        assert got[spec].dtype == torch.long

    # Once frozen, a cache hit must not touch the stream at all.
    sys.modules["datasets"] = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit unexpectedly opened the stream")
        )
    )
    assert torch.equal(ROWCACHE.get(*specs[1]), got[specs[1]])


def test_incomplete_stream_is_rejected_not_cached(monkeypatch, tmp_path):
    documents, fw = make_documents()
    install_fake_modules(monkeypatch, documents, fw)
    monkeypatch.setattr(ROWCACHE, "CACHE", str(tmp_path))

    with pytest.raises(RuntimeError, match="refusing to cache an incomplete result"):
        ROWCACHE.get(100, 0)
    assert not Path(ROWCACHE._path(100, 0)).exists()


def test_malformed_cache_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(ROWCACHE, "CACHE", str(tmp_path))
    path = Path(ROWCACHE._path(2, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(1, ROWCACHE.T_LEN, dtype=torch.long), path)

    with pytest.raises(RuntimeError, match="invalid FineWeb row cache"):
        ROWCACHE.get(2, 3)
