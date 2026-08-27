"""Disk-cached, single-pass FineWeb row loading.  ADDITIVE HELPER -- census_lib
is NOT modified (SWARM_RUNBOOK infra-freeze: purely additive helpers are allowed
mid-wave, behaviour-changing edits to census_lib are not).

WHY THIS EXISTS (measured 2026-08-27):
`census_lib.fineweb_rows(n, skip)` streams with `streaming=True` and advances to
its offset example-by-example (`for ex in ds: if sk<skip: sk+=1; continue`), so
reaching skip=25000 re-downloads and iterates 25,000 examples EVERY call.  With
no HF_TOKEN on this box (credentials.huggingface false; 297 runlogs carry the
unauthenticated warning) that is rate-limited on top.  `writer_floor_question`
spent >24 min at 0% GPU utilisation caching 3x96 rows that five earlier runs had
already streamed at the same skips.

TWO SAVINGS:
  1. disk cache keyed by (n, skip) -- rows at a fixed (n, skip) are deterministic
     given the fixed FW dedup set, so a repeat call is a torch.load;
  2. single-pass multi-harvest -- N distinct skips cost ONE stream to max(skip)
     instead of N streams.

!! UNVERIFIED UNTIL THE GATE PASSES !!  `verify()` must reproduce
census_lib.fineweb_rows exactly (bit-identical tensors) at a cheap skip before
any experiment relies on this.  Do not import it into a scored run until then.
Run `python rowcache.py --verify` when both lanes are clear -- it streams, so
running it during a live lane makes the contention it exists to fix worse.
"""
import os
import torch

CACHE = '/workspace/tensor_language/basis_aligned/bilinear_quotient/.rowcache'
T_LEN = 513


def _path(n, skip):
    return os.path.join(CACHE, f'fineweb_n{n}_skip{skip}.pt')


def _load_checked(path, n):
    rows = torch.load(path, map_location='cpu', weights_only=True)
    expected = (n, T_LEN)
    if not isinstance(rows, torch.Tensor) or tuple(rows.shape) != expected:
        raise RuntimeError(
            f'invalid FineWeb row cache {path}: expected tensor {expected}, '
            f'got {type(rows).__name__} {getattr(rows, "shape", None)}')
    if rows.dtype != torch.long:
        raise RuntimeError(
            f'invalid FineWeb row cache {path}: expected torch.long, got {rows.dtype}')
    return rows


def _save_atomic(rows, path):
    """Never expose a truncated torch.save as a valid cache entry."""
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        torch.save(rows, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def get(n, skip):
    """Cached single (n, skip). Falls back to a one-spec multi-harvest."""
    p = _path(n, skip)
    if os.path.exists(p):
        return _load_checked(p, n)
    return multi([(n, skip)])[(n, skip)]


def multi(specs):
    """specs: [(n, skip), ...] -> {(n, skip): LongTensor}.  One stream for all
    uncached specs, replicating census_lib.fineweb_rows' loop exactly."""
    import census_lib as cl
    os.makedirs(CACHE, exist_ok=True)
    out, need = {}, []
    for n, skip in specs:
        p = _path(n, skip)
        if os.path.exists(p):
            out[(n, skip)] = _load_checked(p, n)
        else:
            need.append((n, skip))
    if not need:
        return out

    from datasets import load_dataset
    e = cl.enc()
    ds = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True)
    # identical dedup set to census_lib.fineweb_rows
    seen = {tuple(cl.FW[r, :32].tolist()) for r in range(cl.FW.shape[0])}
    active = {spec: [] for spec in need}          # spec -> harvested rows
    sk = 0
    for ex in ds:
        if not any(sk >= skip for _, skip in need):
            sk += 1
            continue
        tk = e.encode_ordinary(ex['text'])
        for spec in need:
            n, skip = spec
            if sk < skip or len(active[spec]) >= n:
                continue
            for s0 in range(0, len(tk) - T_LEN, T_LEN):
                row = tk[s0:s0 + T_LEN]
                if tuple(row[:32]) in seen:
                    continue
                active[spec].append(row)
                if len(active[spec]) >= n:
                    break
        sk += 1
        if all(len(active[s]) >= s[0] for s in need):
            break
    for spec in need:
        t = torch.tensor(active[spec], dtype=torch.long)
        if tuple(t.shape) != (spec[0], T_LEN):
            raise RuntimeError(
                f'FineWeb stream ended before {spec}: harvested {tuple(t.shape)}; '
                'refusing to cache an incomplete result')
        _save_atomic(t, _path(*spec))
        out[spec] = t
    return out


def verify(n=8, skip=40):
    """GATE: must reproduce census_lib.fineweb_rows bit-identically."""
    import census_lib as cl
    ref = cl.fineweb_rows(n, skip=skip)
    p = _path(n, skip)
    if os.path.exists(p):
        os.remove(p)
    got = get(n, skip)
    ok = ref.shape == got.shape and bool(torch.equal(ref, got))
    print(f'verify(n={n}, skip={skip}): ref{tuple(ref.shape)} got{tuple(got.shape)} '
          f'-> {"IDENTICAL - gate PASSED" if ok else "MISMATCH - gate FAILED, do not use"}')
    return ok


if __name__ == '__main__':
    import sys
    if '--verify' in sys.argv:
        raise SystemExit(0 if verify() else 1)
    print(__doc__)
