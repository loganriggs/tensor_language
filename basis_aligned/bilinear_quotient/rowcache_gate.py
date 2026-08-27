# rowcache_gate: the REAL-STREAM verification gate for rowcache.py.
#
# rowcache.py carries an offline mock-stream suite (Codex, 23/23) but its
# docstring is explicit that no scored experiment may import the cache until it
# reproduces census_lib.fineweb_rows against the LIVE stream. bqrunner invokes
# `python <script>` with no arguments, so `rowcache.py --verify` can never run as
# a queued job -- this is the queueable wrapper.
#
# Deliberately tiny: n=8, skip=40. One stream, ~40 examples. If even this cannot
# complete under current bandwidth, that is itself the finding and it means the
# cache cannot be validated until HF_TOKEN lands.
#
# Registered predictions:
#   pred_a rowcache.get(8, skip=40) is BIT-IDENTICAL to census_lib.fineweb_rows(8, skip=40).
#   pred_b the second call is served from disk and returns the identical tensor
#          without opening a stream (cache hit is genuinely a hit).
#   pred_c the cached file round-trips as torch.long with shape exactly (8, 513).
import json, sys, time, os
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch
import census_lib as cl
import rowcache

OUT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/rowcache_gate_results.json'
N, SKIP = 8, 40


def main():
    t0 = time.time()
    p = rowcache._path(N, SKIP)
    if os.path.exists(p):
        os.remove(p)
    print(f'reference stream: census_lib.fineweb_rows({N}, skip={SKIP})', flush=True)
    ref = cl.fineweb_rows(N, skip=SKIP)
    print(f'  ref {tuple(ref.shape)} {ref.dtype} in {time.time()-t0:.1f}s', flush=True)

    t1 = time.time()
    got = rowcache.get(N, SKIP)
    print(f'  rowcache (cold) {tuple(got.shape)} in {time.time()-t1:.1f}s', flush=True)
    pa = tuple(ref.shape) == tuple(got.shape) and bool(torch.equal(ref, got))

    t2 = time.time()
    again = rowcache.get(N, SKIP)
    warm = time.time() - t2
    pb = bool(torch.equal(got, again)) and warm < 2.0
    print(f'  rowcache (warm) in {warm:.3f}s -> cache hit {"YES" if warm < 2.0 else "NO"}', flush=True)

    pc = again.dtype == torch.long and tuple(again.shape) == (N, 513)

    out = {'config': {'n': N, 'skip': SKIP, 'cache_path': p},
           'ref_shape': list(ref.shape), 'got_shape': list(got.shape),
           'warm_seconds': round(warm, 4),
           'predictions': {'pred_a_bit_identical': bool(pa),
                           'pred_b_warm_cache_hit_identical': bool(pb),
                           'pred_c_dtype_shape_roundtrip': bool(pc)},
           'verdict': ('GATE PASSED - scored experiments may import rowcache'
                       if (pa and pb and pc) else
                       'GATE FAILED - rowcache must NOT be imported by scored work'),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}", flush=True)
    print(out['verdict'], flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # LESSONS 14


if __name__ == '__main__':
    main()
