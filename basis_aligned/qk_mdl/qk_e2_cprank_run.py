"""E2 CP-RANK CAP ON THE BILINEAR MLPS (fresh single-epoch protocol).

Each V8-base MLP computes Down(Left(xn) * Right(xn)): a bilinear form whose CP
rank equals the hidden width (EXP*D = 1056 at width 264) -- every hidden
channel h contributes one rank-1 term Down[:,h] (Left[h,:] x Right[h,:]).
Capping the CP rank at r is therefore EXACTLY shrinking the hidden width to r:
Left, Right become (r x D) and Down becomes (D x r). Arms r=8 and r=32.
Question: what does feature count per module cost in CE (paired vs E0a/E0b on
fresh held), i.e. is per-module feature count usable as an architectural dial?
Param savings reported (each capped MLP drops 12*D^2 - 3*D*r params).

Positive control: E2 with r = EXP*D (full rank) has base-identical shapes;
loading a base V8Route state dict into it must reproduce the base forward to
< 1e-4 (fp32) -- certifies the factory/plumbing changes nothing but the rank.
Note: the capped Down keeps the family's nonzero write init (std
0.02/sqrt(2*depth), generator seed 888 re-drawn at the new shape).
Results -> qk_e2.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, R, V8T, C, W, DEPTH, nn, torch

JP = E.jpath('qk_e2.json')
ARMS = ((8, 'qk_e2_r8', 'E2r8'), (32, 'qk_e2_r32', 'E2r32'))


class E2Route(V8T.V8Route):
    """V8Route with every block's bilinear MLP hidden width (= CP rank of the
    bilinear form) capped at `rank`. Forward is inherited unchanged."""

    def __init__(self, variant, depth, rank):
        super().__init__(variant, depth)
        self.rank = rank
        Dm = self.wte.weight.shape[1]
        gw = torch.Generator().manual_seed(888)      # write-init convention
        for blk in self.h:
            blk.Left = nn.Linear(Dm, rank, bias=False)
            blk.Right = nn.Linear(Dm, rank, bias=False)
            blk.Down = nn.Linear(rank, Dm, bias=False)
            with torch.no_grad():
                blk.Down.weight.copy_(
                    torch.randn(blk.Down.weight.shape, generator=gw)
                    * R.WRITE_INIT_STD)


def make_e2(rank, variant):
    C.register(variant)
    torch.manual_seed(Q.SEED)
    return E2Route(variant, DEPTH, rank).to(E.DEV)


def mlp_params(m):
    return sum(blk.Left.weight.numel() + blk.Right.weight.numel()
               + blk.Down.weight.numel() for blk in m.h)


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    base = C.make_variant('E2ctl').eval().float()
    full = Q.EXP * Q.D
    mfull = make_e2(full, 'E2full').eval().float()
    mfull.load_state_dict(base.state_dict())         # shapes must match exactly
    d = (mfull(idx) - base(idx)).abs().max().item()
    print(f"control E2(r={full})==base with base weights: "
          f"max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-4
    del base, mfull
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()

    counts = {}
    base_m = C.make_variant('E2base_count')
    base_mlp = mlp_params(base_m)
    counts['base'] = dict(W.param_counts(base_m), mlp_params=base_mlp)
    del base_m
    torch.cuda.empty_cache()
    for rank, stem, key in ARMS:
        m = make_e2(rank, key)
        counts[key] = dict(W.param_counts(m), mlp_params=mlp_params(m),
                           rank=rank,
                           mlp_params_saved_vs_base=base_mlp - mlp_params(m))
        del m
        torch.cuda.empty_cache()
    E.merge(JP, 'param_counts', counts)

    for rank, stem, key in ARMS:
        E.train_arm(stem, JP, key,
                    (lambda r=rank, k=key: make_e2(r, k)), E.GC)
        rec = E.loadj(JP).get(key, {})
        rec['rank'] = rank
        E.merge(JP, key, rec)
        E.oldheld_record(stem, (lambda r=rank, k=key: make_e2(r, k)),
                         JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
    print('e2 cprank run done', flush=True)
