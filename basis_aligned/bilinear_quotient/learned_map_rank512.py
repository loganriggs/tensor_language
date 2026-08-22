"""[rank-512: is the middle 2/3 gap RANK or NONLINEARITY? a near-full-rank (512) linear read of the input; if the middle jumps toward the front the 2/3 was higher-RANK-but-linear, if flat it is NONLINEARITY] CLEAN (non-leaky) middle-frontier test: is the middle a smooth function of UPSTREAM content — what it READS
— rather than the downstream L15 content (§908 leakage)? For each component, take the content feature from its
OWN INPUT residual (rank-128 PCA of the residual entering that component), plus token+prev embeddings, fit a
smooth map on train, test fresh. This separates what the middle READS (upstream content, this test) from what
it BUILDS (the extra captured by the downstream feature, §908). The middle aggregates content across depth
(§870), so its output = (function of input content) + (new content it adds); the upstream map captures the
first part.

REGISTERED PREDICTIONS:
  (0) SANITY: front (mlp0) still ~0.9 (token-determined, upstream = embedding);
  (a) READ < DOWNSTREAM-UPPER-BOUND: the middle's upstream-content map is ABOVE the 12-topic ~0.10 (it does
      read upstream content) but BELOW the §908 downstream 0.32 (part of its output is content it BUILDS, not
      reads) -> quantifies read-vs-build for the middle;
  (b) report per-component upstream-map fresh vs §908 downstream + shuffled-feature null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'learned_map_rank512_results.json'
NEVAL = 200; SEQ = 256; RPCA = 512; RIDGE = 1e2
COMPS = [(0, 'mlp'), (5, 'attn'), (5, 'mlp'), (8, 'mlp'), (11, 'mlp'), (16, 'mlp')]
DOWNSTREAM_908 = {'mlp0': 0.91, 'attn5': 0.44, 'mlp5': 0.35, 'mlp8': 0.26, 'mlp11': 0.24, 'mlp16': 0.63}
REPL = {'mode': 'off', 'target': None, 'val': None, 'gmean': None}


def submod(L, kind): return getattr(m.transformer.h[L], kind)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off' or REPL['target'] != tag: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = REPL['gmean'].expand(B, T, D).clone() if REPL['mode'] == 'mean' else REPL['val']
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True; trpos = np.repeat(TRAIN, SEQ-1)
    wte = m.transformer.wte.weight.detach().float()
    cur = S[:, :-1].reshape(-1); prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    tags = [f"{k}{L}" for (L, k) in COMPS]
    hooks = [submod(L, k).register_forward_hook(repl_hook_factory(f"{k}{L}")) for (L, k) in COMPS]
    # capture each component's OUTPUT (post) and INPUT (pre)
    outbuf = {t: [] for t in tags}; inbuf = {t: [] for t in tags}; caph = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mkpost(tag):
            def h(mo, i_, o_): outbuf[tag].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mkpre(tag):
            def h(mo, a): inbuf[tag].append(a[0].detach().float().reshape(-1, D))
            return h
        caph.append(submod(L, k).register_forward_hook(mkpost(tag)))
        caph.append(submod(L, k).register_forward_pre_hook(mkpre(tag)))
    REPL['mode'] = 'off'
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in caph: h.remove()
    outs = {t: torch.cat(outbuf[t], 0) for t in tags}; ins = {t: torch.cat(inbuf[t], 0) for t in tags}
    wte_cur = wte[torch.tensor(cur, device=DEV)]; wte_prev = wte[torch.tensor(np.where(prev >= 0, prev, 0), device=DEV)]
    ones = torch.ones(len(cur), 1, device=DEV)
    rng = np.random.RandomState(0); permidx = torch.tensor(rng.permutation(len(cur)), device=DEV)
    tr = torch.tensor(trpos, device=DEV)
    # full CE (test)
    te_blocks = blocks[~TRAIN]; row0 = ntr*(SEQ-1)
    def ce_test(valfull):
        tot = 0.0; n = 0; row = row0
        for i in range(0, te_blocks.shape[0], 8):
            bb = te_blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous(); bsz = idx.shape[0]
            REPL['val'] = valfull[row:row+bsz*(SEQ-1)].reshape(bsz, SEQ-1, D).to(DEV)
            lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel(); row += bsz*(SEQ-1)
        return tot/n
    REPL['mode'] = 'off'; row = row0; tot = 0.0; n = 0
    for i in range(0, te_blocks.shape[0], 8):
        bb = te_blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    ce_full = tot/n
    out = {'ce_full': round(ce_full, 3), 'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; O = outs[tag]; Xin = ins[tag]; REPL['target'] = tag; REPL['gmean'] = O.mean(0)
        gci = Xin.mean(0, keepdim=True); Vp = torch.linalg.svd(Xin[tr] - gci, full_matrices=False)[2][:RPCA].T.contiguous()
        feat = torch.cat([wte_cur, wte_prev, (Xin - gci) @ Vp, ones], 1); featperm = feat[permidx]
        Ftr = feat[tr]; A = Ftr.T @ Ftr + RIDGE*torch.eye(feat.shape[1], device=DEV)
        Ftr_sh = featperm[tr]; A_sh = Ftr_sh.T @ Ftr_sh + RIDGE*torch.eye(feat.shape[1], device=DEV)
        W = torch.linalg.solve(A, Ftr.T @ O[tr]); pred = feat @ W
        Wsh = torch.linalg.solve(A_sh, Ftr_sh.T @ O[tr]); pred_sh = featperm @ Wsh
        REPL['mode'] = 'mean'; ce_mean = ce_test(O)
        REPL['mode'] = 'set'; ce_map = ce_test(pred); ce_sh = ce_test(pred_sh)
        REPL['mode'] = 'off'
        denom = max(ce_mean - ce_full, 1e-6)
        out['components'][tag] = {'upstream_map_fresh': round(float((ce_mean-ce_map)/denom), 3),
                                  'shuffled_null': round(float((ce_mean-ce_sh)/denom), 3),
                                  'downstream_908': DOWNSTREAM_908.get(tag)}
        print(f"{tag:>6}: UPSTREAM map {out['components'][tag]['upstream_map_fresh']:.2f} (null {out['components'][tag]['shuffled_null']:+.2f}) | downstream(§908) {DOWNSTREAM_908.get(tag)}", flush=True)
    for h in hooks: h.remove()
    mids = ['attn5','mlp5','mlp8','mlp11']
    out['mean_upstream_middle'] = round(float(np.mean([out['components'][t]['upstream_map_fresh'] for t in mids])), 3)
    out['mean_downstream_middle'] = round(float(np.mean([DOWNSTREAM_908[t] for t in mids])), 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmiddle: UPSTREAM {out['mean_upstream_middle']} vs DOWNSTREAM(§908) {out['mean_downstream_middle']} — gap = content the middle BUILDS not reads", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
