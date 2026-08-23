"""K-SWEEP of §1059's causal content-patching, to close its one caveat (the random-subspace baseline was inflated by
the large K=256 patch dimension). Repeat the activation-patching test at K in {16,64,256}: at SMALL K a few CONTENT
directions should transport topic while a few RANDOM directions inject almost nothing, so the content>>random excess
should be LARGE at small K and shrink as K grows. Same design as §1059 (patch target residual's subspace component with
the source's, deep-middle L6-14, position-aligned), read/write-independent.

REGISTERED PREDICTIONS:
  (a) CONTENT EXCESS CONCENTRATED AT SMALL K: at K=16/64 the content patch's alignment-to-source and KL exceed the
      random-subspace control by a LARGE margin (ratio grows as K shrinks) -> a few top content directions carry the
      topic; the §1059 near-tie in alignment at K=256 was a large-patch artifact;
  (b) report alignment + KL for content vs random at each K.
ORIGINAL §1059 docstring below.
Causal mediation, read/write-independent: does the deep-middle content SUBSPACE causally carry the topic the model
uses downstream? §1055 named it (interpretable topic/register axes), §1056 showed it matters for loss (but that could be
mere perturbation). Decisive test = ACTIVATION PATCHING: run a SOURCE text and a TARGET text; while running the target,
replace the target residual's content-subspace component (projection onto the deep-middle content basis Uref) with the
SOURCE's, at every deep-middle layer (L6-14, after each block, position-aligned). If the target's output moves toward
the SOURCE's predictions -- more than a random-subspace patch of equal dimension does -- then the content subspace
causally transports topic (independent of read=write, unlike steering).

REGISTERED PREDICTIONS:
  (0) SANITY: patching content changes the output (KL>0); random-subspace patch of equal dim changes it less.
  (a) CONTENT CARRIES TOPIC: content-subspace patching moves target predictions TOWARD source predictions -- alignment
      A_content = mean cos(logits_patch - logits_base, logits_source - logits_base) is clearly POSITIVE and >> the
      random-subspace control A_rand; and KL(patch||base) >> KL(rand||base) -> the content subspace causally carries the
      topic used downstream;
  (b) report KL and alignment for content vs random-subspace patch."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_patching_sweep_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15)); KS = [16, 64, 256]; KMAX = 256
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None}   # mode: None|'cap'|'patch'


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ABL:
            U = ST['U']; xs = ST['srcres'][li]
            x = x - (x @ U) @ U.T + (xs @ U) @ U.T     # replace content coords with source's
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_dev(blocks):
    caps = {L: [] for L in REF_LAYERS}; toks = []; hs = []
    for L in REF_LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): caps[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    ST['mode'] = None
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in REF_LAYERS}, torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    # build content basis Uref from pooled L8-12 content deviation
    caps, tok = capture_dev(blocks)
    devsum = None
    for L in REF_LAYERS:
        X = caps[L]; xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X, dv
    dev = devsum / len(REF_LAYERS); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False); U = Vt[:KMAX].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    R = torch.linalg.qr(torch.randn(D, KMAX, generator=g, device=DEV))[0]   # random-subspace control
    del caps, devsum, dev, devc

    # pair sources / targets
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2*n].contiguous()
    # accumulators per (K, subspace)
    acc = {(k, s): {'kl': 0.0, 'al': 0.0} for k in KS for s in ('c', 'r')}; npos = 0
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None   # source (residuals + logits)
        srcres = {L: ST['store'][L] for L in ABL}
        ST['mode'] = None; lb = fwd(ti).float(); base = F.log_softmax(lb, -1)            # target baseline
        dsrc = (ls - lb).reshape(-1, V)
        for k in KS:
            for U_, name in ((U[:, :k], 'c'), (R[:, :k], 'r')):
                ST['mode'] = 'patch'; ST['U'] = U_.contiguous(); ST['srcres'] = srcres
                lp = fwd(ti).float(); ST['mode'] = None
                patch = F.log_softmax(lp, -1)
                kl = (patch.exp() * (patch - base)).sum(-1)
                cos = F.cosine_similarity((lp - lb).reshape(-1, V), dsrc, dim=-1)
                acc[(k, name)]['kl'] += float(kl.sum()); acc[(k, name)]['al'] += float(cos.sum())
        npos += si.shape[0] * si.shape[1]
    out = {'KS': KS, 'abl_range': [ABL[0], ABL[-1]], 'n_positions': npos, 'perK': {}}
    for k in KS:
        c = acc[(k, 'c')]; r = acc[(k, 'r')]
        out['perK'][str(k)] = {
            'content': {'kl': round(c['kl']/npos, 4), 'alignment': round(c['al']/npos, 4)},
            'random': {'kl': round(r['kl']/npos, 4), 'alignment': round(r['al']/npos, 4)},
            'align_excess': round((c['al']-r['al'])/npos, 4), 'kl_ratio': round(c['kl']/max(r['kl'], 1e-6), 2)}
        p = out['perK'][str(k)]
        print(f"K={k}: content KL {p['content']['kl']} al {p['content']['alignment']} | random KL {p['random']['kl']} al {p['random']['alignment']} | excess {p['align_excess']} klx {p['kl_ratio']}", flush=True)
    ksmall = str(KS[0])
    out['pred_a_excess_concentrated_small_K'] = bool(out['perK'][ksmall]['kl_ratio'] > out['perK'][str(KS[-1])]['kl_ratio'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a excess concentrated at small K: {out['pred_a_excess_concentrated_small_K']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
