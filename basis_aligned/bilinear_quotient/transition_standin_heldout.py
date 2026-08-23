"""BENCHMARK CERTIFICATION (rule 1: held-out everything — §1088's singleton-leak lesson): the transition
band's own-basis recoveries (§1095 L4 own-64 0.74; §1100 L3 0.927, L5 0.767) were computed with IN-SAMPLE
per-token means and bases. Certify under the strict rule: xbar + own bases built on half A (192 rows),
substitution CE recoveries evaluated on half B (192 rows), layers 3/4/5, modes mtok | own64 | own256 | meanabl
(full sanity). Tokens unseen in A -> global mean. These numbers, if they hold, go into the per-module 90%
table (modules/benchmark.md) as certified.

REGISTERED PREDICTIONS:
  (0) SANITY: full ~0 both halves' pipelines; own256 >= own64 >= mtok.
  (a) CERTIFIED: held-out own-64 recoveries land within 0.12 of the in-sample values (L3 >= 0.8, L4 >= 0.6,
      L5 >= 0.65) -> the transition-band stand-ins are real, benchmark table updated (L3 joins the ~understood
      set; L4/L5 partial);
  (b) LEAK-INFLATED: any layer drops > 0.2 -> the §1100 number was mean-leak (like §1088's deep tok claims);
      correct the ledger/dossier plainly with the certified number."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'transition_standin_heldout_results.json'
NSEQ = 384; SEQ = 256; K = 64
TARGETS = [3, 4, 5]; ALLBASIS = [3, 4, 5]
H = m.transformer.h
SUB = {'layer': -1, 'U': None, 'mode': None}
ST = {'xbar': {}, 'obar': {}}
CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        mt = ST['xbar'][L][CUR['tok']].to(x.dtype)
        unseen = ~ST['seen'][CUR['tok']]
        if unseen.any(): mt[unseen] = ST['glob'][L].to(x.dtype)
        if SUB['mode'] == 'meanabl':
            return ST['obar'][L].view(1, 1, D).expand_as(o_).to(o_.dtype)
        if SUB['mode'] == 'full': xin = x
        elif SUB['mode'] == 'mtok': xin = mt
        else:
            U = SUB['U']; dv = (x - mt).float()
            xin = mt + ((dv @ U) @ U.T).to(x.dtype)
        y = mo.Down(mo.Left(xin)*mo.Right(xin)) + mo.Down_bias
        return y.to(o_.dtype)
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    blocks = rows[:NSEQ//2]          # half A: build
    evalb = rows[NSEQ//2:]           # half B: certify
    V = int(m.lm_head.weight.shape[0])

    cap = {L: [] for L in ALLBASIS}; capO = {L: [] for L in TARGETS}; hs = []
    for L in ALLBASIS:
        def mk(L):
            def h(mo, i_, o_):
                cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
                if L in TARGETS: capO[L].append(o_.detach().float().reshape(-1, D))
                return None
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    bases = {}; S_own = {}
    def dev_of(layers):
        devsum = None
        for L in layers:
            X = torch.cat(cap[L], 0)
            xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
            xb = xb/cn.clamp_min(1).unsqueeze(1)
            if len(layers) == 1 and L in TARGETS:
                ST['xbar'][L] = xb.half(); ST['glob'][L] = X.mean(0)
            dv = X - xb[tok]
            devsum = dv if devsum is None else devsum + dv
        dev = devsum/len(layers); return dev - dev.mean(0)
    sc = torch.zeros(V, device=DEV); sc.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    ST['seen'] = sc > 0; ST['glob'] = {}
    for L in [3, 4, 5]:
        dev = dev_of([L]); _, S, Vt = torch.linalg.svd(dev, full_matrices=False)
        bases[f'own{L}'] = Vt; S_own[L] = S; del dev

    g = torch.Generator(device=DEV).manual_seed(0)
    Ur = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    for L in TARGETS: ST['obar'][L] = torch.cat(capO[L], 0).mean(0)
    for L in ALLBASIS: cap[L] = []
    U = {n: bases[n][:K].T.contiguous() for n in bases}
    overlaps = {}

    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in TARGETS]
    SUB['layer'] = -1; base = ce(evalb)
    res = {}
    for L in TARGETS:
        row = {}
        conds = {'full': None, 'mtok': None,
                 'own64': U[f'own{L}'], 'own256': bases[f'own{L}'][:256].T.contiguous(),
                 'rand64': Ur, 'meanabl': None}
        for mode, UU in conds.items():
            SUB['layer'] = L; SUB['mode'] = mode; SUB['U'] = UU
            row[mode] = round(ce(evalb) - base, 4)
            SUB['layer'] = -1; SUB['mode'] = None
        abl = max(row['meanabl'], 1e-6)
        row_recov = {mode: round(1 - row[mode]/abl, 3) for mode in conds if mode != 'meanabl'}
        res[str(L)] = {'costs': row, 'recov': row_recov}
        print(f"L{L} recov: {row_recov} (meanabl {row['meanabl']})", flush=True)
    for h in hs: h.remove()

    insample = {'3': 0.927, '4': 0.74, '5': 0.767}
    out = {'base_ce': round(base, 4), 'per_layer': res, 'insample_own64': insample,
           'drop': {L2: round(insample[L2] - res[L2]['recov']['own64'], 3) for L2 in insample}}
    out['pred_a_certified'] = bool(res['3']['recov']['own64'] >= 0.8 and res['4']['recov']['own64'] >= 0.6
                                   and res['5']['recov']['own64'] >= 0.65 and max(out['drop'].values()) <= 0.12)
    out['pred_b_leak'] = bool(max(out['drop'].values()) > 0.2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"held-out own-64 recoveries: L3 {res['3']['recov']['own64']} | L4 {res['4']['recov']['own64']} | L5 {res['5']['recov']['own64']} | drops {out['drop']}", flush=True)
    print(f"pred_a certified {out['pred_a_certified']} | pred_b leak {out['pred_b_leak']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
