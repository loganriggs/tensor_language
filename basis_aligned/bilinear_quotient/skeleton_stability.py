"""SPLIT-HALF STABILITY AUDIT of the skeleton arc (§1115-1118) before its numbers harden (the §1101
data-scale rule): two fully independent halves (bases, xbars, SAEs retrained per half; 384 rows total),
re-measuring the headline fractions: deep-band read split (skel ~13% / tail ~72% of fullrem, §1116 idiom at
REF=[8,10,12] to keep runtime; §1115's 12%/78%) per half.

REGISTERED PREDICTIONS:
  (0) SANITY: noop ~0 both halves; partitions near-additive.
  (a) STABLE: skel/fullrem and tail/fullrem agree across halves within 0.07 absolute and bracket the
      original (0.122/0.778) -> the skeleton law is data-stable;
  (b) any fraction outside the band -> flag UNSTABLE and correct the arc's ledger claims (report which)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'skeleton_stability_results.json'
NSEQ = 384; SEQ = 256; REF = [8, 10, 12]; K = 64; NATOM = 256; TOPK = 8; STEPS = 3000; RARE_MAX = 2
H = m.transformer.h
SUB = {'mode': None}
ST = {}
CUR = {}


def fwd(idx):
    CUR['idx'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(n, d, generator=g)*0.1)
        self.Dm = torch.nn.Parameter(torch.randn(d, n, generator=g)*0.1)
        self.k = k
    def forward(self, x):
        a = x @ self.E.T
        top = a.topk(self.k, -1)
        code = torch.zeros_like(a).scatter_(-1, top.indices, top.values)
        return code @ self.Dm.T, code


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        xbar = ST[f'xbar{L}'][CUR['idx']].to(x.dtype)
        dv = (x - xbar).float()
        c = dv @ ST['Uc']                                   # B,T,K coords
        r, _ = ST['sae'](c.reshape(-1, K)); r = r.view_as(c)
        if SUB['mode'] == 'noop':   c2 = c
        elif SUB['mode'] == 'skel': c2 = r                  # tail removed
        elif SUB['mode'] == 'tail': c2 = c - r              # skeleton removed
        else:                        c2 = torch.zeros_like(c)  # fullrem
        xm = x + ((c2 - c) @ ST['Uc'].T).to(x.dtype)
        y = mo.Down(mo.Left(xm)*mo.Right(xm)) + mo.Down_bias
        return y.to(o_.dtype)
    return h


@torch.no_grad()
def ce_split(blocks, is_rare):
    tot = 0.0; n = 0; tr = 0.0; nr = 0; tf = 0.0; nf = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt]
        rm = is_rare[tgt]
        tot += float(ce_tok.sum()); n += tgt.shape[0]
        tr += float(ce_tok[rm].sum()); nr += int(rm.sum())
        tf += float(ce_tok[~rm].sum()); nf += int((~rm).sum())
    return tot/n, tr/max(nr, 1), tf/max(nf, 1)


@torch.no_grad()
def run_half(blocks, tag):
    V = int(m.lm_head.weight.shape[0])
    tfreq = torch.zeros(V, device=DEV)
    ta = blocks[:, 1:].to(DEV).reshape(-1); tfreq.index_add_(0, ta, torch.ones_like(ta, dtype=torch.float))
    is_rare = tfreq <= RARE_MAX

    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0); devsum = None
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    for L in REF:
        X = torch.cat(cap[L], 0); cap[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xb = xb/cn.clamp_min(1).unsqueeze(1)
        ST[f'xbar{L}'] = xb.half()
        dv = X - xb[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    ST['Uc'] = Vt[:K].T.contiguous()
    Cc = (dev @ ST['Uc']).contiguous(); del dev, devsum

    sae = TopKSAE(K, NATOM, TOPK, 0).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    with torch.enable_grad():
        for step in range(STEPS):
            idx2 = torch.randint(0, Cc.shape[0], (4096,), device=DEV)
            x = Cc[idx2]
            xh, _ = sae(x)
            loss = ((xh - x)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xh, _ = sae(Cc[:20000])
        r2 = 1 - float(((xh - Cc[:20000])**2).sum()/(Cc[:20000]**2).sum())
    ST['sae'] = sae
    print(f"SAE recon R2 {r2:.4f} (skeleton variance share)", flush=True)

    hks = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in REF]
    SUB['mode'] = None
    base, base_r, base_f = ce_split(blocks, is_rare)
    res = {}
    for md in ['noop', 'skel', 'tail', 'fullrem']:
        SUB['mode'] = md
        c, cr, cf = ce_split(blocks, is_rare)
        res[md] = {'cost': round(c-base, 4), 'rare': round(cr-base_r, 4), 'freq': round(cf-base_f, 4)}
        SUB['mode'] = None
        print(f"{md:>8}: cost {res[md]['cost']} | rare {res[md]['rare']} | freq {res[md]['freq']}", flush=True)
    for h in hks: h.remove()
    fr = max(res['fullrem']['cost'], 1e-6)
    half = {'base_ce': round(base, 4), 'sae_r2': round(r2, 4), 'conditions': res,
            'skel_frac': round(res['skel']['cost']/fr, 3), 'tail_frac': round(res['tail']['cost']/fr, 3)}
    print(f"[{tag}] skel/fullrem {half['skel_frac']} | tail/fullrem {half['tail_frac']}", flush=True)
    ST.clear()
    return half


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    ha = run_half(rows[:NSEQ//2], 'halfA')
    hb = run_half(rows[NSEQ//2:], 'halfB')
    orig = {'skel_frac': 0.122, 'tail_frac': 0.778}
    stable = {}
    for k2 in orig:
        ok = abs(ha[k2]-hb[k2]) <= 0.07 and min(ha[k2], hb[k2]) - 0.07 <= orig[k2] <= max(ha[k2], hb[k2]) + 0.07
        stable[k2] = {'halfA': ha[k2], 'halfB': hb[k2], 'original': orig[k2], 'stable': bool(ok)}
    out = {'halfA': ha, 'halfB': hb, 'stability': stable,
           'pred_a_stable': bool(all(v['stable'] for v in stable.values())),
           'runtime_s': round(time.time()-t0, 1)}
    import json as js
    js.dump(out, open(OUT, 'w'), indent=1)
    print(f"stability: {stable} | pred_a all-stable {out['pred_a_stable']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
