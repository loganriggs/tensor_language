"""CLOSING experiment of the §1113-1116 arc: §1116 relocated the content manifold's main consumer DOWNSTREAM
(deep MLP reads total 0.34 nats vs stream-level +8.4 §1056). Is the READOUT's read also skeleton-dominated, or
is the readout the one true reader of the dense tail? Same skel/tail/fullrem split applied at TWO exit points:
(A) mlp15/16/17 inputs (readout MLPs' reads); (B) the FINAL residual before rms_norm->lm_head (the §1082
component; xbar/U_c/SAE all rebuilt at the final residual for point B — the object rotates by then §1049).

REGISTERED PREDICTIONS:
  (0) SANITY: noop ~0 at both points; partitions near-additive; B's fullrem >> A's (the direct logit path).
  (a) THE TAIL'S CONSUMER IS THE READOUT: at point B, skeleton-only (tail removed) costs >= 0.5x fullrem ->
      the high-rank tail is read exactly once, at the exit; upstream all reads are skeleton (§1115/§1116) —
      the cleanest possible division: skeleton = working code, tail = payload delivered to the readout;
  (b) SKELETON EVERYWHERE: if B's skel/fullrem <= 0.3 too, every reader in the model is skeleton-dominated
      and the tail is largely vestigial for CE — then §1056's stream catastrophe must be re-pooling/norm
      mediated, not read-mediated (report plainly; that would demand a §1056 reinterpretation)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_skeleton_readout_results.json'
NSEQ = 192; SEQ = 256; REF = [15, 16, 17]; K = 64; NATOM = 256; TOPK = 8; STEPS = 3000; RARE_MAX = 2
H = m.transformer.h
SUB = {'mode': None}
ST = {}
CUR = {}


def fwd(idx):
    CUR['idx'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    if SUB.get('final_mode'):
        xbar = ST['xbarF'][idx].to(x.dtype)
        dv = (x - xbar).float()
        c = dv @ ST['UcF']
        r, _ = ST['saeF'](c.reshape(-1, c.shape[-1])); r = r.view_as(c)
        md = SUB['final_mode']
        if md == 'noop': c2 = c
        elif md == 'skel': c2 = r
        elif md == 'tail': c2 = c - r
        else: c2 = torch.zeros_like(c)
        x = x + ((c2 - c) @ ST['UcF'].T).to(x.dtype)
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
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

    # point B: final-residual basis + SAE (rebuilt at the final residual)
    capF = []
    def fcap(idx):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H: x, v1 = blk(x, v1, x0)
        capF.append(x.detach().float().reshape(-1, D))
    for i in range(0, NSEQ, 8): fcap(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    XF = torch.cat(capF, 0); capF.clear()
    xbF = torch.zeros(V, D, device=DEV); xbF.index_add_(0, tok, XF)
    xbF = xbF/cn.clamp_min(1).unsqueeze(1); ST['xbarF'] = xbF.half()
    dvF = XF - xbF[tok]; dvF = dvF - dvF.mean(0); del XF
    _, _, VtF = torch.linalg.svd(dvF, full_matrices=False)
    ST['UcF'] = VtF[:K].T.contiguous()
    CcF = (dvF @ ST['UcF']).contiguous(); del dvF
    saeF = TopKSAE(K, NATOM, TOPK, 0).to(DEV)
    optF = torch.optim.Adam(saeF.parameters(), lr=3e-3)
    with torch.enable_grad():
        for step in range(STEPS):
            idx2 = torch.randint(0, CcF.shape[0], (4096,), device=DEV)
            x2 = CcF[idx2]; xh2, _ = saeF(x2)
            lossF = ((xh2 - x2)**2).mean()
            optF.zero_grad(); lossF.backward(); optF.step()
    with torch.no_grad():
        xh2, _ = saeF(CcF[:20000])
        r2F = 1 - float(((xh2 - CcF[:20000])**2).sum()/(CcF[:20000]**2).sum())
    ST['saeF'] = saeF
    print(f"final-residual SAE recon R2 {r2F:.4f}", flush=True)

    hks = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in REF]
    SUB['mode'] = None; SUB['final_mode'] = None
    base, base_r, base_f = ce_split(blocks, is_rare)
    res = {}
    for md in ['noop', 'skel', 'tail', 'fullrem']:
        SUB['mode'] = md
        c, cr, cf = ce_split(blocks, is_rare)
        res[md] = {'cost': round(c-base, 4), 'rare': round(cr-base_r, 4), 'freq': round(cf-base_f, 4)}
        SUB['mode'] = None
        print(f"{md:>8}: cost {res[md]['cost']} | rare {res[md]['rare']} | freq {res[md]['freq']}", flush=True)
    resB = {}
    for md in ['noop', 'skel', 'tail', 'fullrem']:
        SUB['final_mode'] = md
        c, cr, cf = ce_split(blocks, is_rare)
        resB[md] = {'cost': round(c-base, 4), 'rare': round(cr-base_r, 4), 'freq': round(cf-base_f, 4)}
        SUB['final_mode'] = None
        print(f"FINAL {md:>8}: cost {resB[md]['cost']} | rare {resB[md]['rare']} | freq {resB[md]['freq']}", flush=True)
    for h in hks: h.remove()
    fr = max(res['fullrem']['cost'], 1e-6)
    frB = max(resB['fullrem']['cost'], 1e-6)
    out = {'base_ce': round(base, 4), 'sae_r2': round(r2, 4), 'sae_r2_final': round(r2F, 4),
           'readout_mlps': res, 'final_residual': resB,
           'A_skel_frac': round(res['skel']['cost']/fr, 3), 'A_tail_frac': round(res['tail']['cost']/fr, 3),
           'B_skel_frac': round(resB['skel']['cost']/frB, 3), 'B_tail_frac': round(resB['tail']['cost']/frB, 3)}
    out['pred_a_readout_reads_tail'] = bool(out['B_skel_frac'] >= 0.5)
    out['pred_b_skeleton_everywhere'] = bool(out['B_skel_frac'] <= 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"A (readout MLPs): skel {out['A_skel_frac']} tail {out['A_tail_frac']} | B (final resid): skel {out['B_skel_frac']} tail {out['B_tail_frac']}", flush=True)
    print(f"pred_a readout-reads-tail {out['pred_a_readout_reads_tail']} | pred_b skeleton-everywhere {out['pred_b_skeleton_everywhere']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
