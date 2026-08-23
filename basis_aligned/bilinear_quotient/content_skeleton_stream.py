"""CLOSING test of the §1117 reinterpretation: §1056's stream catastrophe (+8.4) must be CONSTRUCTION
disruption, not read deprivation (no reader consumes the tail, §1115-1117). Stream-level version of the split:
apply skel/tail/fullrem filtering to the RESIDUAL STREAM ITSELF after every deep-band block (L5-14) — a
persistent edit that corrupts what downstream layers build on, matching §1056's intervention style. Uses the
mid-stack basis+SAE (pooled L8-12 deviations).

REGISTERED PREDICTIONS:
  (0) SANITY: costs here >> the read-interface versions (stream edits compound); noop ~0.
  (a) CONSTRUCTION NEEDS THE SKELETON ONLY: stream-level TAIL removal stays mild (< 0.3x fullrem) while
      SKELETON removal reproduces most of the catastrophe (> 0.6x) -> even construction is skeleton-based;
      the tail is inert residue at every level and §1056's number is skeleton-loss + off-regime compounding;
  (b) CONSTRUCTION NEEDS FULL RANK: if TAIL removal is also catastrophic (> 0.5x fullrem), the construction
      process itself consumes the high-rank object (the §1042/§1051 full-rank facts are about building, not
      reading) — the final division: reads are sparse, construction is dense (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_skeleton_stream_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64; NATOM = 256; TOPK = 8; STEPS = 3000; RARE_MAX = 2
H = m.transformer.h
SUB = {'mode': None}
ST = {}
CUR = {}


BAND = list(range(5, 15))


def fwd(idx):
    CUR['idx'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for Li, blk in enumerate(H):
        x, v1 = blk(x, v1, x0)
        if SUB['mode'] is not None and Li in BAND:
            xbar = ST['xbarS'][idx].to(x.dtype)
            dv = (x - xbar).float()
            c = dv @ ST['Uc']
            r, _ = ST['sae'](c.reshape(-1, c.shape[-1])); r = r.view_as(c)
            md = SUB['mode']
            if md == 'noop': c2 = c
            elif md == 'skel': c2 = r
            elif md == 'tail': c2 = c - r
            else: c2 = torch.zeros_like(c)
            x = x + ((c2 - c) @ ST['Uc'].T).to(x.dtype)
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

    # capture the RESIDUAL STREAM after ref band blocks (8,10,12) for basis + a pooled stream xbar
    capS = []; ids3 = []; idsL = []
    def fcap(idx):
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for Li, blk in enumerate(H):
            x, v1 = blk(x, v1, x0)
            if Li in REF:
                capS.append(x.detach().float().reshape(-1, D))
                ids3.append(idx.reshape(-1))
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fcap(idx)
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    Xs = torch.cat(capS, 0); capS.clear()
    tok3 = torch.cat(ids3, 0)   # row-aligned with Xs (interleaved captures)
    xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok3, Xs)
    cn3 = torch.zeros(V, device=DEV); cn3.index_add_(0, tok3, torch.ones_like(tok3, dtype=torch.float))
    xb = xb/cn3.clamp_min(1).unsqueeze(1)
    ST['xbarS'] = xb  # float32 (massive dims)
    dev = Xs - xb[tok3]; dev = dev - dev.mean(0); del Xs
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    ST['Uc'] = Vt[:K].T.contiguous()
    Cc = (dev @ ST['Uc']).contiguous(); del dev

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

    SUB['mode'] = None
    base, base_r, base_f = ce_split(blocks, is_rare)
    res = {}
    for md in ['noop', 'skel', 'tail', 'fullrem']:
        SUB['mode'] = md
        c, cr, cf = ce_split(blocks, is_rare)
        res[md] = {'cost': round(c-base, 4), 'rare': round(cr-base_r, 4), 'freq': round(cf-base_f, 4)}
        SUB['mode'] = None
        print(f"{md:>8}: cost {res[md]['cost']} | rare {res[md]['rare']} | freq {res[md]['freq']}", flush=True)
    fr = max(res['fullrem']['cost'], 1e-6)
    out = {'base_ce': round(base, 4), 'sae_r2': round(r2, 4), 'conditions': res,
           'skel_cost_frac_of_fullrem': round(res['skel']['cost']/fr, 3),
           'tail_cost_frac_of_fullrem': round(res['tail']['cost']/fr, 3)}
    # NOTE condition names: 'skel' = tail REMOVED from stream (skeleton kept); 'tail' = skeleton REMOVED
    out['pred_a_construction_skeleton_only'] = bool(res['skel']['cost'] < 0.3*fr and res['tail']['cost'] > 0.6*fr)
    out['pred_b_construction_full_rank'] = bool(res['skel']['cost'] >= 0.5*fr)
    dom = 'tail' if res['tail']['cost'] > res['skel']['cost'] else 'skel'
    out['dominant_rare_freq'] = round(res[dom]['rare']/max(res[dom]['freq'], 1e-4), 2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"skel/fullrem {out['skel_cost_frac_of_fullrem']} | tail/fullrem {out['tail_cost_frac_of_fullrem']} | dominant {dom} rare/freq {out['dominant_rare_freq']}", flush=True)
    print(f"pred_a construction-skeleton-only {out['pred_a_construction_skeleton_only']} | pred_b construction-full-rank {out['pred_b_construction_full_rank']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
