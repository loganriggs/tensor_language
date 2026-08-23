"""OPEN-PROBLEM #6 (the last registered item): the content API's causal grain. Single-atom interventions are
generic (§1128-1129); full-pattern subspace patching is source-specific (§1059-60). WHERE does specificity
emerge? For grain k in {1, 2, 4, 8}: for each target atom a, choose donor positions where a is HIGH, and set the
values of a k-atom group (a + the k−1 next-most-used atoms) to the donor's values for those atoms, at every
position. Readout: a's affinity-set log-prob gain vs other atoms' sets vs a wrong-direction null (same deltas
along non-group directions). Specificity-vs-k curve = the code's effective codeword length.

REGISTERED PREDICTIONS:
  (0) SANITY: k=1 reproduces §1129 (generic, own ≈ other); the k=8 full-pattern swap toward a-high donors is
      the §1059-60 regime and should show the LARGEST own-vs-other margin.
  (a) GRADED EMERGENCE: specificity (own/other ratio) rises monotonically with k and crosses 2x by k=4 ->
      the codeword is ~half the API (report the curve);
  (b) HOLISTIC: only k=8 shows specificity -> the code is all-or-nothing (the full pattern is the codeword);
  (c) if even k=8 is generic at affinity-set grain, the minimal causal unit is larger than the 8-atom skeleton
      (the §1059-60 K=16-64 subspace) — report plainly."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'skeleton_pattern_grain_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64; NATOM = 256; TOPK = 8; STEPS = 3000
H = m.transformer.h
enc = tiktoken.get_encoding('gpt2')
STEER = {'vec': None, 'alpha': 0.0}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def steer_hook(L):
    def h(mo, i_, o_):
        if STEER['vec'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        if STEER['vec'] == 'field':
            fld = STEER['field'][:, :x.shape[1]] if STEER['field'].shape[0] >= x.shape[0] else None
            xm = x + STEER['field_cur'].to(x.dtype)
        else:
            d = STEER['delta']
            xm = x + (d.unsqueeze(-1)*STEER['vec']).to(x.dtype)
        y = mo.Down(mo.Left(xm)*mo.Right(xm)) + mo.Down_bias
        return y.to(o_.dtype)
    return h


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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); nb = NSEQ; T = SEQ - 1

    # capture REF mlp inputs only
    capR = {L: [] for L in REF}
    hs = []
    for L in REF:
        def mkr(L):
            def h(mo, i_, o_): capR[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mkr(L)))
    idsL = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    devsum = None
    for L in REF:
        X = torch.cat(capR[L], 0); capR[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous()
    Cc = (dev @ Uc).contiguous(); del dev, devsum

    sae = TopKSAE(K, NATOM, TOPK, 0).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    with torch.enable_grad():
        for step in range(STEPS):
            ii = torch.randint(0, Cc.shape[0], (4096,), device=DEV)
            xh, _ = sae(Cc[ii]); loss = ((xh - Cc[ii])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    _, code = sae(Cc)
    usage = (code != 0).float().mean(0)
    atoms = usage.argsort(descending=True)[:8].tolist()

    # affinity sets: top tokens by mean atom activation at the PRECEDING position
    aff = {}
    nextok = torch.roll(tok, -1)
    for a in atoms:
        act = code[:, a]
        sums = torch.zeros(V, device=DEV); sums.index_add_(0, nextok, act)
        mean_act = sums / cn.clamp_min(5)
        mean_act[cn < 5] = -1e9
        aff[a] = mean_act.argsort(descending=True)[:100]
    ov = []
    al = list(atoms)
    for i2 in range(len(al)):
        for j2 in range(i2+1, len(al)):
            ov.append(len(set(aff[al[i2]].tolist()) & set(aff[al[j2]].tolist()))/100)
    print(f"atoms {atoms} | affinity pairwise overlap mean {sum(ov)/len(ov):.3f}", flush=True)

    # ATOM INTERCHANGE at REF mlp inputs
    dirs = {a: F.normalize(Uc @ sae.Dm[:, a].detach(), dim=0) for a in atoms}
    hks = [H[L].mlp.register_forward_hook(steer_hook(L)) for L in REF]
    ev_n = 64
    ev_blocks = blocks[:ev_n]
    code_ev = code[:ev_n*T].view(ev_n, T, NATOM)
    g = torch.Generator(device=DEV).manual_seed(0)
    @torch.no_grad()
    def mean_logprob_sets(sets):
        lp_sums = {nm: 0.0 for nm in sets}; ncount = 0
        for i in range(0, ev_n, 8):
            idx = ev_blocks[i:i+8].to(DEV)[:, :-1].contiguous()
            if STEER['vec'] == 'field':
                STEER['field_cur'] = STEER['field'][i:i+8]
            elif STEER['vec'] is not None:
                STEER['delta'] = STEER['delta_full'][i:i+8]
            lp = F.log_softmax(fwd(idx).float(), -1)
            for nm, tset in sets.items():
                lp_sums[nm] += float(lp[..., tset].mean())
            ncount += 1
        return {nm: lp_sums[nm]/ncount for nm in sets}
    sets = {str(a): aff[a] for a in atoms}
    STEER['vec'] = None
    base_lp = mean_logprob_sets(sets)
    # grain sweep: swap k-atom group values to donor (a-high position) values
    _, code_full = sae(Cc)
    grains = [1, 2, 4, 8]
    res_g = {}
    g3 = torch.Generator(device=DEV).manual_seed(5)
    for kk in grains:
        owns = []; oths = []; nuls = []
        for j, a in enumerate(atoms[:4]):                    # 4 target atoms per grain (runtime)
            group = [a] + [b for b in atoms if b != a][:kk-1]
            # donor: a random a-high position's values for the group atoms
            hi_pos = (code[:, a] > code[:, a][code[:, a] > 0].quantile(0.8)).nonzero().squeeze(1)
            dp = hi_pos[torch.randint(0, hi_pos.shape[0], (1,), generator=g3, device=DEV)]
            donor_vals = code[dp, :].squeeze(0)[group]        # k values
            # per-position delta vector in D-space: sum over group of (donor - current)*dir_g (dirs unit; scale by decoder norms)
            deltaD = torch.zeros(code_ev.shape[0], code_ev.shape[1], D, device=DEV)
            for gi, ga in enumerate(group):
                dsc = float((Uc @ sae.Dm[:, ga].detach()).norm())
                dd = (float(donor_vals[gi]) - code_ev[..., ga])*dsc
                deltaD += dd.unsqueeze(-1)*dirs[ga]
            # apply via the steer hook in "vector field" mode: reuse delta_full for magnitude with unit dir? need full field:
            STEER['field'] = deltaD
            STEER['vec'] = 'field'
            lp_a = mean_logprob_sets(sets)
            # wrong-direction null: same magnitudes along rotated (other-atom) directions
            deltaN = torch.zeros_like(deltaD)
            for gi, ga in enumerate(group):
                gb = atoms[(atoms.index(ga)+3) % len(atoms)]
                dsc = float((Uc @ sae.Dm[:, ga].detach()).norm())
                dd = (float(donor_vals[gi]) - code_ev[..., ga])*dsc
                deltaN += dd.unsqueeze(-1)*dirs[gb]
            STEER['field'] = deltaN
            lp_n = mean_logprob_sets(sets)
            STEER['vec'] = None; STEER['field'] = None
            own = lp_a[str(a)] - base_lp[str(a)]
            oth = sum(lp_a[str(c2)] - base_lp[str(c2)] for c2 in atoms if c2 != a)/(len(atoms)-1)
            nul = lp_n[str(a)] - base_lp[str(a)]
            owns.append(own); oths.append(oth); nuls.append(nul)
        mo_, mt_, mn_ = sum(owns)/len(owns), sum(oths)/len(oths), sum(nuls)/len(nuls)
        ratio = mo_/max(mt_, 1e-4) if mt_ > 0 else float('inf') if mo_ > 0 else 0.0
        res_g[str(kk)] = {'own': round(mo_, 4), 'other': round(mt_, 4), 'wrongdir': round(mn_, 4),
                          'own_over_other': round(mo_/max(abs(mt_), 1e-4), 2)}
        print(f"grain k={kk}: own {mo_:+.4f} | other {mt_:+.4f} | wrong-dir {mn_:+.4f} | ratio {res_g[str(kk)]['own_over_other']}", flush=True)
    for h in hks: h.remove()

    rr = {kk: res_g[str(kk)]['own_over_other'] for kk in [1, 2, 4, 8]}
    out = {'atoms': atoms, 'affinity_overlap_mean': round(sum(ov)/len(ov), 3), 'grain_curve': res_g}
    out['pred_a_graded_by_k4'] = bool(rr[4] >= 2 and rr[2] >= rr[1])
    out['pred_b_holistic'] = bool(rr[8] >= 2 and rr[4] < 2)
    out['pred_c_beyond_8'] = bool(rr[8] < 2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"n_specific {ok}/8 | pred_a value-causal {out['pred_a_value_causal']} | pred_b package-smallest {out['pred_b_package_smallest_unit']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
