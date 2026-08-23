"""CLOSER registered in §1128: dose-steering was feature-GENERIC (0/8); the program's law says interchange
works where steering fails (§892/§894/§1105). Single-ATOM INTERCHANGE at the read locus (L8/10/12 MLP inputs):
at positions where atom a is LOW, add (donor_coeff − own_coeff)·dir_a where donor_coeff is sampled from atom
a's HIGH-activation positions — i.e., set the atom's VALUE to a donor's value, changing nothing else. Readout:
Δ mean log-prob of atom a's affinity set vs other atoms' sets vs a SHUFFLED-ATOM null (same coefficient deltas
applied along a random other atom's direction).

REGISTERED PREDICTIONS:
  (0) SANITY: affinity sets ~disjoint (<0.15 overlap); interventions sized by each atom's own high-quantile.
  (a) VALUE-CAUSAL PER FEATURE: >= 5/8 atoms show own-affinity gain >= 3x other-affinity mean and >= 3x the
      shuffled-atom null -> the two machines share one design law (package-read bundles of separately-authored,
      VALUE-CAUSAL features — §1105's class result at content grain);
  (b) if interchange is also generic, single API features are not independent causal variables under any
      instrument — the package is the smallest causal unit (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'skeleton_atom_interchange_results.json'
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
        d = STEER['delta']                                   # B,T per-position coefficient delta
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
            if STEER['vec'] is not None:
                STEER['delta'] = STEER['delta_full'][i:i+8]
            lp = F.log_softmax(fwd(idx).float(), -1)
            for nm, tset in sets.items():
                lp_sums[nm] += float(lp[..., tset].mean())
            ncount += 1
        return {nm: lp_sums[nm]/ncount for nm in sets}
    sets = {str(a): aff[a] for a in atoms}
    STEER['vec'] = None
    base_lp = mean_logprob_sets(sets)
    res_i = {}; ok = 0
    other_atoms = {a: atoms[(j+1) % len(atoms)] for j, a in enumerate(atoms)}
    for a in atoms:
        acts = code_ev[..., a]
        hi = code[:, a][code[:, a] > 0]
        donor_val = float(hi.quantile(0.9)) if hi.numel() > 100 else float(code[:, a].max())
        delta = (donor_val - acts).clamp_min(0.0)            # raise every position's atom-a value to donor level
        dscale = float((Uc @ sae.Dm[:, a].detach()).norm())
        STEER['delta_full'] = delta*dscale
        STEER['vec'] = dirs[a]
        lp_a = mean_logprob_sets(sets)
        # shuffled-atom null: same deltas along a DIFFERENT atom's direction
        b = other_atoms[a]
        STEER['vec'] = dirs[b]
        lp_n = mean_logprob_sets(sets)
        STEER['vec'] = None
        own = lp_a[str(a)] - base_lp[str(a)]
        oth = sum(lp_a[str(c2)] - base_lp[str(c2)] for c2 in atoms if c2 != a)/(len(atoms)-1)
        nul = lp_n[str(a)] - base_lp[str(a)]                 # does atom-b's direction move atom-a's tokens?
        good = bool(own >= 3*max(oth, 1e-4) and own >= 3*max(nul, 1e-4) and own > 0)
        ok += int(good)
        res_i[str(a)] = {'own': round(own, 4), 'other_mean': round(oth, 4), 'wrongdir_null': round(nul, 4), 'specific': good}
        print(f"interchange atom {a}: own {own:+.4f} | other {oth:+.4f} | wrong-dir {nul:+.4f} | specific {good}", flush=True)
    for h in hks: h.remove()

    out = {'atoms': atoms, 'affinity_overlap_mean': round(sum(ov)/len(ov), 3),
           'interchange': res_i, 'n_specific': ok}
    out['pred_a_value_causal'] = bool(ok >= 5)
    out['pred_b_package_smallest_unit'] = bool(ok <= 2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"n_specific {ok}/8 | pred_a value-causal {out['pred_a_value_causal']} | pred_b package-smallest {out['pred_b_package_smallest_unit']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
