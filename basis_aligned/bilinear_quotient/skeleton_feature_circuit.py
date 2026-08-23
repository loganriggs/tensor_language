"""NEW THREAD: circuits for the content API's named features (§1113-1119: 8 stable atoms, mostly named,
universal). Two questions per atom: (1) WHO WRITES IT — correlation map over all 36 components between each
component's output projected on the atom's D-space direction (U_c @ d_atom) and the atom's activation at the
L8 read point; (2) IS IT INDIVIDUALLY STEERABLE — dose-steer one atom (add α·U_c d_atom to the L8/10/12 MLP
inputs, patching-style at the read locus) and measure Δ log-prob on the atom's AFFINITY TOKEN SET (top-100
tokens by mean atom activation at the position before them) vs (i) other atoms' affinity sets and (ii) a
random-direction null. §1103 found the CLASS code is written/read as a package — this asks the same of the
content API at atom grain.

REGISTERED PREDICTIONS:
  (0) SANITY: affinity sets are mostly disjoint across atoms (<30% pairwise overlap); steering α chosen at
      ~2x the atom's activation std moves SOMETHING (else dose too small).
  (a) FEATURE-SPECIFIC STEERING: for >= 5 of 8 atoms, own-affinity gain >= 3x mean other-affinity gain and
      >= 3x the random-direction null -> the API features are individually addressable causal variables
      (the §1105 result at content grain);
  (b) WRITER PROFILES DIFFER: per-atom writer correlation profiles across the 36 components are NOT uniform
      (mean pairwise profile correlation < 0.7) — features have distinct sources (else: written as a package,
      the §1103 mirror — report plainly);
  (c) attention-dominance: writer profiles concentrate on attention components (per §1074)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'skeleton_feature_circuit_results.json'
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
        xm = x + (STEER['alpha']*STEER['vec']).to(x.dtype)
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

    # capture: all 36 component outputs + REF mlp inputs
    TAGS = [(k2, L) for L in range(18) for k2 in ('attn', 'mlp')]
    capC = {t: [] for t in TAGS}; capR = {L: [] for L in REF}
    hs = []
    for (k2, L) in TAGS:
        mod = getattr(H[L], k2)
        def mk(t):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                capC[t].append(y.detach().float().reshape(-1, D))
            return h
        hs.append(mod.register_forward_hook(mk((k2, L))))
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

    # (1) writer map
    dirs = {a: F.normalize(Uc @ sae.Dm[:, a].detach(), dim=0) for a in atoms}
    act_z = {a: (code[:, a] - code[:, a].mean())/code[:, a].std().clamp_min(1e-6) for a in atoms}
    writer = {a: {} for a in atoms}
    for t in TAGS:
        O = torch.cat(capC[t], 0); capC[t] = None
        for a in atoms:
            pr = O @ dirs[a]
            prz = (pr - pr.mean())/pr.std().clamp_min(1e-6)
            writer[a][f'{t[0]}{t[1]}'] = round(float((prz*act_z[a]).mean()), 3)
        del O
    profs = torch.tensor([[writer[a][f'{k2}{L}'] for (k2, L) in TAGS] for a in atoms])
    pz = (profs - profs.mean(1, keepdim=True))/profs.std(1, keepdim=True).clamp_min(1e-6)
    pc = (pz @ pz.T)/profs.shape[1]
    off = (pc.sum() - pc.diag().sum())/(len(atoms)**2 - len(atoms))
    attn_share = {a: round(sum(v for k3, v in writer[a].items() if k3.startswith('attn') and v > 0) /
                           max(sum(v for v in writer[a].values() if v > 0), 1e-6), 3) for a in atoms}
    for a in atoms:
        top3 = sorted(writer[a].items(), key=lambda kv: -kv[1])[:3]
        print(f"atom {a}: top writers {top3} | attn-share {attn_share[a]}", flush=True)
    print(f"mean pairwise writer-profile corr {float(off):.3f}", flush=True)

    # (2) dose steering at REF mlp inputs
    hks = [H[L].mlp.register_forward_hook(steer_hook(L)) for L in REF]
    ev_blocks = blocks[:64]
    @torch.no_grad()
    def mean_logprob_sets(sets):
        STEERED = {}
        lp_sums = {nm: 0.0 for nm in sets}; ncount = 0
        for i in range(0, ev_blocks.shape[0], 8):
            idx = ev_blocks[i:i+8].to(DEV)[:, :-1].contiguous()
            lp = F.log_softmax(fwd(idx).float(), -1)
            for nm, tset in sets.items():
                lp_sums[nm] += float(lp[..., tset].mean())*1.0
            ncount += 1
        return {nm: lp_sums[nm]/ncount for nm in sets}
    sets = {str(a): aff[a] for a in atoms}
    g = torch.Generator(device=DEV).manual_seed(0)
    rnd_dir = F.normalize(Uc @ torch.randn(K, generator=g, device=DEV), dim=0)
    STEER['vec'] = None
    base_lp = mean_logprob_sets(sets)
    steer_res = {}
    ok = 0
    for a in atoms:
        alpha = 2.0*float(code[:, a].std())*float((Uc @ sae.Dm[:, a].detach()).norm())/max(float(dirs[a].norm()), 1e-6)
        STEER['vec'] = dirs[a]; STEER['alpha'] = alpha
        lp_a = mean_logprob_sets(sets)
        STEER['vec'] = rnd_dir; STEER['alpha'] = alpha
        lp_r = mean_logprob_sets(sets)
        STEER['vec'] = None
        own = lp_a[str(a)] - base_lp[str(a)]
        oth = sum(lp_a[str(b)] - base_lp[str(b)] for b in atoms if b != a)/(len(atoms)-1)
        rnd_own = lp_r[str(a)] - base_lp[str(a)]
        good = bool(own >= 3*max(oth, 1e-4) and own >= 3*max(rnd_own, 1e-4) and own > 0)
        ok += int(good)
        steer_res[str(a)] = {'own': round(own, 4), 'other_mean': round(oth, 4), 'random_null': round(rnd_own, 4), 'specific': good}
        print(f"steer atom {a}: own {own:+.4f} | other {oth:+.4f} | rnd {rnd_own:+.4f} | specific {good}", flush=True)
    for h in hks: h.remove()

    out = {'atoms': atoms, 'affinity_overlap_mean': round(sum(ov)/len(ov), 3),
           'writer_map': {str(a): writer[a] for a in atoms}, 'attn_share': {str(a): attn_share[a] for a in atoms},
           'writer_profile_corr': round(float(off), 3), 'steering': steer_res, 'n_specific': ok}
    out['pred_a_feature_specific'] = bool(ok >= 5)
    out['pred_b_writers_differ'] = bool(float(off) < 0.7)
    out['pred_c_attention_dominant'] = bool(sum(attn_share.values())/len(atoms) > 0.6)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"n_specific {ok}/8 | writer-profile corr {out['writer_profile_corr']} | mean attn-share {sum(attn_share.values())/len(atoms):.3f}", flush=True)
    print(f"pred_a specific {out['pred_a_feature_specific']} | pred_b writers-differ {out['pred_b_writers_differ']} | pred_c attn-dominant {out['pred_c_attention_dominant']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
