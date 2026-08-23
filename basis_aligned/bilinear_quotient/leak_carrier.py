# leak_carrier: which component physically carries each term of the transport law?
#
# §1153's two-term law: patched positions read their own coords (~full); unpatched positions
# get a small, distance-decaying, causally-masked LEAK. §1156-57: at the readout, ~68% of
# full-patch KL is the coords directly, ~32% is source-flavored material MANUFACTURED by the
# deep blocks. Neither term has a named carrier yet. Candidates: middle ATTENTION (the only
# cross-position component — the leak's only possible carrier, but test it) and the deep
# MLPs (position-local — the natural manufacturer of the 32%).
#
# Method: PATH FREEZING. During a patched target run, replace a module family's outputs in
# blocks 7-14 with the outputs captured from the UNPATCHED base run of the same text (same
# positions, same run structure — matched removal point, §1066). A frozen family cannot
# propagate any patch-induced change; whatever transport survives is carried by the other
# paths. Patch layers L6-14 as usual (patch applies to the residual AFTER each block).
#   NOTE on ordering: freezing uses base-run module outputs, so within each block the frozen
#   family contributes exactly its base behavior; the patch then re-imposes source coords on
#   the residual. Attention-frozen ⇒ no cross-position spread of patched content in L7-14.
#
# Conditions (K=256, fresh rows, §1150-58 harness; capture base attn/mlp outputs per batch):
#   scat50            — reference (patched 0.891 / unpatched 0.358)
#   scat50_attnfroze  — attention outputs L7-14 frozen to base
#   scat50_mlpfroze   — MLP outputs L7-14 frozen to base
#   full9             — reference (0.8994, KL 8.02)
#   full9_attnfroze   — cross-position manufacture blocked
#   full9_mlpfroze    — local manufacture blocked
#   r256              — null
#
# Registered predictions:
#   pred_a LEAK IS ATTENTION-BORNE: scat50_attnfroze unpatched-position alignment < 0.4 ×
#          the reference unpatched value, while its patched-position alignment stays ≥ 0.85 ×
#          reference patched.
#   pred_b MANUFACTURE IS MLP-BORNE: full9_mlpfroze KL loses ≥ half of the consequence share
#          (KL drop from full9 ≥ 0.15 × 8.0), while full9_attnfroze KL drop is smaller.
#   pred_c LOCALITY UNTOUCHED BY MLP FREEZE: scat50_mlpfroze patched-position alignment
#          ≥ 0.9 × reference patched (own-coord reading is the readout's, not the deep MLPs').
# Control: r256; freezing on an UNPATCHED run is exact identity (checked: freeze_null ≈ 0 KL).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'leak_carrier_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
FRZ = list(range(7, 15))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'mask': None,
      'freeze': None, 'frozen_out': None, 'capture_out': None}

# hooks on attn and mlp modules of blocks 7-14: capture or replace outputs
def mk_hook(kind, li):
    def h(mo, i_, o_):
        out = o_[0] if isinstance(o_, tuple) else o_
        if ST['capture_out'] is not None:
            ST['capture_out'][(kind, li)] = out.detach()
            return None
        if ST['freeze'] == kind and ST['frozen_out'] is not None:
            rep = ST['frozen_out'][(kind, li)]
            return (rep,) + tuple(o_[1:]) if isinstance(o_, tuple) else rep
        return None
    return h

HOOKS = []
for li in FRZ:
    HOOKS.append(m.transformer.h[li].attn.register_forward_hook(mk_hook('attn', li)))
    HOOKS.append(m.transformer.h[li].mlp.register_forward_hook(mk_hook('mlp', li)))


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ABL:
            U = ST['U']; xs = ST['srcres'][li]
            xn = x - (x @ U) @ U.T + (xs @ U) @ U.T
            x = torch.where(ST['mask'], xn, x) if ST['mask'] is not None else xn
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_dev(blocks):
    caps = {L: [] for L in REF_LAYERS}; toks = []; hs = []
    for L in REF_LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_):
                caps[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    ST['mode'] = None
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in REF_LAYERS}, torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])

    caps, tok = capture_dev(blocks)
    devsum = None
    for L in REF_LAYERS:
        X = caps[L]; xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X, dv
    dev = devsum / len(REF_LAYERS); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    U256 = Vt[:256].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    R256 = torch.linalg.qr(torch.randn(D, 256, generator=g, device=DEV))[0]
    del caps, devsum, dev, devc

    CONDS = [('scat50', U256, 'scat', None), ('scat50_attnfroze', U256, 'scat', 'attn'),
             ('scat50_mlpfroze', U256, 'scat', 'mlp'),
             ('full9', U256, 'full', None), ('full9_attnfroze', U256, 'full', 'attn'),
             ('full9_mlpfroze', U256, 'full', 'mlp'), ('r256', R256, 'full', None)]
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {c: {'kl': 0.0, 'al': 0.0} for c, _, _, _ in CONDS}
    pp = {c: [0.0, 0] for c in ('scat50', 'scat50_attnfroze', 'scat50_mlpfroze')}
    uu = {c: [0.0, 0] for c in ('scat50', 'scat50_attnfroze', 'scat50_mlpfroze')}
    freeze_null_kl = 0.0
    npos = 0
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        # base run: capture module outputs for freezing
        ST['capture_out'] = {}; lb = fwd(ti).float()
        frozen = ST['capture_out']; ST['capture_out'] = None
        base = F.log_softmax(lb, -1)
        # freeze-identity null (once per batch): freeze attn on UNPATCHED run == base exactly
        if npos == 0:
            ST['freeze'] = 'attn'; ST['frozen_out'] = frozen
            l0 = fwd(ti).float(); ST['freeze'] = None
            p0 = F.log_softmax(l0, -1)
            freeze_null_kl = float((p0.exp() * (p0 - base)).sum(-1).mean())
        B, T = ti.shape
        scat = torch.zeros(B, T, dtype=torch.bool, device=DEV)
        for b in range(B):
            perm = torch.randperm(T, generator=gp, device=DEV)
            scat[b, perm[:T // 2]] = True
        for cname, U, mtype, frz in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres
            ST['mask'] = scat.unsqueeze(-1) if mtype == 'scat' else None
            ST['freeze'] = frz; ST['frozen_out'] = frozen if frz else None
            lp = fwd(ti).float()
            ST['mode'] = None; ST['mask'] = None; ST['freeze'] = None; ST['frozen_out'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb), (ls - lb), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
            if cname in pp:
                pp[cname][0] += float(cos[scat].sum()); pp[cname][1] += int(scat.sum())
                uu[cname][0] += float(cos[~scat].sum()); uu[cname][1] += int((~scat).sum())
        npos += B * T

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    P = {c: round(v[0] / max(v[1], 1), 4) for c, v in pp.items()}
    Uu = {c: round(v[0] / max(v[1], 1), 4) for c, v in uu.items()}
    kl_full = res['full9']['kl']
    out = {'n_positions': npos, 'conds': res, 'freeze_null_kl': round(freeze_null_kl, 6),
           'scat50_patched': P, 'scat50_unpatched': Uu,
           'pred_a_leak_attn_borne': bool(Uu['scat50_attnfroze'] < 0.4 * Uu['scat50'] and
                                          P['scat50_attnfroze'] >= 0.85 * P['scat50']),
           'pred_b_manufacture_mlp_borne': bool((kl_full - res['full9_mlpfroze']['kl']) >= 0.15 * kl_full and
                                                (kl_full - res['full9_mlpfroze']['kl']) >
                                                (kl_full - res['full9_attnfroze']['kl'])),
           'pred_c_locality_survives_mlpfreeze': bool(P['scat50_mlpfroze'] >= 0.9 * P['scat50']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c, _, _, _ in CONDS:
        print(f"{c:>18}: KL {res[c]['kl']:7.3f} | align {res[c]['alignment']:+.4f}", flush=True)
    print(f"patched   {P}")
    print(f"unpatched {Uu}")
    print(f"freeze-null KL {out['freeze_null_kl']}")
    print(f"pred_a leak-attn {out['pred_a_leak_attn_borne']} | pred_b mfg-mlp {out['pred_b_manufacture_mlp_borne']} | pred_c locality {out['pred_c_locality_survives_mlpfreeze']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
