# mlp4_reads: WHAT DOES mlp4 READ? (User directive 2026-08-25: "content" modules are a
# residual label, not a structure claim — decompose them against what we DO understand.
# Start with mlp4: universal first mid, +.72 question-kit drop cost, dossier = "keep it".)
# The residual stream is additive: mlp4's input = wte + attn0..4 outputs + mlp0..3
# outputs. PATH-PATCH each upstream contribution INTO mlp4's input only (replace that
# component's contribution with its positional mean when computing mlp4's input; the
# component still writes normally to the stream everywhere else) and measure model dCE.
# This names mlp4's actual inputs in understood-component units. Also: full-input patch
# (all contributions meaned = mlp4 sees mean input) as the ceiling, and a class
# histogram of the top input's damage. NR=960, skip=5600.
#
# Registered predictions:
#   pred_a mlp4 reads the TOKEN TABLE tier most: the largest single upstream
#          contribution is mlp0 or mlp1 (dCE >= 1.5x the next-largest).
#   pred_b attention is a minority input: sum of attn0-4 patch damages <= 50% of the
#          summed all-component patch damages.
#   pred_c the top-3 upstream contributions account for >= 60% of the full-input-patch
#          ceiling (mlp4's diet is concentrated, not diffuse).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_reads_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
COMPONENTS = ['wte', 'attn0', 'attn1', 'attn2', 'attn3', 'attn4',
              'mlp0', 'mlp1', 'mlp2', 'mlp3']


@torch.no_grad()
def fwd_patch(idx, patch, cmeans, capture=None):
    """Full forward; at mlp4's input, subtract the patched components' contributions
    and add their means. patch: list of component names, or 'ALL', or None.
    capture: dict to accumulate per-component positional means."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    contrib = {'wte': x.clone()}
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        q = are(F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k = are(F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,)), cos, sin)
        q2 = are(F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        k2 = are(F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,)), cos, sin)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
            * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
        tril = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        pat = pat.masked_fill(~tril, 0.0)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv)
        yo = at.c_proj(y.reshape(B, T, D))
        # NOTE: xm mixes x and x0 by lambdas — contributions tracked in the x basis;
        # the lambda-mix means upstream contributions reach mlp4 scaled; we patch in
        # the x-accumulator, which is exact for the residual sum entering block 4.
        if L < 4:
            contrib[f'attn{L}'] = yo.clone()
        x = xm + yo
        if L == 4:
            contrib['attn4'] = yo.clone()
            if capture is not None:
                for c in COMPONENTS:
                    capture.setdefault(c, []).append(contrib[c].float().mean(0).cpu())
            if patch:
                names = COMPONENTS if patch == 'ALL' else patch
                delta = torch.zeros_like(x)
                for c in names:
                    delta = delta - contrib[c] + cmeans[c].to(x.dtype)
                mlp_in = F.rms_norm(x + delta, (D,))
            else:
                mlp_in = F.rms_norm(x, (D,))
            x = x + blk.mlp(mlp_in)
        else:
            mlp_out = blk.mlp(F.rms_norm(x, (D,)))
            if L < 4:
                contrib[f'mlp{L}'] = mlp_out.clone()
            x = x + mlp_out
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    capture = {}
    for i in range(0, NMEAN, 4):
        fwd_patch(MEANR[i:i + 4, :-1].to(DEV).contiguous(), None, None, capture=capture)
    cmeans = {c: torch.stack(capture[c]).mean(0).to(DEV) for c in COMPONENTS}
    print("component means cached (positional)", flush=True)

    def ce_run(patch):
        s_ = 0.0; n_ = 0; percls = {}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd_patch(idx, patch, cmeans).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            m_ = torch.ones_like(tg, dtype=torch.bool); m_[:, :64] = False
            s_ += float(ce[m_].sum()); n_ += int(m_.sum())
        return s_ / max(n_, 1)

    base = ce_run(None)
    print(f"base {base:.4f}", flush=True)
    res = {}
    for c in COMPONENTS:
        res[c] = round(ce_run([c]) - base, 4)
        print(f"patch {c}: dCE {res[c]:+.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    full = round(ce_run('ALL') - base, 4)
    print(f"patch ALL: dCE {full:+.4f}", flush=True)

    ranked = sorted(res, key=lambda c: -res[c])
    top, second = ranked[0], ranked[1]
    pa = top in ('mlp0', 'mlp1') and res[top] >= 1.5 * max(res[second], 1e-4)
    attn_sum = sum(res[c] for c in COMPONENTS if c.startswith('attn'))
    all_sum = sum(max(res[c], 0.0) for c in COMPONENTS)
    pb = attn_sum <= 0.50 * max(all_sum, 1e-4)
    top3 = sum(res[c] for c in ranked[:3])
    pc = top3 >= 0.60 * max(full, 1e-4)
    out = {'base': round(base, 4), 'patch_dce': res, 'full_input_patch': full,
           'ranked': ranked, 'attn_sum': round(attn_sum, 4),
           'top3_sum': round(top3, 4),
           'pred_a_table_tier_top': bool(pa), 'pred_b_attn_minority': bool(pb),
           'pred_c_concentrated': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ranked {ranked} | full {full}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
