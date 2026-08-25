# mids_reads: DIET-NAMING FOR mlp5 AND mlp6 (§1422 method walked up the stack; §1422:
# mlp4 reads {mlp0 .674, mlp3 .373, mlp2 .121}, mlp1~0, attn 5%). Path-patch each
# upstream contribution into the TARGET mlp's input only; components for target L:
# wte, attn0..L, mlp0..L-1. NR=960, skip=5600.
#
# Registered predictions:
#   pred_a the code CHAINS: mlp4 is a top-2 input for mlp5 (by patch dCE).
#   pred_b attention stays minority for BOTH targets (attn sum <= 50% of all-sum).
#   pred_c both diets concentrated: top-3 inputs >= 60% of that target's
#          full-input-patch ceiling.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mids_reads_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
def comps_for(LT):
    return ['wte'] + [f'attn{i}' for i in range(LT + 1)] + \
        [f'mlp{i}' for i in range(LT)]


@torch.no_grad()
def fwd_patch(idx, LT, patch, cmeans, capture=None):
    """Full forward; at mlp{LT}'s input, subtract the patched components' contributions
    and add their means. patch: list of component names, or 'ALL', or None."""
    COMP = comps_for(LT)
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
        if L <= LT:
            contrib[f'attn{L}'] = yo
        x = xm + yo
        if L == LT:
            if capture is not None:
                for c in COMP:
                    capture.setdefault(c, []).append(contrib[c].float().mean(0).cpu())
            if patch:
                names = COMP if patch == 'ALL' else patch
                delta = torch.zeros_like(x)
                for c in names:
                    delta = delta - contrib[c] + cmeans[c].to(x.dtype)
                mlp_in = F.rms_norm(x + delta, (D,))
            else:
                mlp_in = F.rms_norm(x, (D,))
            x = x + blk.mlp(mlp_in)
        else:
            mlp_out = blk.mlp(F.rms_norm(x, (D,)))
            if L < LT:
                contrib[f'mlp{L}'] = mlp_out
            x = x + mlp_out
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    all_out = {}
    for LT in (5, 6):
        COMP = comps_for(LT)
        capture = {}
        for i in range(0, NMEAN, 4):
            fwd_patch(MEANR[i:i + 4, :-1].to(DEV).contiguous(), LT, None, None,
                      capture=capture)
        cmeans = {c: torch.stack(capture[c]).mean(0).to(DEV) for c in COMP}

        def ce_run(patch):
            s_ = 0.0; n_ = 0
            for i in range(0, NR, 8):
                bb = EVR[i:i + 8].to(DEV)
                idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd_patch(idx, LT, patch, cmeans).float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                     reduction='none').view(tg.shape)
                m_ = torch.ones_like(tg, dtype=torch.bool); m_[:, :64] = False
                s_ += float(ce[m_].sum()); n_ += int(m_.sum())
            return s_ / max(n_, 1)

        base = ce_run(None)
        res = {}
        for c in COMP:
            res[c] = round(ce_run([c]) - base, 4)
            print(f"mlp{LT} patch {c}: dCE {res[c]:+.4f}", flush=True)
            json.dump({'partial': True, 'all': all_out, 'cur': res}, open(OUT, 'w'), indent=1)
        full = round(ce_run('ALL') - base, 4)
        ranked = sorted(res, key=lambda c: -res[c])
        attn_sum = sum(res[c] for c in COMP if c.startswith('attn'))
        all_sum = sum(max(res[c], 0.0) for c in COMP)
        top3 = sum(res[c] for c in ranked[:3])
        all_out[f'mlp{LT}'] = {'base': round(base, 4), 'patch_dce': res,
                               'full_input_patch': full, 'ranked': ranked,
                               'attn_sum': round(attn_sum, 4),
                               'all_sum': round(all_sum, 4), 'top3_sum': round(top3, 4)}
        print(f"mlp{LT}: ranked {ranked[:4]} full {full}", flush=True)

    r5, r6 = all_out['mlp5'], all_out['mlp6']
    pa = 'mlp4' in r5['ranked'][:2]
    pb = all(r['attn_sum'] <= 0.50 * max(r['all_sum'], 1e-4) for r in (r5, r6))
    pc = all(r['top3_sum'] >= 0.60 * max(r['full_input_patch'], 1e-4) for r in (r5, r6))
    out = {'targets': all_out,
           'pred_a_code_chains': bool(pa), 'pred_b_attn_minority': bool(pb),
           'pred_c_concentrated': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
