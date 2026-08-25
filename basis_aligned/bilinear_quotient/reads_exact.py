# reads_exact: CORRECTED DIET INSTRUMENT (§1426 retraction: the naive ledger was stale
# under lambda-mixing). EXACT running decomposition: x = sum(contrib) maintained by
# rescaling every stored contribution by lambda0 at each block and crediting
# lambda1*x0 to the embedding ledger; RECONSTRUCTION ASSERT on the first batch
# (||x - sum contrib|| / ||x|| < 1e-4) before any patching. Re-runs the mlp4, mlp5,
# mlp6 diets. NR=960, skip=5600.
#
# Registered predictions:
#   pred_a mlp4's §1422 ranking reproduces under the exact ledger: mlp0 is its top
#          input and mlp1's patch dCE <= 0.05.
#   pred_b the instrument is sane: mlp5's full-input patch lands within 3x of its
#          module-ladder stake (|full| <= 0.35).
#   pred_c the code chains: for mlp5 or mlp6, an upstream mid (mlp4 or mlp5) ranks
#          top-3 among that target's inputs.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'reads_exact_results.json'
NMEAN = 24; NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
def comps_for(LT):
    return ['wte'] + [f'attn{i}' for i in range(LT + 1)] + \
        [f'mlp{i}' for i in range(LT)]


SELFTEST = {'done': False}


@torch.no_grad()
def fwd_patch(idx, LT, patch, cmeans, capture=None):
    """Full forward with an EXACT running decomposition of the residual stream up to
    block LT: every stored contribution is rescaled by lambda0 at each block and
    lambda1*x0 is credited to the embedding ledger, so sum(contrib) == x exactly."""
    COMP = comps_for(LT)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    contrib = {'wte': x.clone()}
    for L, blk in enumerate(H):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if L <= LT:
            for c in contrib:
                contrib[c] = blk.lambdas[0] * contrib[c]
            contrib['wte'] = contrib['wte'] + blk.lambdas[1] * x0
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
            contrib[f'attn{L}'] = yo.clone()
        x = xm + yo
        if L == LT:
            if not SELFTEST['done']:
                tot = sum(contrib.values())
                rel = float((x.float() - tot.float()).norm() / x.float().norm())
                assert rel < 1e-3, f"ledger reconstruction failed: rel={rel}"
                SELFTEST['done'] = True
                print(f"ledger self-test passed (rel {rel:.2e})", flush=True)
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
                contrib[f'mlp{L}'] = mlp_out.clone()
            x = x + mlp_out
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    all_out = {}
    for LT in (4, 5, 6):
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

    r4, r5, r6 = all_out['mlp4'], all_out['mlp5'], all_out['mlp6']
    pa = r4['ranked'][0] == 'mlp0' and r4['patch_dce']['mlp1'] <= 0.05
    pb = abs(r5['full_input_patch']) <= 0.35
    pc = any(mid in r['ranked'][:3] for r, mids in ((r5, ('mlp4',)), (r6, ('mlp4', 'mlp5')))
             for mid in mids)
    out = {'targets': all_out,
           'pred_a_mlp4_ranking_reproduces': bool(pa), 'pred_b_instrument_sane': bool(pb),
           'pred_c_code_chains': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
