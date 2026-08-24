# quote_mechanism: HOW does 13.8 know a delimiter is open? (§1271 next step.) Two cheap
# instruments from the validated kit:
#  (1) OFFSET/READ profile: on real prose, where does 13.8 attend AT quote-close targets vs
#      elsewhere? If it reads THE OPENER's position (matcher-style), target-position pattern
#      mass should concentrate on opener tokens; measured as the share of |pattern| mass on
#      quote/bracket-token key positions, target vs elsewhere.
#  (2) WEIGHTS-ONLY criterion (§1238 instrument): raw rms(wte) codes of quote/open-bracket
#      tokens through 13.8's q/k pipelines vs 512 ordinary tokens — does the head's bilinear
#      form single out delimiter keys structurally?
#
# Registered predictions:
#   pred_a OPENER-READER: at target positions, >= 3x more of 13.8's |pattern| mass sits on
#          delimiter-token keys than at elsewhere positions (share ratio >= 3).
#   pred_b WEIGHTS SEE DELIMITERS: mean |score| of (any-query x delimiter-key) pairs >= 2x
#          (any-query x ordinary-key) pairs under 13.8's pipelines on raw codes.
#   pred_c CONTROL HEAD FLAT: the same two measurements on inert late head 13.1 (near-zero
#          quote damage, §1271) show ratios <= 1.5.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'quote_mechanism_results.json'
NR = 96; QPOS = 200; KPOS = 72
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h


@torch.no_grad()
def pattern13(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
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
        if L == 13:
            return pat.abs()
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return None


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def ids_for(chars):
        out = set()
        for t in range(50257):
            try:
                d = enc.decode([t])
            except Exception:
                continue
            if any(c in d for c in chars):
                out.add(t)
        return torch.tensor(sorted(out), device=DEV)
    dl_ids = ids_for(['"', '(', '[', '{', ')', ']', '}'])
    print(f"delimiter ids: {len(dl_ids)}", flush=True)

    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    tgt_all = ROWS[:, 1:]
    isq = torch.isin(tgt_all, dl_ids.cpu())
    ctx = torch.zeros_like(isq)
    tok_isq = torch.isin(ROWS[:, :-1], dl_ids.cpu())
    for w in range(1, 65):
        sh = torch.zeros_like(tok_isq)
        sh[:, w:] = tok_isq[:, :-w]
        ctx |= sh
    TGT = isq & ctx; TGT[:, :64] = False

    shares = {8: {'tar': [], 'els': []}, 1: {'tar': [], 'els': []}}
    for i in range(0, 48, 4):
        idx = ROWS[i:i + 4, :-1].to(DEV).contiguous()
        pat = pattern13(idx)                                   # (B,9,T,T) abs
        isdl = torch.isin(idx, dl_ids)                         # (B,T) key-token is delimiter
        for h in (8, 1):
            p = pat[:, h]                                      # (B,T,T)
            mdl_mass = (p * isdl.unsqueeze(1).float()).sum(-1)
            tot = p.sum(-1).clamp_min(1e-9)
            share = (mdl_mass / tot)                           # (B,T)
            tm = TGT[i:i + 4].to(DEV)
            em = ~tm; em[:, :64] = False
            shares[h]['tar'].append(share[tm].cpu()); shares[h]['els'].append(share[em].cpu())
    res_pat = {}
    for h in (8, 1):
        st = torch.cat(shares[h]['tar']).mean(); se = torch.cat(shares[h]['els']).mean()
        res_pat[h] = {'target_share': round(float(st), 4), 'else_share': round(float(se), 4),
                      'ratio': round(float(st / se.clamp_min(1e-6)), 2)}
    print(f"pattern shares {res_pat}", flush=True)

    # weights-only: delimiter keys vs ordinary keys under 13.8 and 13.1
    rows_all = cl.fineweb_rows(8)[:, :256].reshape(-1)
    uniq = torch.unique(rows_all)
    ordinary = uniq[~torch.isin(uniq, dl_ids.cpu())]
    g = torch.Generator().manual_seed(4)
    ord_sel = ordinary[torch.randperm(len(ordinary), generator=g)[:512]].to(DEV)
    dl_sel = dl_ids[:min(len(dl_ids), 256)]
    at = H[13].attn
    res_w = {}
    for h in (8, 1):
        x_q = F.rms_norm(m.transformer.wte(ord_sel), (D,))
        x_kd = F.rms_norm(m.transformer.wte(dl_sel), (D,))
        x_ko = F.rms_norm(m.transformer.wte(ord_sel), (D,))
        dummy = torch.zeros(1, QPOS + 1, 9, 128, device=DEV)
        cos_t, sin_t = at.rotary(dummy)
        def pipe(lin, x, pos):
            z = F.rms_norm(lin(x).view(-1, 9, 128), (128,)).view(1, -1, 9, 128)
            return are(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])[0, :, h]
        q1 = pipe(at.c_q, x_q, QPOS); q2 = pipe(at.c_q2, x_q, QPOS)
        kd1 = pipe(at.c_k, x_kd, KPOS); kd2 = pipe(at.c_k2, x_kd, KPOS)
        ko1 = pipe(at.c_k, x_ko, KPOS); ko2 = pipe(at.c_k2, x_ko, KPOS)
        sd = (torch.einsum('qd,kd->qk', q1.float(), kd1.float()) / 128) * \
             (torch.einsum('qd,kd->qk', q2.float(), kd2.float()) / 128)
        so = (torch.einsum('qd,kd->qk', q1.float(), ko1.float()) / 128) * \
             (torch.einsum('qd,kd->qk', q2.float(), ko2.float()) / 128)
        res_w[h] = {'dl_absmean': round(float(sd.abs().mean()), 5),
                    'ord_absmean': round(float(so.abs().mean()), 5),
                    'ratio': round(float(sd.abs().mean() / so.abs().mean().clamp_min(1e-9)), 2)}
    print(f"weights scores {res_w}", flush=True)

    out = {'n_rows': NR, 'pattern_shares': {str(k): v for k, v in res_pat.items()},
           'weights_scores': {str(k): v for k, v in res_w.items()},
           'pred_a_opener_reader': bool(res_pat[8]['ratio'] >= 3),
           'pred_b_weights_see': bool(res_w[8]['ratio'] >= 2),
           'pred_c_control_flat': bool(res_pat[1]['ratio'] <= 1.5 and res_w[1]['ratio'] <= 1.5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a opener {out['pred_a_opener_reader']} | pred_b weights {out['pred_b_weights_see']} | pred_c ctrl {out['pred_c_control_flat']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
