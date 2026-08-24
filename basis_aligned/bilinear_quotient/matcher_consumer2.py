# matcher_consumer2: the §1241 instrument, null-controlled. v1's keep-until-L-then-subtract
# produced a non-monotone curve (cut4 +0.03, cut6 +3.80, cut9 +0.53) that violates clean-
# removal semantics — quarantined. v2 adds the discriminating controls:
#   NULL: identical design but subtracting NON-STATION heads' writes (2.1 + 3.3 — same
#         layers, no copy role) at the same cuts. If null-cut6 also spikes, the v1 spike is
#         subtraction-inconsistency (artifact); if null is flat, the spike is real.
#   ZEROWRITE anchor: matcher head outputs zeroed AT THE SOURCE (never written) — the clean
#         removal whose cost is interpretable (§1228-family operation).
#
# Registered predictions:
#   pred_a ARTIFACT VERDICT: null-cut6 >= 0.5 x matcher-cut6 (the spike replicates with
#          no-role heads -> v1's curve carries no consumption information; the §1241
#          quarantine becomes permanent).
#   pred_b ZEROWRITE INTERPRETABLE: cost(zerowrite) in [0.5, 3.0] (nonzero, §1207-order).
#   pred_c SANITY: base = true model (±0.005); matcher cut4/cut6 replicate v1 (±0.1).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'matcher_consumer2_results.json'
NR = 24; QSTART = 128
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
MATCHERS = {2: [5], 3: [8]}
NULLH = {2: [1], 3: [3]}


@torch.no_grad()
def forward_cut(idx, spec):
    lcut, HSET = spec
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    contribs = []
    for L, blk in enumerate(m.transformer.h):
        if lcut != 'zero' and L == lcut:
            for c in contribs:
                x = x - c
            contribs = []
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
        if L in HSET and lcut == 'zero':
            for h in HSET[L]:
                y[:, :, h, :] = 0.0
        elif L in HSET and lcut is not None and L < lcut:
            for h in HSET[L]:
                yh = torch.zeros_like(y)
                yh[:, :, h, :] = y[:, :, h, :]
                contribs.append(at.c_proj(yh.reshape(B, T, D)))
        x = xm + at.c_proj(y.reshape(B, T, D))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous().clone()
    ROWS[:, 128:256] = ROWS[:, 0:128]
    qp = torch.arange(QSTART, T, device=DEV)
    CUTS = {'base': (None, MATCHERS), 'cut4': (4, MATCHERS), 'cut6': (6, MATCHERS), 'cut9': (9, MATCHERS),
             'null4': (4, NULLH), 'null6': (6, NULLH), 'null9': (9, NULLH), 'zerowrite': ('zero', MATCHERS)}
    ce = {c: 0.0 for c in CUTS}; ce_true = 0.0; n = 0
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lt = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        ce_true += float(F.cross_entropy(lt[:, qp].reshape(-1, lt.shape[-1]), tgt[:, qp].reshape(-1), reduction='sum'))
        for cname, sp in CUTS.items():
            lo = forward_cut(idx, sp).float()
            ce[cname] += float(F.cross_entropy(lo[:, qp].reshape(-1, lo.shape[-1]),
                                               tgt[:, qp].reshape(-1), reduction='sum'))
        n += 4 * len(qp)
    CE = {c: round(v / n, 4) for c, v in ce.items()}
    CE['true_model'] = round(ce_true / n, 4)
    cost = {c: round(CE[c] - CE['base'], 4) for c in CUTS if c != 'base'}
    out = {'n_rows': NR, 'ce': CE, 'cost_vs_base': cost,
           'sanity': bool(abs(CE['base'] - CE['true_model']) <= 0.005),
           'pred_a_artifact': bool(cost['null6'] >= 0.5 * max(cost['cut6'], 1e-6)),
           'pred_b_zerowrite': bool(0.5 <= cost['zerowrite'] <= 3.0),
           'pred_c_replicates': bool(abs(cost['cut4'] - 0.0337) <= 0.1 and
                                     abs(cost['cut6'] - 3.7972) <= 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"costs {cost}")
    print(f"sanity {out['sanity']} | pred_a artifact {out['pred_a_artifact']} | pred_b zw {out['pred_b_zerowrite']} | pred_c repl {out['pred_c_replicates']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
