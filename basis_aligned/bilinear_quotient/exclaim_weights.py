# exclaim_weights: completes the §1315-19 thread — is the "!"-pair's criterion
# embedding-native or stream-computed? Raw rms(wte) codes of "!"-containing tokens as
# keys vs 512 ordinary in-corpus tokens under 17.2 and 17.3's q/k pipelines (the §1238
# instrument at L17). Control head 17.0.
#
# Registered predictions:
#   pred_a STREAM-COMPUTED: 17.2 and 17.3 weights ratios both <= 2.0 (late register
#          heads read processed state, like 13.8/10.5/8.1, unlike 8.7).
#   pred_b CONTROL SIMILAR: 17.0's ratio within [0.5, 2.0] (nothing special anywhere).
#   pred_c SANITY: the matcher 2.5 run on the same token sets DOES show structure only on
#          identity (its "!"-vs-ordinary ratio also <= 2.0 — class labels are not what
#          matchers see; guards the instrument).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'exclaim_weights_results.json'
QPOS = 200; KPOS = 180
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h


@torch.no_grad()
def ratio(L, h, keys_special, keys_ord, queries):
    at = H[L].attn
    def codes(t):
        return F.rms_norm(m.transformer.wte(t), (D,))
    dummy = torch.zeros(1, QPOS + 1, 9, 128, device=DEV)
    cos_t, sin_t = at.rotary(dummy)
    def pipe(lin, x, pos):
        z = F.rms_norm(lin(x).view(-1, 9, 128), (128,)).view(1, -1, 9, 128)
        return are(z, cos_t[:, pos:pos + 1], sin_t[:, pos:pos + 1])[0, :, h]
    xq = codes(queries); xs = codes(keys_special); xo = codes(keys_ord)
    q1 = pipe(at.c_q, xq, QPOS); q2 = pipe(at.c_q2, xq, QPOS)
    ks1 = pipe(at.c_k, xs, KPOS); ks2 = pipe(at.c_k2, xs, KPOS)
    ko1 = pipe(at.c_k, xo, KPOS); ko2 = pipe(at.c_k2, xo, KPOS)
    ss = (q1.float() @ ks1.float().T / 128) * (q2.float() @ ks2.float().T / 128)
    so = (q1.float() @ ko1.float().T / 128) * (q2.float() @ ko2.float().T / 128)
    return float(ss.abs().mean() / so.abs().mean().clamp_min(1e-9))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    ex = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '!' in d:
            ex.add(tok)
    rows = cl.fineweb_rows(8)[:, :256].reshape(-1)
    uniq = torch.unique(rows)
    ex_ids = torch.tensor(sorted(ex))
    ordinary = uniq[~torch.isin(uniq, ex_ids)]
    g = torch.Generator().manual_seed(37)
    ord_sel = ordinary[torch.randperm(len(ordinary), generator=g)[:512]].to(DEV)
    q_sel = ordinary[torch.randperm(len(ordinary), generator=g)[:512]].to(DEV)
    ex_sel = ex_ids[:min(len(ex_ids), 256)].to(DEV)
    res = {}
    for name, (L, h) in (('17.2', (17, 2)), ('17.3', (17, 3)), ('17.0', (17, 0)),
                         ('2.5', (2, 5))):
        res[name] = round(ratio(L, h, ex_sel, ord_sel, q_sel), 3)
        print(f"{name}: ratio {res[name]}", flush=True)
    pa = res['17.2'] <= 2.0 and res['17.3'] <= 2.0
    pb = 0.5 <= res['17.0'] <= 2.0
    pc = res['2.5'] <= 2.0
    out = {'ratios': res, 'pred_a_stream_computed': bool(pa),
           'pred_b_control_similar': bool(pb), 'pred_c_matcher_guard': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a stream {pa} | pred_b ctrl {pb} | pred_c guard {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
