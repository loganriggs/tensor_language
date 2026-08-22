"""IS THE EMBEDDING THE SOURCE the model re-derives class+position from? (§823 follow-up).
§823 showed class steering works but needs amplification (α=8-32) at the components, because
the model keeps recomputing class+position from the current-token embedding that never leaves
the stream (FINDINGS 14). Conclusive test: steer the class content AT THE EMBEDDING (wte
output) instead of at the components. If the embedding is the source, editing it should steer
the prediction toward the target class at MUCH lower amplification than component steering
needed. Steer wte output: e_steer = e + α·proj_class( wte(B) − mean_wte ), all positions,
sweep small α; matched-norm random-direction null.

REGISTERED PREDICTIONS:
  (0) SANITY: α=0 reproduces normal KL; hook fires for α>0;
  (a) EMBEDDING IS SOURCE: embedding steering moves the prediction toward p_B (KL drops below
      normal) at LOW α (<=4), i.e. much less amplification than the components needed (§823:
      α=8-32), and >> matched random-direction null -> the embedding is the persistent source;
  (b) if embedding steering needs comparably high α or fails, the embedding is not privileged
      over the components as the re-derivation source;
  NULL: random-direction steer of matched per-α norm."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_steer_embedding_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64
SRC_TOKENS = [262, 257, 290]        # " the", " a", " and" (same as §823)
ALPHAS = [0.0, 1.0, 2.0, 4.0]
ST = {'on': False, 'mode': 'cp', 'alpha': 0.0, 'delta': None, 'randdir': None}


def emb_hook(mo, i_, o_):
    if not ST['on'] or ST['alpha'] == 0.0: return o_
    sh = o_.shape; e = o_.reshape(-1, D).float()
    d = ST['delta'] if ST['mode'] == 'cp' else ST['randdir']   # (1,D)
    e2 = e + ST['alpha'] * d
    return e2.reshape(sh).to(o_.dtype)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_emb(rows):
    caps = []; toks = []
    def h(mo, i_, o_): caps.append(o_.detach().float().reshape(-1, D))
    hh = m.transformer.wte.register_forward_hook(h)
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        toks.append(idx.cpu().numpy().reshape(-1))
    hh.remove(); return torch.cat(caps, 0), np.concatenate(toks)


def class_subspace(O, labels, r, g):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def avg_pred(rows, tok_id=None):
    ps = []
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1).reshape(-1, lg.shape[-1])
        if tok_id is None: ps.append(p.mean(0).cpu())
        else:
            mk = (idx.reshape(-1) == tok_id)
            if mk.any(): ps.append(p[mk].cpu())
    return (torch.cat(ps, 0) if tok_id is not None else torch.stack(ps, 0)).mean(0)


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p * (p/q).log()).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, toks = capture_emb(rows); gm = O.mean(0, keepdim=True)
    g = torch.Generator(device=DEV).manual_seed(0)
    U = class_subspace(O, toks, RTOK, gm)      # class subspace at the embedding
    tmean = {}
    for b in SRC_TOKENS:
        mk = toks == b
        if mk.sum() >= MINCOUNT: tmean[b] = O[mk].mean(0, keepdim=True).to(DEV)
    steer_hook = m.transformer.wte.register_forward_hook(emb_hook)
    ST['on'] = False; p_normal = avg_pred(rows)
    out = {'per_source': {}}
    for b in SRC_TOKENS:
        if b not in tmean: continue
        p_B = avg_pred(rows, b)
        dev = tmean[b] - gm.to(DEV); dcp = (dev @ U) @ U.T          # class-deviation at embedding
        ST['delta'] = dcp
        rd = torch.randn(1, D, generator=g, device=DEV); ST['randdir'] = rd / rd.norm() * dcp.norm()
        row = {'kl_normal': round(kl(p_normal, p_B), 4), 'cp': {}, 'rand': {}}
        for a in ALPHAS:
            if a == 0.0: continue
            ST['on'] = True; ST['alpha'] = a
            ST['mode'] = 'cp'; row['cp'][str(a)] = round(kl(avg_pred(rows), p_B), 4)
            ST['mode'] = 'rand'; row['rand'][str(a)] = round(kl(avg_pred(rows), p_B), 4)
            ST['on'] = False
        out['per_source'][str(b)] = row
        print(f"src {b}: normal {row['kl_normal']} | emb-cp-steer {row['cp']} | rand {row['rand']}", flush=True)
    steer_hook.remove()
    drops = []
    for b in SRC_TOKENS:
        if str(b) not in out['per_source']: continue
        r = out['per_source'][str(b)]; ba = min(r['cp'], key=lambda k: r['cp'][k])
        drops.append((r['kl_normal'] - r['cp'][ba], r['rand'][ba] - r['cp'][ba], float(ba)))
    md = float(np.mean([d[0] for d in drops])); mr = float(np.mean([d[1] for d in drops])); ma = float(np.mean([d[2] for d in drops]))
    out['mean_drop_vs_normal'] = round(md, 4); out['mean_cp_below_rand'] = round(mr, 4); out['mean_best_alpha'] = round(ma, 2)
    out['pred_a_embedding_source'] = bool(md > 0.2 and mr > 0.1 and ma <= 4.0); out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nmean KL drop vs normal {md:+.3f} | cp below rand {mr:+.3f} | mean best α {ma:.1f} (§823 components needed 8-32)", flush=True)
    print(f"(a) embedding is the low-α source: {out['pred_a_embedding_source']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
