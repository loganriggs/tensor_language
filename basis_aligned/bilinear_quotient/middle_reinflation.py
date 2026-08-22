"""WHAT DOES THE MIDDLE RE-INFLATE? (§857 follow-up; finishes the middle's characterization). §857: the
residual-stream token geometry collapses to eff-dim 23 at L5 then RE-EXPANDS to the stack peak 51 by L9,
while relative similarities barely change (RSA 0.95-0.98). What feature does the middle add that expands
the dimension? Decode, from the residual stream after L5 vs after L9: current grammatical class, fine
token-identity, previous token, position. Whatever decodes BETTER at L9 than L5 is what the middle
re-inflates.

REGISTERED PREDICTIONS:
  (0) SANITY: current-token decodes high at both (embedding always present);
  (a) report which feature GAINS most L5->L9 (that is the middle's re-inflation content) — candidates:
      finer token-identity, previous token, position, or class; a gain in token-fine/prev would mean the
      middle re-introduces lexical/context detail the L5 collapse removed;
  (b) if nothing decodable gains, the re-inflation is in directions none of these features capture
      (content/other), an honest 'not captured'."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_reinflation_results.json'
NEVAL = 240; TOPV = 200
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, layers):
    outs = {L: [] for L in layers}; seqs = []; hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): outs[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    return {L: torch.cat(outs[L], 0) for L in layers}, np.concatenate(seqs, 0)


def acc(Ft, y, valid, ncls, seed=0):
    idx = np.where(valid)[0]; rng = np.random.RandomState(seed); rng.shuffle(idx)
    ntr = int(0.7*len(idx)); tr, te = idx[:ntr], idx[ntr:]
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = Ft[tr].T @ Ft[tr] + 1e2*torch.eye(D, device=DEV); Wp = torch.linalg.solve(A, Ft[tr].T @ Y)
    return float(((Ft[te] @ Wp).argmax(1).cpu().numpy() == y[te]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    reps, seqs = capture(rows, [5, 9]); nseq = seqs.shape[0]
    cur = seqs.reshape(-1); prev = np.full_like(seqs, -1); prev[:, 1:] = seqs[:, :-1]; prev = prev.reshape(-1)
    pos = np.broadcast_to(np.arange(seqs.shape[1]), seqs.shape).reshape(-1); posbin = (pos//32).astype(np.int64)
    curcls = np.array([CLASSES.index(classify(d(int(t)))) for t in cur])
    uniq, cnts = np.unique(cur, return_counts=True); topv = set(uniq[np.argsort(-cnts)[:TOPV]].tolist())
    remap = {t: i for i, t in enumerate(sorted(topv))}; lbl = lambda a: np.array([remap.get(int(t), -1) for t in a])
    cur_l, prev_l = lbl(cur), lbl(prev)
    res = {}
    for L in [5, 9]:
        Ft = reps[L]
        res[f'L{L}'] = {
            'class': round(acc(Ft, curcls, np.ones_like(curcls, bool), len(CLASSES)), 3),
            'token_fine': round(acc(Ft, cur_l, cur_l >= 0, TOPV), 3),
            'prev_token': round(acc(Ft, prev_l, prev_l >= 0, TOPV), 3),
            'position': round(acc(Ft, posbin, np.ones_like(posbin, bool), int(posbin.max())+1), 3),
        }
        print(f"L{L}: class {res[f'L{L}']['class']} | token {res[f'L{L}']['token_fine']} | prev {res[f'L{L}']['prev_token']} | pos {res[f'L{L}']['position']}", flush=True)
    gains = {k: round(res['L9'][k] - res['L5'][k], 3) for k in res['L5']}
    out = {'decode': res, 'gain_L5_to_L9': gains, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"GAIN L5->L9 (what the middle re-inflates): {gains}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
