"""WHY do later layers re-derive class? (user question). Two mechanisms are possible: (1) MAINTENANCE — the
residual is rescaled every block (x = λ0·x + λ1·x0, λ0<1) so class written at L0 DECAYS, and the token
embedding is RE-INJECTED every block (λ1·x0), so later layers can RE-COMPUTE the same token-class from the
always-available token to keep it fresh; (2) NEW CONTEXT-CLASS — later layers add class information that
depends on CONTEXT (the sequence), not derivable from the current token alone. Test which: for each layer,
predict its class-WRITE (its output projected on its class subspace) from (a) the CURRENT TOKEN embedding
alone vs (b) the FULL layer input. If token alone explains most of the middle's class-write, the middle is
RE-DERIVING the token-class (maintenance); the gap (full − token) is context-added class.

REGISTERED PREDICTIONS:
  (0) SANITY: mlp0 class-write is ~fully token-derived (R²_token high);
  (a) MIDDLE RE-DERIVES TOKEN-CLASS (maintenance): for middle layers R²_token is substantial (the class they
      write is largely re-derivable from the current token, i.e. maintenance of the same surface class against
      rescaling decay), with a smaller context gap -> "why re-derive" = the leaky rescaled residual + always-
      available re-injected token make each layer recompute/maintain class;
  (b) if R²_token is low for the middle (full ≫ token), the middle's class is CONTEXT-derived new info, not
      maintenance (report which)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'why_rederive_results.json'
NEVAL = 160; SEQ = 256; RANK = 8; RIDGE = 1e2
COMPS = [(0, 'mlp'), (2, 'mlp'), (5, 'mlp'), (8, 'mlp'), (11, 'mlp'), (14, 'mlp'), (16, 'mlp')]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


def class_sub(R, lab, r):
    g = R.mean(0, keepdim=True); rows = []
    for c in range(len(CLASSES)):
        mk = lab == c
        if mk.sum() < 5: continue
        rows.append(R[torch.tensor(mk, device=DEV)].mean(0)-g[0])
    return torch.linalg.svd(torch.stack(rows, 0), full_matrices=False)[2][:r].T.contiguous()


def r2(Xtr, Ytr, Xte, Yte):
    A = Xtr.T @ Xtr + RIDGE*torch.eye(Xtr.shape[1], device=DEV); W = torch.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xte @ W; ss_res = ((Yte-pred)**2).sum(); ss_tot = ((Yte-Yte.mean(0))**2).sum()
    return float(1 - ss_res/ss_tot)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    wte = m.transformer.wte.weight.detach().float()
    outc = {f"{k}{L}": [] for (L, k) in COMPS}; inc = {f"{k}{L}": [] for (L, k) in COMPS}; seqs = []; hs = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mkpost(tag):
            def h(mo, i_, o_): outc[tag].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mkpre(tag):
            def h(mo, a): inc[tag].append(a[0].detach().float().reshape(-1, D))
            return h
        hs.append(getattr(m.transformer.h[L], k).register_forward_hook(mkpost(tag)))
        hs.append(getattr(m.transformer.h[L], k).register_forward_pre_hook(mkpre(tag)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :SEQ].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    toks = np.concatenate([s.reshape(-1) for s in seqs]); lab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    tok_emb = wte[torch.tensor(toks, device=DEV)]
    n = len(toks); rng = np.random.RandomState(0); idx = rng.permutation(n); ntr = int(0.7*n); tr, te = idx[:ntr], idx[ntr:]
    out = {'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; O = torch.cat(outc[tag], 0); Xin = torch.cat(inc[tag], 0)
        U = class_sub(O, lab, RANK); cw = O @ U                       # class-write (its output's class part)
        r2_tok = r2(tok_emb[tr], cw[tr], tok_emb[te], cw[te])
        r2_in = r2(Xin[tr], cw[tr], Xin[te], cw[te])
        out['components'][tag] = {'r2_token_alone': round(r2_tok, 3), 'r2_full_input': round(r2_in, 3),
                                  'context_gap': round(r2_in - r2_tok, 3)}
        print(f"{tag:>6}: class-write R² from TOKEN alone {r2_tok:.3f} | from FULL input {r2_in:.3f} | context gap {r2_in-r2_tok:+.3f}", flush=True)
    mids = ['mlp5','mlp8','mlp11','mlp14']
    out['mean_r2_token_middle'] = round(float(np.mean([out['components'][t]['r2_token_alone'] for t in mids])), 3)
    out['mean_context_gap_middle'] = round(float(np.mean([out['components'][t]['context_gap'] for t in mids])), 3)
    out['pred_a_middle_maintains_token_class'] = bool(out['mean_r2_token_middle'] > 0.4 and out['mean_r2_token_middle'] > out['mean_context_gap_middle'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nMIDDLE class-write: R² from token alone {out['mean_r2_token_middle']} vs context gap {out['mean_context_gap_middle']}", flush=True)
    print(f"(a) middle re-derives/maintains the TOKEN class (vs new context-class): {out['pred_a_middle_maintains_token_class']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
