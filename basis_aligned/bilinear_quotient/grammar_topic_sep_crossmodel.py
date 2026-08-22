"""IS THE TWO-CHANNEL SEPARABILITY UNIVERSAL? §920: in bilin18 the grammar (next-class) and topic subspaces are
ORTHOGONAL (overlap 0.009 ≈ chance). Is that a general LM property (like the loss budget §880 and the 23/77
split §831)? Measure the grammar-subspace vs topic-subspace overlap in GPT-2 and GPT-2-large at a late layer,
vs chance.

CAVEAT: GPT-2 is WebText-trained (slightly OOD on FineWeb); the orthogonality (a within-model geometric fact)
is robust to that.

REGISTERED PREDICTIONS:
  (0) SANITY: both subspaces decode their target above chance in each model;
  (a) UNIVERSAL: grammar and topic subspaces are near-ORTHOGONAL (overlap ~ chance) in BOTH GPT-2 models, as in
      bilin18 -> the two-channel (separable grammar/content) structure is a general LM property;
  (b) if overlap is high in GPT-2, the orthogonality is bilin18-specific (report)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_topic_sep_crossmodel_results.json'
NEVAL = 160; SEQ = 256; K = 12; RCLASS = 8; RTOPIC = 11
MODELS = ['gpt2', 'gpt2-large']
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


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def mean_subspace(X, labels, r):
    Dd = X.shape[1]; g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


@torch.no_grad()
def run(mid, blocks, S, d):
    mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval()
    Dm = mdl.config.n_embd if hasattr(mdl.config, 'n_embd') else mdl.config.hidden_size
    L = int(0.8 * mdl.config.n_layer)  # late layer ~80% depth
    reps = []
    def h(mo, i_, o_): reps.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, Dm))
    hh = mdl.transformer.h[L].register_forward_hook(h)
    for i in range(0, blocks.shape[0], 4): mdl(blocks[i:i+4].to(DEV)[:, :-1])
    hh.remove(); R = torch.cat(reps, 0)
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (S.shape[0], SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Uclass, _ = mean_subspace(R, nxtcls, RCLASS)
    Utok, g = mean_subspace(R, toks, 64); Upos, _ = mean_subspace(R, pos.astype(np.int64), 32)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :96].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic, RTOPIC)
    r = min(Uclass.shape[1], Utopic.shape[1]); overlap = float((Uclass.T @ Utopic).pow(2).sum()/r); chance = RCLASS/Dm
    del mdl; torch.cuda.empty_cache()
    return {'layer': L, 'overlap': round(overlap, 4), 'chance': round(chance, 4), 'ratio': round(overlap/chance, 2)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy()
    out = {'bilin18_ref_overlap': 0.009, 'bilin18_ref_chance': 0.007, 'models': {}}
    for mid in MODELS:
        print(f"loading {mid}...", flush=True); r = run(mid, blocks, S, d); out['models'][mid] = r
        print(f"{mid}: grammar-topic subspace overlap {r['overlap']} vs chance {r['chance']} (ratio {r['ratio']}x) @ L{r['layer']}", flush=True)
    out['pred_a_universal_orthogonal'] = bool(all(out['models'][mid]['ratio'] < 3 for mid in MODELS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) two-channel separability universal (grammar⊥topic in all): {out['pred_a_universal_orthogonal']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
