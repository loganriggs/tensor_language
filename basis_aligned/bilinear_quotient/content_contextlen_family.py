"""GENERALIZE the content receptive-field finding (§995: content is broad/long-range and ~4x more context-hungry than
grammar) across the tensor-language family, via an ARCHITECTURE-AGNOSTIC instrument (no monkeypatch): feed only the
last K tokens before a fixed query position and measure the chain-rule CE split at that position; sweep K. Content
(within-CE) should keep improving with K (broad), grammar (class-CE) should saturate early (local), in EVERY model
(bilin18, bilin12, swiglu18).

REGISTERED PREDICTIONS:
  (0) SANITY: K=full reproduces the model's normal CE at the query position; CE decreases (or is flat) as K grows.
  (a) CONTENT LONG-RANGE / GRAMMAR LOCAL, FAMILY-GENERAL: in EVERY model, the within-CE (content) improvement from
      K=1 to K=full is much larger than the class-CE (grammar) improvement (content/grammar context-hunger ratio > 2),
      and within-CE keeps dropping past K=8 while class-CE saturates early -> content broad, grammar local, in all
      architectures incl swiglu;
  (b) report per-model within-CE and class-CE at each K + the K=1->full content vs grammar drop and their ratio."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import m as BILIN, DEV
import census_lib as cl
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_contextlen_family_results.json'
NEVAL = 300; SEQ = 256; QUERY = 200; KS = [1, 2, 4, 8, 16, 32, 64, 128, QUERY]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}


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


def forward_logits(mdl, idx, Dm):
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(mdl.lm_head(F.rms_norm(x, (Dm,)))/30.0)


@torch.no_grad()
def ce_at_query(mdl, rows, cidx, C, Dm, K):
    # feed the last K tokens before QUERY; predict the token at QUERY; CE + chain-rule split at that position
    tot = 0.0; totc = 0.0; n = 0; Cmat = F.one_hot(cidx, C).float()
    for i in range(0, rows.shape[0], 16):
        bb = rows[i:i+16].to(DEV)
        idx = bb[:, QUERY-K:QUERY].contiguous()          # last K tokens
        tgt = bb[:, QUERY].contiguous()                   # the query's next token
        lg = forward_logits(mdl, idx, Dm).float()[:, -1]  # logits at the last fed position
        lp = F.log_softmax(lg, -1)
        lp_tok = lp[torch.arange(tgt.shape[0], device=DEV), tgt]
        pcls = (lp.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tgt.shape[0], device=DEV), cidx[tgt]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tgt.shape[0]
    full = tot/n; classce = totc/n
    return {'full': round(full, 4), 'class': round(classce, 4), 'within': round(full-classce, 4)}


@torch.no_grad()
def run_model(mdl, rows, d, tag):
    Dm = mdl.transformer.wte.weight.shape[1]; V = int(mdl.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(rows.cpu().numpy().reshape(-1)):
        if tid < V: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    byK = {}
    for K in KS:
        byK[str(K)] = ce_at_query(mdl, rows, cidx, C, Dm, K)
        print(f"  [{tag}] K={K}: within {byK[str(K)]['within']} class {byK[str(K)]['class']}", flush=True)
    within_drop = round(byK['1']['within'] - byK[str(QUERY)]['within'], 4)
    class_drop = round(byK['1']['class'] - byK[str(QUERY)]['class'], 4)
    ratio = round(within_drop/max(abs(class_drop), 1e-6), 2)
    res = {'by_K': byK, 'within_drop_1_to_full': within_drop, 'class_drop_1_to_full': class_drop, 'content_grammar_hunger_ratio': ratio}
    print(f"{tag}: content-drop {within_drop} vs grammar-drop {class_drop} (ratio {ratio})", flush=True)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous(); d = dec()
    out = {'models': {}}
    out['models']['bilin18'] = run_model(BILIN, rows, d, 'bilin18')
    for short in ['bilin12', 'swiglu18']:
        try:
            mdl, cfg = load_elriggs(short); mdl = mdl.to(DEV).eval()
            out['models'][short] = run_model(mdl, rows, d, short); del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = [k for k in out['models'] if 'content_grammar_hunger_ratio' in out['models'][k]]
    out['pred_a_content_broad_general'] = bool(len(ok) >= 2 and all(out['models'][k]['content_grammar_hunger_ratio'] > 2 for k in ok))
    out['swiglu_ratio'] = out['models'].get('swiglu18', {}).get('content_grammar_hunger_ratio')
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a content-broad/grammar-local family-general {out['pred_a_content_broad_general']} | swiglu ratio {out['swiglu_ratio']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
