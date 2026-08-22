"""Does the ERROR STRUCTURE (§972/§973: bilin18's mistakes are CONTENT mistakes — it either picks the wrong content
word or HEDGES to a function word) generalize across the model FAMILY? Test swiglu18 and bilin12 (+ bilin18 ref).
Uses TRANSFORM-INVARIANT signals only (argmax + token frequencies), so it is robust to per-model output clamps and
is NOT confound-prone (unlike the §974 entropy metric). Per model report:
  - error partition: HIT / CONTENT-ERROR (top-1 class == true class, wrong token) / GRAMMAR-ERROR (top-1 class !=
    true class);
  - hedging signature on GRAMMAR-ERROR positions: top-1 function-class fraction (vs corpus base), and mean
    log-frequency of the TRUE token vs on HITs (rarer = content-hard).

REGISTERED PREDICTIONS:
  (0) SANITY: HIT rate above chance in each model.
  (a) FAMILY-WIDE CONTENT ERRORS + HEDGING: in swiglu18 and bilin12, as in bilin18 (§973), on GRAMMAR-ERROR
      positions the top-1 is a FUNCTION word far above the corpus base rate AND the true token is rarer than on
      hits -> the "mistakes are content mistakes, hedge to function words" structure is family-wide;
  (b) report the partition + hedging signature per model."""
import json, time, sys, torch
import numpy as np
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_family_errors_results.json'
NEVAL = 160; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}
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


@torch.no_grad()
def logits(mdl, idx, Dm):
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    return mdl.lm_head(F.rms_norm(x, (Dm,)))  # argmax is invariant to the monotone 30*tanh clamp


@torch.no_grad()
def run(mdl, blocks, S, Dm, cidx, is_func, logfreq):
    nb = blocks.shape[0]
    hit=cont=gram=0; n=0; g_top1func=0; g_n=0; g_true_lf=[]; h_true_lf=[]
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = logits(mdl, idx, Dm).float(); tf = tgt.reshape(-1); pred = lg.reshape(-1, lg.shape[-1]).argmax(1)
        tc = cidx[tf]; pc = cidx[pred]
        is_hit = pred == tf; is_cont = (~is_hit) & (pc == tc); is_gram = (~is_hit) & (pc != tc)
        hit += int(is_hit.sum()); cont += int(is_cont.sum()); gram += int(is_gram.sum()); n += tf.shape[0]
        gm = is_gram.cpu().numpy(); pred_np = pred.cpu().numpy(); tf_np = tf.cpu().numpy(); hm = is_hit.cpu().numpy()
        g_top1func += int(is_func[pred_np[gm]].sum()); g_n += int(gm.sum())
        g_true_lf.append(logfreq[tf_np[gm]]); h_true_lf.append(logfreq[tf_np[hm]])
    g_lf = float(np.concatenate(g_true_lf).mean()) if g_n else 0.0
    h_lf = float(np.concatenate(h_true_lf).mean())
    return {'hit_frac': round(hit/n,3), 'content_err_frac': round(cont/n,3), 'grammar_err_frac': round(gram/n,3),
            'gram_top1_func_frac': round(g_top1func/max(g_n,1),3), 'gram_true_logfreq': round(g_lf,3),
            'hit_true_logfreq': round(h_lf,3)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy()
    V = int(BILIN.lm_head.weight.shape[0])
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); is_func = np.array([CLASSES[c] in FUNCTION for c in tok2cls], bool)
    tgt_all = S[:, 1:].reshape(-1); ct = Counter(tgt_all); Ntot = len(tgt_all)
    logfreq = np.log(np.array([ct.get(int(t),1) for t in range(V)])/Ntot + 1e-12)
    base_func = float(is_func[tgt_all].mean())
    out = {'corpus_base_func_rate': round(base_func,4), 'models': {}}
    r = run(BILIN, blocks, S, 1152, cidx, is_func, logfreq); out['models']['bilin18'] = r; print(f"bilin18: {r}", flush=True)
    for short in ['swiglu18','bilin12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd')
            r = run(mdl, blocks, S, Dm, cidx, is_func, logfreq); out['models'][short] = r; print(f"{short}: {r}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = all('gram_top1_func_frac' in out['models'][k] and out['models'][k]['gram_top1_func_frac'] > base_func + 0.1
             and out['models'][k]['gram_true_logfreq'] < out['models'][k]['hit_true_logfreq'] for k in out['models'])
    out['pred_a_family_content_errors'] = bool(ok)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"corpus base func {base_func:.3f}", flush=True)
    print(f"(a) content-error + hedging structure family-wide: {ok}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
