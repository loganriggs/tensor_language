"""THIRD universality leg (with §1010 multiplicative-content, §1011 context-window): is the DEPTH structure
"grammar resolves EARLY, content builds LATE" architecture-general? Logit-lens each layer's residual through the
model's own head, chain-rule-split into class-CE (grammar) and within-CE (content), per model (bilin18, bilin12,
swiglu18). Find the DEPTH (as a fraction of layers) at which each reaches 90% of its total (embedding->final)
reduction. Grammar should reach 90% at a shallower depth than content, in every model.

REGISTERED PREDICTIONS:
  (0) SANITY: logit-lens CE decreases monotone-ish with depth; final-layer lens ~ the model's actual CE.
  (a) GRAMMAR-EARLY / CONTENT-LATE, FAMILY-GENERAL: in EVERY model, grammar (class-CE) reaches 90% of its reduction
      at a SHALLOWER depth-fraction than content (within-CE) -> grammar resolves front, content builds through the
      stack, in all architectures incl swiglu;
  (b) report per-model per-layer class-CE and within-CE + the 90%-reduction depth-fraction for each."""
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
OUT = PT + 'content_depth_onset_family_results.json'
NEVAL = 160; SEQ = 256
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


@torch.no_grad()
def lens_ce_by_layer(mdl, blocks, cidx, C, Dm):
    nl = len(mdl.transformer.h)
    # accumulate class/within CE at each depth 0..nl (0 = embedding, nl = final)
    acc = {L: [0.0, 0.0, 0] for L in range(nl+1)}  # sum_tok, sum_class, n
    Cmat = F.one_hot(cidx, C).float()
    def lens(x):
        return 30.0*torch.tanh(mdl.lm_head(F.rms_norm(x, (Dm,)))/30.0)
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous(); tf = tgt.reshape(-1)
        x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
        states = [x]
        for blk in mdl.transformer.h:
            x, v1 = blk(x, v1, x0); states.append(x)
        for L in range(nl+1):
            lp = F.log_softmax(lens(states[L]).float(), -1); lpf = lp.reshape(-1, lp.shape[-1])
            lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
            pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
            acc[L][0] += float(-lp_tok.sum()); acc[L][1] += float(-lp_cls.sum()); acc[L][2] += tf.shape[0]
    cls = [acc[L][1]/acc[L][2] for L in range(nl+1)]
    wth = [(acc[L][0]-acc[L][1])/acc[L][2] for L in range(nl+1)]
    return cls, wth, nl


def onset_frac(series, nl):
    # depth-fraction at which `series` reaches 90% of its (layer0 -> final) reduction
    start, end = series[0], series[-1]; span = start - end
    if span <= 1e-6: return None
    for L in range(nl+1):
        if (start - series[L]) >= 0.9*span: return round(L/nl, 3)
    return 1.0


@torch.no_grad()
def run_model(mdl, blocks, d, tag):
    Dm = mdl.transformer.wte.weight.shape[1]; V = int(mdl.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)):
        if tid < V: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    cls, wth, nl = lens_ce_by_layer(mdl, blocks, cidx, C, Dm)
    gf = onset_frac(cls, nl); cf = onset_frac(wth, nl)
    res = {'nlayer': nl, 'class_ce_by_layer': [round(c, 3) for c in cls], 'within_ce_by_layer': [round(w, 3) for w in wth],
           'grammar_onset_frac': gf, 'content_onset_frac': cf}
    print(f"{tag} (nl {nl}): grammar 90%-onset depth {gf} vs content {cf} | final class {cls[-1]:.3f} within {wth[-1]:.3f}", flush=True)
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
    ok = [k for k in out['models'] if 'grammar_onset_frac' in out['models'][k] and out['models'][k]['grammar_onset_frac'] is not None and out['models'][k]['content_onset_frac'] is not None]
    out['pred_a_grammar_early_general'] = bool(len(ok) >= 2 and all(out['models'][k]['grammar_onset_frac'] < out['models'][k]['content_onset_frac'] for k in ok))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a grammar-early/content-late family-general {out['pred_a_grammar_early_general']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
