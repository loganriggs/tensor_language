"""GENERALIZE this session's headline (§1000/§1005: linearizing MLPs costs mostly CONTENT, ~4x grammar) across the
tensor-language family. §942 showed the middle-nonlinearity DIP is universal, but did NOT split it into content vs
grammar. Here: for each model, compositionally linearize the MIDDLE MLP band (each layer fit on the already-linearized
upstream, §1000 method) and chain-rule-split the CE cost into class-CE (grammar) vs within-CE (content). Includes
swiglu18 -- a NON-bilinear (SwiGLU) model -- to test whether "multiplication serves content" is ARCHITECTURE-GENERAL
or specific to the bilinear form.

REGISTERED PREDICTIONS:
  (0) SANITY: baseline CE reasonable per model; linearizing the middle costs > 0 within-CE for each.
  (a) MULTIPLICATION SERVES CONTENT, ARCHITECTURE-GENERAL: for EVERY model (bilin18, bilin12, swiglu18), linearizing
      the middle band costs within-CE (content) MUCH more than class-CE (grammar), ratio > 2.5 -> the multiplication's
      content role is a family-general fact, not bilinear-specific (swiglu18 shows it too);
  (b) report per-model middle within/class linearization cost + ratio."""
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
OUT = PT + 'cross_family_mult_content_results.json'
NCAL = 64; NEVAL = 120; SEQ = 256; RIDGE = 10.0
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
WLIN = {}; CTX = {'installed': set(), 'capture': None, 'buf': None}


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


def hook_factory(L, Dm):
    def h(mo, i_, o_):
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_
        if L == CTX['capture']:
            CTX['buf'][0].append(x.reshape(-1, Dm).detach().cpu()); CTX['buf'][1].append(o.float().reshape(-1, o.shape[-1]).detach().cpu())
            return None
        if L in CTX['installed']:
            x1 = torch.cat([x.reshape(-1, Dm), torch.ones(x.reshape(-1, Dm).shape[0], 1, device=DEV)], 1)
            return (x1 @ WLIN[L]).reshape(o.shape).to(o.dtype)
        return None
    return h


@torch.no_grad()
def fit_middle_compositional(mdl, calib, band, Dm):
    CTX['installed'] = set()
    for L in band:
        CTX['capture'] = L; CTX['buf'] = ([], [])
        for i in range(0, calib.shape[0], 8): forward_logits(mdl, calib[i:i+8].to(DEV)[:, :-1].contiguous(), Dm)
        X = torch.cat(CTX['buf'][0], 0).to(DEV); Y = torch.cat(CTX['buf'][1], 0).to(DEV)
        n = min(X.shape[0], 12000)
        if X.shape[0] > n: sel = torch.randperm(X.shape[0], device=DEV)[:n]; X = X[sel]; Y = Y[sel]
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        WLIN[L] = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(Dm+1, device=DEV), X1.T @ Y)
        CTX['installed'].add(L); del X, Y
    CTX['capture'] = None


@torch.no_grad()
def split_ce(mdl, blocks, cidx, C, Dm):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(mdl, idx, Dm).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    return {'full_ce': round(tot/n, 4), 'class_ce': round(totc/n, 4), 'within_ce': round((tot-totc)/n, 4)}


@torch.no_grad()
def run_model(mdl, calib, blocks, d, tag):
    Dm = mdl.transformer.wte.weight.shape[1]; nl = len(mdl.transformer.h)
    V = int(mdl.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)):
        if tid < V: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    band = list(range(nl//3, 2*nl//3))   # middle third
    hooks = [mdl.transformer.h[L].mlp.register_forward_hook(hook_factory(L, Dm)) for L in range(nl)]
    WLIN.clear(); CTX['installed'] = set(); CTX['capture'] = None
    base = split_ce(mdl, blocks, cidx, C, Dm)
    fit_middle_compositional(mdl, calib, band, Dm)
    CTX['installed'] = set(band); CTX['capture'] = None
    r = split_ce(mdl, blocks, cidx, C, Dm)
    for h in hooks: h.remove()
    CTX['installed'] = set(); WLIN.clear()
    wc = round(r['within_ce']-base['within_ce'], 4); cc = round(r['class_ce']-base['class_ce'], 4)
    res = {'baseline_ce': base['full_ce'], 'middle_band': [band[0], band[-1]], 'within_cost': wc, 'class_cost': cc,
           'content_grammar_ratio': round(wc/max(abs(cc), 1e-6), 2)}
    print(f"{tag}: middle L{band[0]}-{band[-1]} within +{wc} class +{cc} ratio {res['content_grammar_ratio']}", flush=True)
    return res


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    out = {'models': {}}
    out['models']['bilin18'] = run_model(BILIN, calib, blocks, d, 'bilin18')
    for short in ['bilin12', 'swiglu18']:
        try:
            mdl, cfg = load_elriggs(short); mdl = mdl.to(DEV).eval()
            out['models'][short] = run_model(mdl, calib, blocks, d, short)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    ok = [k for k in out['models'] if 'content_grammar_ratio' in out['models'][k]]
    out['pred_a_content_general'] = bool(len(ok) >= 2 and all(out['models'][k]['content_grammar_ratio'] > 2.5 and out['models'][k]['within_cost'] > 0 for k in ok))
    out['swiglu_ratio'] = out['models'].get('swiglu18', {}).get('content_grammar_ratio')
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a multiplication-serves-content architecture-general {out['pred_a_content_general']} | swiglu ratio {out['swiglu_ratio']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
