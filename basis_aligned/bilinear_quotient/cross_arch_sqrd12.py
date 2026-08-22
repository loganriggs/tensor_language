"""DOES THE WHOLE ACCOUNT REPLICATE ACROSS MLP ARCHITECTURE? (final generality test). bilin18 = bilinear
attention + BILINEAR MLP. Its sister swiglu18 (Elriggs/gpt2-bilinear-swiglu-18l-9h-1152embd) is the SAME size
(18L/9H/1152), SAME data, SAME bilinear attention, but a SWIGLU MLP. If the two-machine + induction + loss-
budget account is a property of the learned computation (not of the bilinear-MLP form), the account should show the same signatures at smaller depth/scale. Run on swiglu18: (1) the loss budget by position type (§879), (2) synthetic induction +
which attention layer gates it (§877), (3) overall grammar/content split (§829). Compare to bilin18 references.

BILIN18 REFERENCES: budget first-mention 78.4% / seen-other 20.3% / inductable 1.3%; overall CE 3.24 = class
0.76 + within 2.48; synthetic induction score 11.8, gated by L5.

REGISTERED PREDICTIONS:
  (0) SANITY: sqrd12 runs and gives a sane overall CE (~3-4 nats);
  (a) ARCHITECTURE-INDEPENDENT: sqrd12 reproduces the budget SHAPE (first-mention dominates >60%, induction
      cheap <5%), a strong synthetic induction score (>5), and a ~20-30/70-80 grammar/content split -> the
      account is not specific to the bilinear MLP;
  (b) if sqrd12 differs markedly (weak induction, different budget shape, or induction gated at a very
      different layer), note the architecture-dependent piece plainly."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd'); sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cross_arch_sqrd12_results.json'
DEV = 'cuda'; NEVAL = 200; SEQ = 256; NSYN = 48; L = 64
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
ABL = {'layer': -1}


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


def make_forward(m):
    Dm = m.transformer.wte.weight.shape[1]
    def fwd(idx):
        x = F.rms_norm(m.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
        return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (Dm,)))/30.0)
    return fwd


def ablate_hook(mo, i_, o_):
    if ABL['layer'] < 0: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; z = torch.zeros_like(y)
    return (z,) + tuple(o_[1:]) if isinstance(o_, tuple) else z


@torch.no_grad()
def main():
    t0 = time.time()
    m, cfg = load_elriggs('sqrd12', device=DEV, dtype=torch.float32)
    NL = len(m.transformer.h); fwd = make_forward(m); d = dec()
    cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(S.reshape(-1)):
        if int(tid) < V: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    # per-position class/within CE
    cl_ce = []; wi_ce = []
    for i in range(0, nb, 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); pcl = lp.exp() @ Cmat
        tf = tgt.reshape(-1); lpf = lp.reshape(-1, V); tcl = cidx[tf]
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        lp_cls = (pcl.reshape(-1, len(CLASSES))[torch.arange(tf.shape[0], device=DEV), tcl] + 1e-12).log()
        cl_ce.append((-lp_cls).cpu().numpy()); wi_ce.append((-(lp_tok - lp_cls)).cpu().numpy())
    cl_f = np.concatenate(cl_ce); wi_f = np.concatenate(wi_ce); tot_f = cl_f + wi_f
    # buckets
    inductable = np.zeros((nb, SEQ-1), dtype=bool); firstment = np.zeros((nb, SEQ-1), dtype=bool)
    for r in range(nb):
        seen_tok = set(); seen_big = {}
        for pp in range(SEQ-1):
            cur = int(S[r, pp]); nxt = int(S[r, pp+1])
            firstment[r, pp] = nxt not in seen_tok
            if cur in seen_big and seen_big[cur] == nxt: inductable[r, pp] = True
            seen_big[cur] = nxt; seen_tok.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable; other = ~inductable & ~firstment
    budget = {}
    for name, mk in [('inductable', inductable), ('first_mention', firstment), ('seen_other', other)]:
        frac = float(mk.mean()); ce = float(tot_f[mk].mean())
        budget[name] = {'fraction': round(frac, 3), 'mean_ce': round(ce, 3),
                        'pct_of_total_loss': round(float(100*frac*ce/float(tot_f.mean())), 1)}
    # synthetic induction + locate
    g = torch.Generator(device=DEV).manual_seed(0)
    base = torch.randint(0, 50000, (NSYN, L), generator=g, device=DEV); seqs = torch.cat([base, base], 1)
    def ind_score():
        idx = seqs[:, :-1].contiguous(); tgt = seqs[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); l = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return float(l[:, :L-1].mean()) - float(l[:, L:2*L-1].mean())
    ABL['layer'] = -1; base_ind = ind_score()
    per = {}
    for Li in range(NL):
        hh = m.transformer.h[Li].attn.register_forward_hook(ablate_hook)
        ABL['layer'] = Li; per[Li] = round(base_ind - ind_score(), 3); ABL['layer'] = -1; hh.remove()
    topL = max(per, key=per.get)
    out = {'model': 'sqrd12', 'overall_ce': round(float(tot_f.mean()), 3),
           'overall_class_ce': round(float(cl_f.mean()), 3), 'overall_within_ce': round(float(wi_f.mean()), 3),
           'within_frac': round(float(wi_f.mean()/tot_f.mean()), 3), 'budget': budget,
           'synthetic_induction_score': round(base_ind, 3), 'induction_gate_layer': topL,
           'induction_drop_at_gate': per[topL], 'induction_drop_by_layer_top5': sorted(per.items(), key=lambda kv: -kv[1])[:5],
           'bilin18_ref': {'first_mention_pct': 78.4, 'within_frac': 0.765, 'induction_score': 11.8, 'gate_layer': 5},
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_architecture_independent'] = bool(
        budget['first_mention']['pct_of_total_loss'] > 60 and budget['inductable']['pct_of_total_loss'] < 5 and
        base_ind > 5 and 0.6 < out['within_frac'] < 0.85)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"sqrd12 overall CE {out['overall_ce']} = class {out['overall_class_ce']} + within {out['overall_within_ce']} (within frac {out['within_frac']})", flush=True)
    print(f"budget: first-mention {budget['first_mention']['pct_of_total_loss']}% | seen-other {budget['seen_other']['pct_of_total_loss']}% | inductable {budget['inductable']['pct_of_total_loss']}%", flush=True)
    print(f"synthetic induction score {out['synthetic_induction_score']} (bilin18 11.8) | gate layer L{topL} (bilin18 L5) | top5 {out['induction_drop_by_layer_top5']}", flush=True)
    print(f"(a) architecture-independent: {out['pred_a_architecture_independent']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
