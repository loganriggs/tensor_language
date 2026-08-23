"""FRONTIER / benchmark ceiling: the whole-model understanding benchmark's content term is capped because the middle
is genuinely MULTIPLICATIVE (§941: middle MLPs ~60% multiplicative; §993). How much of the CONTENT specifically is
irreducible to a linear stand-in? Replace whole BANDS of MLPs with their BEST-FIT linear surrogate (ridge-fit
input->output on a calibration set; §941/§993 method) and split the CE cost by the chain rule into class-CE
(grammar) vs within-CE (content). The middle band's within-CE cost is the multiplicative CONTENT that no linear /
table / bag stand-in can capture -- the benchmark's content ceiling, mechanistically.

REGISTERED PREDICTIONS:
  (0) NULL: (i) shuffled-input linear surrogate for the middle ~= mean-ablating the middle (a genuine linear map
      beats a broken one); (ii) linearizing the FRONT band is CHEAP (front MLPs are ~linear anyway, §941/§993).
  (a) MULTIPLICATIVE MIDDLE = CONTENT: linearizing the MIDDLE band (L6-15) costs WITHIN-CE (content) MUCH more than
      class-CE (grammar) -> the middle's irreducible multiplication is a CONTENT computation; this within-cost is the
      benchmark's multiplicative-content ceiling;
  (b) report class/within cost for linearizing front (L0-5), middle (L6-15), all (L0-17), + the middle shuffled-null
      and mean-ablate-middle references."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_multiplicative_ceiling_results.json'
NCAL = 96; NEVAL = 160; SEQ = 256; RIDGE = 10.0
FRONT = list(range(0, 6)); MIDDLE = list(range(6, 16)); ALLL = list(range(18))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
WLIN = {}; WNULL = {}; MEANOUT = {}
MODE = {'kind': None, 'layers': set()}   # kind in {linear, shuffled, meanablate}


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


def sub_hook_factory(L):
    def h(mo, i_, o_):
        if MODE['kind'] is None or L not in MODE['layers']: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_
        if MODE['kind'] == 'meanablate':
            return MEANOUT[L].to(o.dtype).expand_as(o)
        W = WLIN[L] if MODE['kind'] == 'linear' else WNULL[L]
        x1 = torch.cat([x.reshape(-1, D), torch.ones(x.reshape(-1, D).shape[0], 1, device=DEV)], 1)
        return (x1 @ W).reshape(o.shape).to(o.dtype)
    return h


@torch.no_grad()
def fit(blocks):
    Xs = {L: [] for L in ALLL}; Ys = {L: [] for L in ALLL}; caps = {}; hs = []
    for L in ALLL:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = (i_[0] if isinstance(i_, tuple) else i_).float()
                caps[L] = (x.reshape(-1, D).detach(), (o_[0] if isinstance(o_, tuple) else o_).float().reshape(-1, D).detach())
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    for i in range(0, blocks.shape[0], 8):
        forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
        for L in ALLL:
            x, o = caps[L]; n = x.shape[0]; idx = torch.randperm(n, device=DEV)[:300]
            Xs[L].append(x[idx].cpu()); Ys[L].append(o[idx].cpu())
    for h in hs: h.remove()
    for L in ALLL:
        X = torch.cat(Xs[L], 0).to(DEV); Y = torch.cat(Ys[L], 0).to(DEV)
        MEANOUT[L] = Y.mean(0, keepdim=True)
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        A = X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV)
        WLIN[L] = torch.linalg.solve(A, X1.T @ Y)
        Xsh = X[torch.randperm(X.shape[0], device=DEV)]; Xsh1 = torch.cat([Xsh, torch.ones(Xsh.shape[0], 1, device=DEV)], 1)
        WNULL[L] = torch.linalg.solve(Xsh1.T @ Xsh1 + RIDGE*torch.eye(D+1, device=DEV), Xsh1.T @ Y)
        del X, Y


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    full = tot/n; classce = totc/n
    return {'full_ce': round(full, 4), 'class_ce': round(classce, 4), 'within_ce': round(full-classce, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    fit(calib)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(sub_hook_factory(L)) for L in ALLL]
    MODE['kind'] = None; base = split_ce(blocks, cidx, C)
    print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'conditions': {}}

    def cond(kind, layers):
        MODE['kind'] = kind; MODE['layers'] = set(layers)
        r = split_ce(blocks, cidx, C); MODE['kind'] = None
        r['within_cost'] = round(r['within_ce'] - base['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base['class_ce'], 4)
        return r

    for tag, kind, layers in [('lin_front', 'linear', FRONT), ('lin_middle', 'linear', MIDDLE),
                              ('lin_all', 'linear', ALLL), ('shuf_middle', 'shuffled', MIDDLE),
                              ('meanabl_middle', 'meanablate', MIDDLE)]:
        r = cond(kind, layers); out['conditions'][tag] = r
        print(f"{tag:>15}: within-cost +{r['within_cost']} class-cost +{r['class_cost']} (full {r['full_ce']})", flush=True)
    for h in hooks: h.remove()
    lm = out['conditions']['lin_middle']; lf = out['conditions']['lin_front']
    sm = out['conditions']['shuf_middle']; mm = out['conditions']['meanabl_middle']
    out['pred_0_null_ok'] = bool(abs(sm['full_ce'] - mm['full_ce']) < 0.15 and lf['within_cost'] < lm['within_cost'])
    out['pred_a_mult_middle_is_content'] = bool(lm['within_cost'] > abs(lm['class_cost']) and lm['within_cost'] > 0.1)
    out['multiplicative_content_ceiling_nats'] = lm['within_cost']
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lin_middle within-cost {lm['within_cost']} vs class-cost {lm['class_cost']} | shuf_middle {sm['full_ce']} ~ meanabl_middle {mm['full_ce']}", flush=True)
    print(f"pred_0 null {out['pred_0_null_ok']} | pred_a mult-middle=content {out['pred_a_mult_middle_is_content']} | ceiling {out['multiplicative_content_ceiling_nats']} nats", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
