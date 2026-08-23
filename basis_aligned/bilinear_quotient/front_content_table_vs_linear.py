"""REFINE the §1000 multiplicative-content ceiling. §1000 measured what a best-fit LINEAR MAP of each MLP's input
misses (+1.59 nats content). But the whole-model benchmark replaces components with per-TOKEN TABLES, which capture
ANY function of the current token (including nonlinear word-sense), not just linear ones. So part of §1000's
"multiplicative content" -- especially at the FRONT, whose input is dominated by the current-token embedding (x0
re-injection, §987) -- may be current-token-NONLINEAR content that a TABLE captures but a linear map misses, i.e. NOT
truly context-irreducible. The truly irreducible content is what BOTH a linear map AND a per-token table miss.

TEST: compositionally replace an MLP band (each layer fit on the already-replaced upstream, as in §1000) with either
(i) a best-fit LINEAR map of the input, or (ii) a per-TOKEN TABLE (current-token conditional-mean output; unseen
tokens -> global mean). Measure within-CE (content) cost for each, for the FRONT (0-5) and MIDDLE (6-15) bands.

REGISTERED PREDICTIONS:
  (0) NULL: no-replacement baseline == original CE; per-token table with a SHUFFLED token index ~= global-mean
      ablate (table is genuine, not leakage).
  (a) FRONT CONTENT IS PARTLY TABLE-CAPTURABLE: at the FRONT, the token-TABLE within-cost is MUCH LOWER than the
      LINEAR within-cost (table captures current-token-nonlinear word-sense the linear map misses) -> part of
      §1000's front ceiling is current-token lookup, NOT truly context-irreducible;
  (b) MIDDLE IS CONTEXT-MULTIPLICATIVE: at the MIDDLE, table ~= linear (both fail similarly) -> the middle content is
      genuinely context-dependent, not captured by any current-token stand-in;
  (c) the TRUE irreducible content ceiling = the token-table cost (what even a per-token table misses); report
      linear vs table within/class cost for front and middle + the table-shuffled null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_content_table_vs_linear_results.json'
NCAL = 96; NEVAL = 160; SEQ = 256; RIDGE = 10.0
FRONT = list(range(0, 6)); MIDDLE = list(range(6, 16)); ALLL = list(range(18))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
WLIN = {}; TABLE = {}; CTX = {'installed': set(), 'capture': None, 'buf': None, 'kind': 'linear', 'tokids': None, 'shuffle': False}


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
    CTX['tokids'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook_factory(L):
    def h(mo, i_, o_):
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_
        if L == CTX['capture']:
            CTX['buf'][0].append(x.reshape(-1, D).detach().cpu())
            CTX['buf'][1].append(o.float().reshape(-1, D).detach().cpu())
            CTX['buf'][2].append(CTX['tokids'].reshape(-1).detach().cpu())
            return None
        if L in CTX['installed']:
            if CTX['kind'] == 'linear':
                x1 = torch.cat([x.reshape(-1, D), torch.ones(x.reshape(-1, D).shape[0], 1, device=DEV)], 1)
                return (x1 @ WLIN[L]).reshape(o.shape).to(o.dtype)
            else:
                tids = CTX['tokids'].reshape(-1)
                if CTX['shuffle']: tids = tids[torch.randperm(tids.shape[0], device=DEV)]
                return TABLE[L][tids].reshape(o.shape).to(o.dtype)
        return None
    return h


@torch.no_grad()
def fit_band(calib, band, kind):
    CTX['installed'] = set(); CTX['kind'] = kind
    V = int(m.lm_head.weight.shape[0])
    for L in band:
        CTX['capture'] = L; CTX['buf'] = ([], [], [])
        for i in range(0, calib.shape[0], 8):
            forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
        X = torch.cat(CTX['buf'][0], 0).to(DEV); Y = torch.cat(CTX['buf'][1], 0).to(DEV); T = torch.cat(CTX['buf'][2], 0).to(DEV)
        if kind == 'linear':
            n = min(X.shape[0], 12000)
            if X.shape[0] > n: sel = torch.randperm(X.shape[0], device=DEV)[:n]; X = X[sel]; Y = Y[sel]
            X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
            WLIN[L] = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV), X1.T @ Y)
        else:
            gmean = Y.mean(0)
            tab = gmean.unsqueeze(0).repeat(V, 1)
            sums = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
            sums.index_add_(0, T, Y); cnts.index_add_(0, T, torch.ones_like(T, dtype=torch.float))
            seen = cnts >= 5
            tab[seen] = sums[seen] / cnts[seen].unsqueeze(1)
            TABLE[L] = tab.half()
        CTX['installed'].add(L); del X, Y, T
    CTX['capture'] = None


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    return {'full_ce': round(tot/n, 4), 'class_ce': round(totc/n, 4), 'within_ce': round((tot-totc)/n, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(hook_factory(L)) for L in ALLL]
    CTX['installed'] = set(); CTX['capture'] = None
    base = split_ce(blocks, cidx, C); print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'conditions': {}}

    def run(band, kind, shuffle=False):
        WLIN.clear(); TABLE.clear()
        fit_band(calib, band, kind)
        CTX['installed'] = set(band); CTX['capture'] = None; CTX['kind'] = kind; CTX['shuffle'] = shuffle
        r = split_ce(blocks, cidx, C)
        CTX['installed'] = set(); CTX['shuffle'] = False
        return {'within_cost': round(r['within_ce'] - base['within_ce'], 4), 'class_cost': round(r['class_ce'] - base['class_ce'], 4)}

    for tag, band, kind in [('front_linear', FRONT, 'linear'), ('front_table', FRONT, 'table'),
                            ('middle_linear', MIDDLE, 'linear'), ('middle_table', MIDDLE, 'table')]:
        out['conditions'][tag] = run(band, kind)
        print(f"{tag:>14}: within-cost +{out['conditions'][tag]['within_cost']} class-cost +{out['conditions'][tag]['class_cost']}", flush=True)
    out['conditions']['front_table_shuffled_null'] = run(FRONT, 'table', shuffle=True)
    print(f"front_table_shuffled_null within +{out['conditions']['front_table_shuffled_null']['within_cost']}", flush=True)
    for h in hooks: h.remove()
    fl = out['conditions']['front_linear']['within_cost']; ft = out['conditions']['front_table']['within_cost']
    ml = out['conditions']['middle_linear']['within_cost']; mt = out['conditions']['middle_table']['within_cost']
    out['front_table_captures_extra'] = round(fl - ft, 4)   # content the table captures beyond linear (front)
    out['pred_0_null_ok'] = bool(out['conditions']['front_table_shuffled_null']['within_cost'] > ft + 0.1)
    out['pred_a_front_table_beats_linear'] = bool(ft < fl - 0.1)
    out['pred_b_middle_table_approx_linear'] = bool(abs(mt - ml) < max(0.1, 0.3*ml))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"FRONT linear {fl} table {ft} (table captures extra {out['front_table_captures_extra']}) | MIDDLE linear {ml} table {mt}", flush=True)
    print(f"pred_0 null {out['pred_0_null_ok']} | pred_a front-table<linear {out['pred_a_front_table_beats_linear']} | pred_b middle table~linear {out['pred_b_middle_table_approx_linear']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
