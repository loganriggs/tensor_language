"""BACK READOUT v2 (cosine write-attribution, fixes §855 rare-token norm artifact)
BACK READOUT mechanism: how do mlp16/17 map the maintained class+position to next-token logits?
(other end of the barbell; §851 showed they read class 10-13x, position 5-6x). These are the highest-
benefit late components (0.89, 0.71) and sit ~1-2 layers from the output, so their output is close to a
DIRECT logit contribution. For their top units, trace READ (which current-token grammatical class the
unit's (Left·x)(Right·x) activation is selective for) and WRITE (the tokens the unit promotes via
Down-row → lm_head). If units read a class and write grammatically-appropriate next tokens (det→nouns,
etc.), that is §828's grammatical sequencing realized at the weight level.

REGISTERED PREDICTIONS:
  (0) SANITY: exact reconstruction of the bilinear output; write-tokens are real vocab tokens;
  (a) READOUT MAP: top mlp17/16 units read a current-token class and WRITE a coherent set of next tokens;
      report read-class -> written-tokens per unit;
  (b) do the written tokens follow the grammatical-sequence expectation (e.g. read determiner -> write
      nouns/content), linking to §828, or are they content/frequency promotions?"""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'back_readout2_results.json'
NEVAL = 200; MINCOUNT = 8; NUNIT = 16; NTOK = 10
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
def capture(rows, L):
    mlp = m.transformer.h[L].mlp; Xs = []; toks = []
    def pre(mo, a): Xs.append(a[0].detach().float().reshape(-1, D))
    hp = mlp.register_forward_pre_hook(pre)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); toks.append(idx.cpu().numpy().reshape(-1))
    hp.remove(); return torch.cat(Xs, 0), np.concatenate(toks)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL); d = dec()
    lm = m.lm_head.weight.detach().float()                # (V, D)
    out = {'layers': {}}
    for L in [16, 17]:
        mlp = m.transformer.h[L].mlp
        Lw = mlp.Left.weight.detach().float(); Rw = mlp.Right.weight.detach().float(); Dw = mlp.Down.weight.detach().float()
        X, toks = capture(rows, L)
        clslab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks]); nc = len(CLASSES)
        la = X @ Lw.T; ra = X @ Rw.T; hid = la * ra        # (N, 4608)
        # unit importance by contribution to output variance: ||Down[:,k]|| * std(hid_k)
        imp = Dw.norm(dim=0) * hid.std(0)
        top = torch.topk(imp, NUNIT).indices.tolist()
        units = []
        for k in top:
            a = hid[:, k].cpu().numpy()
            cm = np.array([np.abs(a[clslab == c]).mean() if (clslab == c).any() else 0 for c in range(nc)])
            rc = int(cm.argmax()); rsel = float(cm[rc]/(cm.mean()+1e-9))
            # write: unit k's output direction Down[:,k] -> logits via lm_head
            sgn = float(np.sign(a.mean() + 1e-9))
            dv = Dw[:, k] * sgn
            cos = (lm @ dv) / (lm.norm(dim=1) * dv.norm() + 1e-9)   # cosine: removes rare-token norm artifact
            wtop = torch.topk(cos, NTOK).indices.cpu().numpy()
            wtoks = [d(int(t)) for t in wtop]; wcls = [classify(x) for x in wtoks]
            from collections import Counter; wc = Counter(wcls).most_common(1)[0]
            units.append({'unit': int(k), 'read_class': CLASSES[rc], 'read_sel': round(rsel, 2),
                          'writes_tokens': [repr(x) for x in wtoks], 'write_class_mode': f'{wc[0]}:{wc[1]}/{NTOK}'})
        out['layers'][f'mlp{L}'] = units
        print(f"=== mlp{L} readout units (read class -> write tokens) ===", flush=True)
        for u in units[:10]:
            print(f"  u{u['unit']}: read {u['read_class']}({u['read_sel']}) -> write[{u['write_class_mode']}] {u['writes_tokens'][:7]}", flush=True)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
