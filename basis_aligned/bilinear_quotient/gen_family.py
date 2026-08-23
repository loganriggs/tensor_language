"""FAMILY-UNIVERSALITY of the generation phenotypes (§1135-1136 found three readable failure modes in bilin18:
function-word soup / fragment salad / unanchored zero-repetition stream). Same battery on swiglu18 (independently
trained, different MLP nonlinearity, same depth/width): free-run 64 tokens from 96 prompts under base |
content-band abl (L5-14 MLPs, §1058 confirmed the same band split) | grammar-band abl (L0-1) | random-2 control
(L7+L12) | value-residual off (lamb=0; the family shares the architecture). Text metrics as §1136 (content-word
rate, legal transitions, rep-4gram) + topic retention under swiglu18's OWN content basis.

REGISTERED PREDICTIONS:
  (0) SANITY: base near natural on cw-rate/legal; random control ≈ base.
  (a) PHENOTYPES REPLICATE: content-abl craters cw-rate >= 1.5x grammar-abl's drop (soup), and vres-off shows
      the unanchored-stream signature (rep4 <= 0.005 AND cw-rate >= natural) -> the failure phenotypes are
      FAMILY-GENERAL behavior, not bilin18 quirks — the strongest behavioral universality claim available;
  (b) VRES-SPECIFIC: if the vres-off stream signature is absent in swiglu18, the zero-repetition phenomenon is
      bilin18-specific (different broadcast weighting) — report plainly;
  (c) uniform degeneration -> phenotypes not family-general (report plainly)."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/rspd')
from tier2_model import load_elriggs
import census_lib as cl
import tiktoken

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gen_family_results.json'
NP = 96; PLEN = 64; GLEN = 64; REF = [8, 10, 12]; K = 64
CONTENT_BAND = list(range(5, 15)); GRAMMAR_BAND = [0, 1]; RAND_BAND = [7, 12]
DEV = 'cuda'
enc = tiktoken.get_encoding('gpt2')
SUB = {'mlp_mean': {}, 'active': set()}

mdl, cfg = load_elriggs('swiglu18', device=DEV, dtype=torch.float32); mdl.eval()
H = mdl.transformer.h
D = mdl.transformer.wte.weight.shape[1]
V = mdl.transformer.wte.weight.shape[0]


@torch.no_grad()
def fwd_logits(idx):
    x = mdl.transformer.wte(idx)
    x = F.rms_norm(x, (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    x = F.rms_norm(x, (D,))
    return 30*torch.tanh(mdl.lm_head(x)/30)


def mlp_mean_hook(L):
    def h(mo, i_, o_):
        if L not in SUB['active']: return None
        y = o_[0] if isinstance(o_, tuple) else o_
        return SUB['mlp_mean'][L].view(1, 1, D).expand_as(y).to(y.dtype)
    return h


DET = {'the','a','an','this','that','these','those','some','any','each','every','no','all','both'}
PREP = {'of','in','on','at','by','for','with','from','to','into','over','under','about','after','before','between','through','during','against','without','within','upon','across','off','up','down','out'}
PRON = {'i','you','he','she','it','we','they','me','him','her','us','them','his','its','their','my','your','our','who','whom','which','what'}
CONJ = {'and','or','but','so','because','if','while','although','though','when','where','as','than','whether','nor','yet','since','unless'}
AUX = {'is','are','was','were','be','been','being','am','has','have','had','do','does','did','will','would','can','could','should','may','might','must','shall','not'}
NCLS = 10


def label_token(tid):
    try: raw = enc.decode([tid])
    except Exception: return 9
    s = raw.strip()
    if s == '': return 9
    low = s.lower()
    if re.fullmatch(r"[0-9][0-9,\.]*", s): return 5
    if re.fullmatch(r"[^\w\s]+", s): return 6
    if low in DET: return 0
    if low in PREP: return 1
    if low in PRON: return 2
    if low in CONJ: return 3
    if low in AUX: return 4
    if s[0].isupper(): return 7
    if s.isalpha(): return 8
    return 9


@torch.no_grad()
def class_bigram(seqs, cls_of):
    c = cls_of[seqs]
    M = torch.zeros(NCLS, NCLS, device=DEV)
    a = c[:, :-1].reshape(-1); b = c[:, 1:].reshape(-1)
    M.index_put_((a, b), torch.ones(a.shape[0], device=DEV), accumulate=True)
    return (M + 0.5) / (M + 0.5).sum(1, keepdim=True)


@torch.no_grad()
def rep4(seqs):
    fr = []
    for row in seqs.tolist():
        grams = [tuple(row[i:i+4]) for i in range(len(row)-3)]
        fr.append(1 - len(set(grams))/max(len(grams), 1))
    return sum(fr)/len(fr)


@torch.no_grad()
def content_coords(seqs, xbars, Uc):
    cap = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): cap[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    SUB['active'] = set()
    for i in range(0, seqs.shape[0], 8): fwd_logits(seqs[i:i+8])
    for h in hs: h.remove()
    tokf = seqs.reshape(-1)
    dev = None
    for L in REF:
        X = torch.cat(cap[L], 0); cap[L] = []
        dv = X - xbars[tokf]
        dev = dv if dev is None else dev + dv
    c = ((dev/len(REF)) @ Uc).view(seqs.shape[0], seqs.shape[1], K)
    return c.mean(1)


@torch.no_grad()
def generate(prompts, cond_active, vres_off, seed):
    orig_vl = [blk.attn.lamb.data.clone() for blk in H]
    if vres_off:
        for blk in H: blk.attn.lamb.data.fill_(0.0)
    SUB['active'] = cond_active
    g = torch.Generator(device=DEV).manual_seed(seed)
    seqs = prompts.clone()
    for step in range(GLEN):
        lg = fwd_logits(seqs)[:, -1].float()/0.8
        p = F.softmax(lg, -1)
        nxt = torch.multinomial(p, 1, generator=g)
        seqs = torch.cat([seqs, nxt], 1)
    SUB['active'] = set()
    for blk, vl in zip(H, orig_vl): blk.attn.lamb.data.copy_(vl)
    return seqs[:, PLEN:]


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NP + 100)[:, :256].contiguous()
    prompts = rows[:NP, :PLEN].to(DEV).clamp_max(V-1)
    natural = rows[NP:NP+100, :128].to(DEV).clamp_max(V-1)
    cls_of = torch.tensor([label_token(t) for t in range(V)], device=DEV)

    # swiglu18's own content basis + per-token means + band output means
    capR = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): capR[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    capO = {L: [] for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND)}
    for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND):
        def mko(L):
            def h(mo, i_, o_):
                y = o_[0] if isinstance(o_, tuple) else o_
                capO[L].append(y.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mko(L)))
    idsL = []
    for i in range(0, 100, 8):
        idx = natural[i:i+8][:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd_logits(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    devsum = None; xb_pool = torch.zeros(V, D, device=DEV)
    for L in REF:
        X = torch.cat(capR[L], 0); capR[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xb = xb/cn.clamp_min(1).unsqueeze(1); xb_pool += xb/len(REF)
        dv = X - xb[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False); Uc = Vt[:K].T.contiguous(); del dev
    for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND):
        SUB['mlp_mean'][L] = torch.cat(capO[L], 0).mean(0); capO[L] = []
    nat_bigram = class_bigram(natural, cls_of)
    content_tok = ((cls_of == 8) | (cls_of == 7)).float()
    legal = nat_bigram > 0.02
    def cw_rate(seqs): return float(content_tok[seqs].mean())
    def legal_rate(seqs):
        c = cls_of[seqs]
        return float(legal[c[:, :-1].reshape(-1), c[:, 1:].reshape(-1)].float().mean())
    nat_cw = cw_rate(natural)
    print(f"swiglu18 natural: cw {nat_cw:.3f} | legal {legal_rate(natural):.3f}", flush=True)

    hooks = [H[L].mlp.register_forward_hook(mlp_mean_hook(L)) for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND)]
    prompt_cc = content_coords(prompts, xb_pool, Uc)
    conds = {'base': (set(), False), 'content_abl': (set(CONTENT_BAND), False),
             'grammar_abl': (set(GRAMMAR_BAND), False), 'random2_abl': (set(RAND_BAND), False),
             'vres_off': (set(), True)}
    res = {}; texts = {}
    for name, (act, voff) in conds.items():
        gen = generate(prompts, act, voff, seed=123)
        cc = content_coords(gen, xb_pool, Uc)
        res[name] = {'topic': round(float(F.cosine_similarity(cc, prompt_cc, dim=-1).mean()), 4),
                     'cw_rate': round(cw_rate(gen), 4), 'legal': round(legal_rate(gen), 4),
                     'rep4': round(rep4(gen), 4)}
        texts[name] = [enc.decode(r) for r in gen.tolist()[:16]]
        print(f"{name:>12}: {res[name]} | sample {texts[name][0][:110]!r}", flush=True)
    for h in hooks: h.remove()

    b = res['base']
    cw_ratio = (b['cw_rate']-res['content_abl']['cw_rate'])/max(b['cw_rate']-res['grammar_abl']['cw_rate'], 1e-4)
    vsig = bool(res['vres_off']['rep4'] <= 0.005 and res['vres_off']['cw_rate'] >= nat_cw)
    out = {'model': 'swiglu18', 'natural_cw': round(nat_cw, 4), 'conditions': res, 'texts': texts,
           'cw_drop_ratio': round(cw_ratio, 2), 'vres_stream_signature': vsig}
    out['pred_a_replicates'] = bool(cw_ratio >= 1.5 and vsig)
    out['pred_b_vres_specific'] = bool(cw_ratio >= 1.5 and not vsig)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"cw-drop ratio {cw_ratio:.2f} | vres stream signature {vsig}", flush=True)
    print(f"pred_a replicates {out['pred_a_replicates']} | pred_b vres-specific {out['pred_b_vres_specific']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
