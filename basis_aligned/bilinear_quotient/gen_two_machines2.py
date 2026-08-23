"""METRIC-SHARPENED rerun of §1135 (registered there): class-bigram KL conflated the two phenotypes (function-
word soup also breaks class stats). Cleaner axis pair: (i) CONTENT-WORD RATE — fraction of generated tokens that
are alphabetic non-function words (the content machine's output payload); (ii) FUNCTION-STRUCTURE SCORE — fraction
of adjacent pairs that are grammatically legal class transitions (top-90% mass transitions of natural prose).
Same five conditions, same seed; ALL generations saved to JSON for re-scoring/reading.

REGISTERED PREDICTIONS:
  (0) SANITY: base/random-control near natural on both metrics.
  (a) CLEAN DISSOCIATION at the sharpened metrics: content-abl craters content-word rate >= 2x more than
      grammar-abl does, while grammar-abl craters the legal-transition score >= 2x more than content-abl ->
      the §1135 phenotypes quantify as a double dissociation with the right instruments;
  (b) vres-off: content-word rate INFLATED or normal but legal-transition score low with rep4 ~0 (the
      unanchored-stream phenotype gets its own quantitative signature);
  (c) if the conflation persists even here, generation damage is genuinely joint (report plainly)."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gen_two_machines2_results.json'
NP = 96; PLEN = 64; GLEN = 64; REF = [8, 10, 12]; K = 64
CONTENT_BAND = list(range(5, 15)); GRAMMAR_BAND = [0, 1]; RAND_BAND = [7, 12]
H = m.transformer.h
enc = tiktoken.get_encoding('gpt2')
SUB = {'mlp_mean': {}, 'active': set(), 'vres_off': False}


def fwd_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mlp_mean_hook(L):
    def h(mo, i_, o_):
        if L not in SUB['active']: return None
        return SUB['mlp_mean'][L].view(1, 1, D).expand_as(o_).to(o_.dtype)
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


def bigram_kl(P, Q):
    return float((P * (P/Q).log()).sum(1).mean())


@torch.no_grad()
def rep4(seqs):
    fr = []
    for row in seqs.tolist():
        grams = [tuple(row[i:i+4]) for i in range(len(row)-3)]
        fr.append(1 - len(set(grams))/max(len(grams), 1))
    return sum(fr)/len(fr)


@torch.no_grad()
def content_coords(seqs, xbars, Uc, V):
    """clean-model mean content coords per row over given token span"""
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
    SUB['vres_off'] = False
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
    prompts = rows[:NP, :PLEN].to(DEV)
    natural = rows[NP:NP+100, :128].to(DEV)
    V = int(m.lm_head.weight.shape[0])
    cls_of = torch.tensor([label_token(t) for t in range(V)], device=DEV)
    cls_of = torch.where(cls_of >= 0, cls_of, torch.full_like(cls_of, 9))

    # clean-model content basis + per-token means (teacher-forced pass over natural rows)
    capR = {L: [] for L in REF}; hs = []
    for L in REF:
        def mk(L):
            def h(mo, i_, o_): capR[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    capO = {L: [] for L in CONTENT_BAND + GRAMMAR_BAND + RAND_BAND}
    hs2 = []
    for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND):
        def mko(L):
            def h(mo, i_, o_): capO[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs2.append(H[L].mlp.register_forward_hook(mko(L)))
    idsL = []
    for i in range(0, 100, 8):
        idx = natural[i:i+8][:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd_logits(idx)
    for h in hs + hs2: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    devsum = None; xb_pool = torch.zeros(V, D, device=DEV)
    for L in REF:
        X = torch.cat(capR[L], 0); capR[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xb = xb/cn.clamp_min(1).unsqueeze(1)
        xb_pool += xb/len(REF)
        dv = X - xb[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False); Uc = Vt[:K].T.contiguous(); del dev
    for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND):
        SUB['mlp_mean'][L] = torch.cat(capO[L], 0).mean(0); capO[L] = []
    nat_bigram = class_bigram(natural, cls_of)

    hooks = [H[L].mlp.register_forward_hook(mlp_mean_hook(L)) for L in set(CONTENT_BAND + GRAMMAR_BAND + RAND_BAND)]
    prompt_cc = content_coords(prompts, xb_pool, Uc, V)

    conds = {'base': (set(), False), 'content_abl': (set(CONTENT_BAND), False),
             'grammar_abl': (set(GRAMMAR_BAND), False), 'random2_abl': (set(RAND_BAND), False),
             'vres_off': (set(), True)}
    # content-word mask: alphabetic, not function class (labels 0-4 = function; 8 = lowercase word; 7 = Capitalized)
    content_tok = ((cls_of == 8) | (cls_of == 7)).float()
    # legal transitions: top-mass class pairs of natural prose
    legal = nat_bigram > 0.02
    def cw_rate(seqs): return float(content_tok[seqs].mean())
    def legal_rate(seqs):
        c = cls_of[seqs]
        return float(legal[c[:, :-1].reshape(-1), c[:, 1:].reshape(-1)].float().mean())
    nat_cw = cw_rate(natural); nat_lr = legal_rate(natural)
    print(f"natural: content-word rate {nat_cw:.3f} | legal-transition {nat_lr:.3f}", flush=True)
    res = {}; texts = {}
    for name, (act, voff) in conds.items():
        gen = generate(prompts, act, voff, seed=123)
        cc = content_coords(gen, xb_pool, Uc, V)
        topic = float(F.cosine_similarity(cc, prompt_cc, dim=-1).mean())
        res[name] = {'topic_retention': round(topic, 4), 'rep4gram': round(rep4(gen), 4),
                     'class_kl': round(bigram_kl(class_bigram(gen, cls_of), nat_bigram), 4),
                     'content_word_rate': round(cw_rate(gen), 4), 'legal_transition': round(legal_rate(gen), 4)}
        texts[name] = [enc.decode(r) for r in gen.tolist()[:24]]
        print(f"{name:>12}: topic {res[name]['topic_retention']} | cw-rate {res[name]['content_word_rate']} | legal {res[name]['legal_transition']} | rep4 {res[name]['rep4gram']}", flush=True)
    for h in hooks: h.remove()

    b = res['base']
    def drop(nm, key): return b[key] - res[nm][key]
    cw_ratio = drop('content_abl', 'content_word_rate')/max(drop('grammar_abl', 'content_word_rate'), 1e-4)
    lg_ratio = drop('grammar_abl', 'legal_transition')/max(drop('content_abl', 'legal_transition'), 1e-4)
    out = {'natural': {'content_word_rate': round(nat_cw, 4), 'legal_transition': round(nat_lr, 4)},
           'conditions': res, 'texts': texts,
           'cw_drop_ratio_content_over_grammar': round(cw_ratio, 2),
           'legal_drop_ratio_grammar_over_content': round(lg_ratio, 2)}
    out['pred_a_clean_dissociation'] = bool(cw_ratio >= 2 and lg_ratio >= 2)
    out['pred_b_vres_signature'] = bool(res['vres_off']['legal_transition'] < b['legal_transition'] - 0.05
                                        and res['vres_off']['rep4gram'] <= 0.005)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"cw-drop ratio (content/grammar) {cw_ratio:.2f} | legal-drop ratio (grammar/content) {lg_ratio:.2f}", flush=True)
    print(f"pred_a clean-dissociation {out['pred_a_clean_dissociation']} | pred_b vres-signature {out['pred_b_vres_signature']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
