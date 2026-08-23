"""Registered refinement of §1096: the cross-register table-transfer ratios (~0.5 prose->code; 0.145 code->prose
for mlp0) conflate "different computation" with "vocabulary coverage" (tables fall back to global mean on unseen
tokens; the code corpus has a narrow vocab). Here: identical design, but CE is evaluated ONLY at positions whose
CURRENT token occurs >= 8 times in BOTH build corpora (proseA and codeA) — every table entry well-sampled on both
sides. If the transfer ratio rises to ~1 on the matched vocab, the front grammar computation is register-GENERAL
and §1096's ~0.5 was coverage; if it stays ~0.5, the computation itself is register-specific per token.

REGISTERED PREDICTIONS:
  (0) SANITY: same-register recoveries on the matched-vocab positions >= the §1096 values (easier subset);
      matched-vocab positions are a substantial fraction (> 30%) of code positions.
  (a) COVERAGE WAS THE GAP: matched-vocab transfer ratios >= 0.75 in BOTH directions and both layers -> the
      per-token grammar function is the SAME across registers where both registers use the token; §1096's ~0.5
      was sampling, and the honest statement becomes 'grammar computation register-general, usage register-specific';
  (b) COMPUTATION DIFFERS: if matched-vocab ratios stay < 0.6, the same token gets a genuinely different
      front computation per register (context-dependence beyond the token table) — report plainly."""
import json, time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_transfer_vmatch_results.json'
NSEQ = 96; SEQ = 256; LAYERS = [0, 1]
H = m.transformer.h
SUB = {'layer': -1, 'table': None, 'obar': None}
CUR = {}


def fwd(idx):
    CUR['tok'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(L):
    def h(mo, i_, o_):
        if SUB['layer'] != L or SUB['table'] is None: return None
        if isinstance(SUB['table'], str) and SUB['table'] == 'meanabl':
            return SUB['obar'].view(1, 1, D).expand_as(o_).to(o_.dtype)
        return SUB['table'][CUR['tok']].to(o_.dtype)
    return h


def load_code(nseq, seq):
    enc = tiktoken.get_encoding('gpt2'); toks = []
    for fp in sorted(glob.glob('/workspace/tensor_language/**/*.py', recursive=True)):
        try: toks.extend(enc.encode(open(fp).read()))
        except Exception: continue
        if len(toks) >= nseq*seq: break
    return torch.tensor(toks[:nseq*seq], dtype=torch.long).view(nseq, seq)


@torch.no_grad()
def build_tables(blocks):
    """per-token mean OUTPUT tables for mlp0/mlp1 + output means, from a corpus."""
    V = int(m.lm_head.weight.shape[0])
    caps = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        def mk(L):
            def h(mo, i_, o_): caps[L].append(o_.detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mk(L)))
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    tabs = {}; obars = {}; seen = cn > 0
    for L in LAYERS:
        O = torch.cat(caps[L], 0); caps[L] = []
        ob = torch.zeros(V, D, device=DEV); ob.index_add_(0, tok, O)
        tab = ob / cn.clamp_min(1).unsqueeze(1)
        glob_mean = O.mean(0)
        tab[~seen] = glob_mean          # unseen tokens -> global mean
        tabs[L] = tab.half(); obars[L] = glob_mean
        del O
    return tabs, obars, cn


VMASK = {'ok': None}   # V-sized bool: token well-sampled in BOTH build corpora


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.reshape(-1).shape[0], device=DEV), tgt]
        keepm = VMASK['ok'][idx.reshape(-1)]
        tot += float(ce_tok[keepm].sum()); n += int(keepm.sum())
    return tot/max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    g = torch.Generator().manual_seed(0)
    prose = cl.fineweb_rows(NSEQ*2)[:, :SEQ].contiguous()
    proseA = prose[:NSEQ]; proseB = prose[NSEQ:]     # build on A, eval on B (held-out within register)
    code = load_code(NSEQ*2, SEQ)
    codeA = code[:NSEQ]; codeB = code[NSEQ:]
    V = int(m.lm_head.weight.shape[0])

    tabs = {}
    tabs['prose'], obars_p, cnt_p = build_tables(proseA)
    tabs['code'], obars_c, cnt_c = build_tables(codeA)
    VMASK['ok'] = (cnt_p >= 8) & (cnt_c >= 8)
    frac_code = float(VMASK['ok'][codeB[:, :-1].to(DEV).reshape(-1)].float().mean())
    frac_prose = float(VMASK['ok'][proseB[:, :-1].to(DEV).reshape(-1)].float().mean())
    print(f"matched-vocab position fraction: code {frac_code:.3f} | prose {frac_prose:.3f}", flush=True)
    # shuffled-token null (prose tables, permuted vocab rows)
    perm = torch.randperm(V, generator=g)
    tabs['shuffled'] = {L: tabs['prose'][L][perm.to(DEV)] for L in LAYERS}

    hs = [H[L].mlp.register_forward_hook(sub_hook(L)) for L in LAYERS]
    out = {'eval': {}}
    for ev_name, ev_blocks, obars in [('code', codeB, obars_c), ('prose', proseB, obars_p)]:
        SUB['layer'] = -1; base = ce(ev_blocks)
        rows = {}
        for L in LAYERS:
            row = {}
            for src in ['prose', 'code', 'shuffled', 'meanabl']:
                SUB['layer'] = L
                SUB['table'] = 'meanabl' if src == 'meanabl' else tabs[src][L]
                SUB['obar'] = obars[L]
                row[src] = round(ce(ev_blocks) - base, 4)
                SUB['layer'] = -1; SUB['table'] = None
            abl = max(row['meanabl'], 1e-6)
            same = 'code' if ev_name == 'code' else 'prose'
            cross = 'prose' if ev_name == 'code' else 'code'
            row['same_recov'] = round(1 - row[same]/abl, 3)
            row['cross_recov'] = round(1 - row[cross]/abl, 3)
            row['shuffled_recov'] = round(1 - row['shuffled']/abl, 3)
            row['transfer_ratio'] = round(row['cross_recov']/max(row['same_recov'], 1e-6), 3)
            rows[f'mlp{L}'] = row
            print(f"eval={ev_name} mlp{L}: same {row[same]} (recov {row['same_recov']}) | cross {row[cross]} (recov {row['cross_recov']}) | shuf {row['shuffled']} ({row['shuffled_recov']}) | abl {row['meanabl']} | transfer {row['transfer_ratio']}", flush=True)
        out['eval'][ev_name] = {'base_ce': round(base, 4), **rows}
    for h in hs: h.remove()

    tr = [out['eval'][ev][f'mlp{L}']['transfer_ratio'] for ev in ('code', 'prose') for L in LAYERS]
    out['transfer_ratios'] = tr
    out['matched_frac_code'] = round(frac_code, 3); out['matched_frac_prose'] = round(frac_prose, 3)
    out['pred_a_coverage_was_gap'] = bool(min(tr) >= 0.75)
    out['pred_b_computation_differs'] = bool(max(tr) < 0.6)
    out['asymmetry_prose_to_code_minus_code_to_prose'] = round(
        sum(out['eval']['code'][f'mlp{L}']['transfer_ratio'] for L in LAYERS)/2 -
        sum(out['eval']['prose'][f'mlp{L}']['transfer_ratio'] for L in LAYERS)/2, 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"matched-vocab transfer ratios {tr} | pred_a coverage-was-gap {out['pred_a_coverage_was_gap']} | pred_b computation-differs {out['pred_b_computation_differs']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
