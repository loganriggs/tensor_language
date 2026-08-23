"""OOD Open-C thread (FINDINGS; dossier mlp-front-grammar/mlp-deep-content): §1080 showed the grammar
REPRESENTATION is register-general (prose<->code front subspace overlap 0.41 vs content 0.20) — but that is a
subspace fact. CAUSAL version: is the grammar COMPUTATION register-general — does prose-built grammar FUNCTION
on code? The front MLPs are ~static per-token functions (mlp1 held-out tok recovery 0.93, §1088), so the
computation is transportable as a TABLE: build per-token output tables for mlp0/mlp1 from PROSE runs, substitute
them into runs on CODE (and vice versa), against (i) same-register tables (ceiling), (ii) shuffled-token tables
(null), (iii) mean-ablation (floor). If prose tables ~ code tables ON CODE, the grammar machine's function
transfers across registers (extends §1080 representation -> function).

REGISTERED PREDICTIONS:
  (0) SANITY: same-register tables reproduce ~0.93 recovery (mlp1, §1088-level); shuffled tables ~ floor or worse.
  (a) GRAMMAR FUNCTION TRANSFERS: cross-register tables recover >= 80% of what same-register tables recover on
      the SAME eval register, for BOTH mlp0 and mlp1, both directions (prose->code and code->prose) -> the front
      grammar computation is register-general causally, not just representationally;
  (b) ASYMMETRY: transfer is better prose->code than code->prose if the code register is a subset/special case
      of prose token usage (report the asymmetry either way);
  (c) if cross-register recovery < 50% of same-register, grammar FUNCTION is register-specific despite the
      §1080 representational overlap (representation-vs-function dissociation — report plainly)."""
import json, time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_transfer_code_results.json'
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
    return tabs, obars, seen


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


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
    tabs['prose'], obars_p, _ = build_tables(proseA)
    tabs['code'], obars_c, _ = build_tables(codeA)
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
    out['pred_a_grammar_transfers'] = bool(min(tr) >= 0.8)
    out['pred_c_register_specific'] = bool(max(tr) < 0.5)
    out['asymmetry_prose_to_code_minus_code_to_prose'] = round(
        sum(out['eval']['code'][f'mlp{L}']['transfer_ratio'] for L in LAYERS)/2 -
        sum(out['eval']['prose'][f'mlp{L}']['transfer_ratio'] for L in LAYERS)/2, 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"transfer ratios {tr} | pred_a transfers {out['pred_a_grammar_transfers']} | asym {out['asymmetry_prose_to_code_minus_code_to_prose']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
