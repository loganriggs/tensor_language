"""Is the CONTENT bottleneck (the hard 77% of loss, §831/§880) FUNDAMENTAL to the task or CAPACITY-limited? Compare
within the FAMILY (same FineWeb training data, same recipe, same GPT-2 tokenizer): bilin18 (18L/1152) vs bilin12
(12L/768). Chain-rule split of next-token CE into CLASS-CE (grammar) + WITHIN-class-CE (content). If the bigger
model has LOWER content-CE (and similar low grammar-CE), the content bottleneck is partly CAPACITY; if content-CE
is similar, it is task-fundamental. Same forward (30*tanh clamp) for both; full-CE sanity-checked.

REGISTERED PREDICTIONS:
  (0) SANITY: full CE is reasonable (~3-4.5 nats) for both (else the output clamp differs and the CE is invalid).
  (a) CONTENT IS PARTLY CAPACITY: the smaller bilin12 has HIGHER within-class (content) CE than bilin18, while
      class (grammar) CE is low and similar for both -> the content bottleneck shrinks with capacity (not fully
      task-fundamental); grammar is solved by both;
  (b) report class-CE and within-CE for bilin18 and bilin12 + the content-CE gap."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
import census_lib as cl
from bilin18_joint_removal import m as BILIN, DEV
from tier2_model import load_elriggs
import torch.nn.functional as F

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_bottleneck_capacity_results.json'
NEVAL = 160; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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
def forward_logits(mdl, idx, Dm):
    x = F.rms_norm(mdl.transformer.wte(idx), (Dm,)); x0 = x; v1 = None
    for blk in mdl.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(mdl.lm_head(F.rms_norm(x, (Dm,)))/30.0)


@torch.no_grad()
def split_ce(mdl, blocks, Dm, cidx, C):
    Cmat = F.one_hot(cidx, C).float()
    tot_ce = 0.0; tot_cls = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(mdl, idx, Dm).float(), -1)
        tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12)  # class marginal
        lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot_ce += float(-lp_tok.sum()); tot_cls += float(-lp_cls.sum()); n += tf.shape[0]
    full = tot_ce/n; classce = tot_cls/n; within = full - classce
    return {'full_ce': round(full, 4), 'class_ce': round(classce, 4), 'within_ce': round(within, 4),
            'content_frac': round(within/full, 3)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous()
    V = int(BILIN.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    out = {'models': {}}
    out['models']['bilin18'] = split_ce(BILIN, blocks, 1152, cidx, C); print(f"bilin18: {out['models']['bilin18']}", flush=True)
    for short in ['bilin12']:
        try:
            mdl, cfg = load_elriggs(short); Dm = cfg.get('n_embd')
            out['models'][short] = split_ce(mdl, blocks, Dm, cidx, C); print(f"{short}: {out['models'][short]}", flush=True)
            del mdl; torch.cuda.empty_cache()
        except Exception as e:
            out['models'][short] = {'error': f'{type(e).__name__}: {e}'}; print(f"{short} ERROR {e}", flush=True)
    b18 = out['models']['bilin18']; b12 = out['models'].get('bilin12', {})
    if 'within_ce' in b12:
        out['content_ce_gap_smaller_minus_bigger'] = round(b12['within_ce'] - b18['within_ce'], 4)
        out['class_ce_gap'] = round(b12['class_ce'] - b18['class_ce'], 4)
        out['sanity_full_ce_ok'] = bool(3.0 < b18['full_ce'] < 5.0 and 3.0 < b12['full_ce'] < 6.0)
        out['pred_a_content_capacity'] = bool(out['sanity_full_ce_ok'] and out['content_ce_gap_smaller_minus_bigger'] > 0.1 and abs(out['class_ce_gap']) < out['content_ce_gap_smaller_minus_bigger'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-CE gap (bilin12 - bilin18) = {out.get('content_ce_gap_smaller_minus_bigger')} | class-CE gap = {out.get('class_ce_gap')}", flush=True)
    print(f"(a) content bottleneck is partly capacity: {out.get('pred_a_content_capacity')}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
