"""NATURAL-TEXT confirmation of §1033 (grammar owns the head / content owns the tail). §1033 showed under INJECTION
that content barely moves the top-5. Here confirm on NATURAL text without any intervention: for each real next-token
target, compute its RANK in the model's predicted distribution (0 = argmax), and split by the target's class. If
grammar lives in the head and content in the tail, then FUNCTION-word targets (grammar) should have LOW rank (the
model puts them near the top) while CONTENT-word targets should have MUCH HIGHER rank (deep in the tail), even when
the model predicts them correctly in expectation.

REGISTERED PREDICTIONS:
  (0) SANITY: overall median rank is small-ish (the model is good); ranks computed on real next tokens.
  (a) CONTENT IN THE TAIL: the median rank of CONTENT-word targets (word/cap/number) is MUCH higher than that of
      FUNCTION-word targets (det/prep/conj/pron/punct) -> content targets sit deep in the tail, grammar targets near
      the head, on natural text -- confirming §1033 without injection;
  (b) also report the fraction of each class's targets that fall in the top-5 (head) -> function >> content.
  Report median/mean rank and top-5 fraction per class group."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_target_rank_results.json'
NEVAL = 160; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}   # grammar/head classes
CONTENT = {'word', 'cap', 'number'}                    # content/tail classes
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}


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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous(); d = dec()
    V = int(m.lm_head.weight.shape[0])
    t2c = np.full(V, 7, np.int64)
    for t in np.unique(rows.cpu().numpy().reshape(-1)): t2c[int(t)] = CLASSES.index(classify(d(int(t))))
    is_func = torch.tensor([CLASSES[c] in FUNCTION for c in range(len(CLASSES))], device=DEV)
    func_ranks = []; cont_ranks = []; func_top5 = 0; func_n = 0; cont_top5 = 0; cont_n = 0
    cidx = torch.tensor(t2c, device=DEV)
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tf = tgt.reshape(-1); lgf = lg.reshape(-1, lg.shape[-1])
        tgt_logit = lgf[torch.arange(tf.shape[0], device=DEV), tf]
        rank = (lgf > tgt_logit.unsqueeze(1)).sum(1)  # 0 = argmax
        tgt_is_func = is_func[cidx[tf]]
        func_ranks.append(rank[tgt_is_func].cpu().numpy()); cont_ranks.append(rank[~tgt_is_func].cpu().numpy())
        func_top5 += int((rank[tgt_is_func] < 5).sum()); func_n += int(tgt_is_func.sum())
        cont_top5 += int((rank[~tgt_is_func] < 5).sum()); cont_n += int((~tgt_is_func).sum())
    fr = np.concatenate(func_ranks); cr = np.concatenate(cont_ranks)
    out = {'function': {'median_rank': int(np.median(fr)), 'mean_rank': round(float(fr.mean()), 1), 'top5_frac': round(func_top5/max(func_n, 1), 3), 'n': func_n},
           'content':  {'median_rank': int(np.median(cr)), 'mean_rank': round(float(cr.mean()), 1), 'top5_frac': round(cont_top5/max(cont_n, 1), 3), 'n': cont_n}}
    out['pred_a_content_in_tail'] = bool(out['content']['median_rank'] > 3 * max(out['function']['median_rank'], 1) and out['content']['median_rank'] > out['function']['median_rank'] + 3)
    out['pred_b_top5_func_gt_content'] = bool(out['function']['top5_frac'] > out['content']['top5_frac'] + 0.15)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"FUNCTION targets: median rank {out['function']['median_rank']} mean {out['function']['mean_rank']} top5 {out['function']['top5_frac']} (n {func_n})", flush=True)
    print(f"CONTENT targets:  median rank {out['content']['median_rank']} mean {out['content']['mean_rank']} top5 {out['content']['top5_frac']} (n {cont_n})", flush=True)
    print(f"pred_a content-in-tail {out['pred_a_content_in_tail']} | pred_b top5 func>content {out['pred_b_top5_func_gt_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
