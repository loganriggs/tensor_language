"""LAYERS 2-5, attention scan: what does each attn2..attn5 WRITE (incl. attn5, the biggest
uncharacterized component, 1.97 nats)? Decode current/prev/prevprev/induction-target + grammatical class
from each attn output. LAYER 1 orig doc:
LAYER 1, attention: what does attn1 WRITE, given layer 0 made the previous token available? (bottom-up
step; attn1 is the biggest attention, 2.22 nats). Census (item 10) says attn0+attn1 build the copy-source
with an "induction-target" motif. Test what attn1's output encodes by linear decode: current token,
previous token (extends copy-source?), PREV-PREV token, and the INDUCTION TARGET — the token that
followed the current token's PREVIOUS occurrence in the sequence (the classic induction signal that
predicts the next token). Compare output vs input vs shuffled null.

REGISTERED PREDICTIONS:
  (0) SANITY: current token decodes high from input;
  (a) if the INDUCTION TARGET decodes from attn1 OUTPUT far above its input and the null, attn1 does
      induction — it writes "what followed this token last time" (a next-token predictor);
  (b) if instead PREV or PREV-PREV jumps, attn1 extends the copy-source deeper rather than doing
      induction; report all four decodes (output vs input vs null)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_scan_front_results.json'
NEVAL = 240; T = 256; TOPV = 200; LAYERS = [2,3,4,5]


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, LAYER):
    attn = m.transformer.h[LAYER].attn; Ins = []; Outs = []; seqs = []
    def pre(mo, args): Ins.append(args[0].detach().float().reshape(-1, D))
    def post(mo, i_, o_): Outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hp = attn.register_forward_pre_hook(pre); ho = attn.register_forward_hook(post)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hp.remove(); ho.remove()
    return torch.cat(Ins, 0), torch.cat(Outs, 0), np.concatenate(seqs, 0)   # seqs: (nseq, T)


def induction_target(seqs):
    """for each (s,i): token that followed the previous occurrence of seqs[s,i] (i.e. last j<i with
    seqs[s,j-1]==seqs[s,i], target=seqs[s,j]); -1 if none."""
    nseq, TT = seqs.shape; tgt = np.full((nseq, TT), -1, dtype=np.int64)
    for s in range(nseq):
        last = {}                    # token -> position it last appeared at (as the 'previous' token)
        row = seqs[s]
        for i in range(TT):
            t = row[i]
            if t in last: tgt[s, i] = row[last[t]+1] if last[t]+1 < TT else -1
            # record: token at i-1 -> i (so if a future token == row[i-1], its follower is row[i])
            if i >= 1: last[row[i-1]] = i-1
    return tgt


def decode_acc(Ft, labels, valid, ncls, seed=0):
    idx = np.where(valid)[0]; rng = np.random.RandomState(seed); rng.shuffle(idx)
    ntr = int(0.7*len(idx)); tr, te = idx[:ntr], idx[ntr:]
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(labels[tr], device=DEV)] = 1.0
    A = Ft[tr].T @ Ft[tr] + 1e2*torch.eye(D, device=DEV)
    Wp = torch.linalg.solve(A, Ft[tr].T @ Y)
    pred = (Ft[te] @ Wp).argmax(1).cpu().numpy()
    return float((pred == labels[te]).mean())


CLASSES = ['det','prep','conj','pron','number','punct','cap','word']
DET={'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP={'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ={'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON={'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
def _dec():
    import tiktoken; enc=tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])
def _classify(x):
    t=x.strip()
    if t=='' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low=t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'

@torch.no_grad()
def main():
    t0=time.time(); cl.use_state(PT+'census_state_diverse.pt')
    rows=cl.fineweb_rows(NEVAL); dcd=_dec()
    out={'layers':{}}
    for LAYER in LAYERS:
        In,Out,seqs=capture(rows,LAYER)
        cur=seqs.reshape(-1)
        prev=np.full_like(seqs,-1); prev[:,1:]=seqs[:,:-1]; prev=prev.reshape(-1)
        pp=np.full_like(seqs,-1); pp[:,2:]=seqs[:,:-2]; pp=pp.reshape(-1)
        ind=induction_target(seqs).reshape(-1)
        uniq,cnts=np.unique(cur,return_counts=True); topv=set(uniq[np.argsort(-cnts)[:TOPV]].tolist())
        remap={t:i for i,t in enumerate(sorted(topv))}; lbl=lambda a:np.array([remap.get(int(t),-1) for t in a])
        cur_l,prev_l,pp_l,ind_l=lbl(cur),lbl(prev),lbl(pp),lbl(ind)
        clsl=np.array([CLASSES.index(_classify(dcd(int(t)))) for t in cur])
        r={}
        r['current_out']=round(decode_acc(Out,cur_l,cur_l>=0,TOPV),3)
        r['prev_out']=round(decode_acc(Out,prev_l,prev_l>=0,TOPV),3)
        r['prevprev_out']=round(decode_acc(Out,pp_l,pp_l>=0,TOPV),3)
        r['induction_out']=round(decode_acc(Out,ind_l,ind_l>=0,TOPV),3)
        r['induction_in']=round(decode_acc(In,ind_l,ind_l>=0,TOPV),3)
        r['class_out']=round(decode_acc(Out,clsl,np.ones_like(clsl,bool),len(CLASSES)),3)
        out['layers'][f'attn{LAYER}']=r
        print(f"attn{LAYER}: cur {r['current_out']} prev {r['prev_out']} prevprev {r['prevprev_out']} induction {r['induction_out']}(in {r['induction_in']}) class {r['class_out']}",flush=True)
    out['runtime_s']=round(time.time()-t0,1)
    json.dump(out,open(OUT,'w'),indent=1); print('wrote',OUT)

if __name__=='__main__':
    main()
