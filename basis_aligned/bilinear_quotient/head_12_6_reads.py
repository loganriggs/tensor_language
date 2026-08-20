"""HEAD 12.6 READS -- 488: layer 12's match-specific long-range
cost is one head. Windowing head 12.6 to four tokens costs +0.1770
at match positions against +0.0147 elsewhere -- 84.5% of the
layer's total and a twelve-fold match/non-match ratio, with the
median layer-12 head at +0.0032 (487).
487's bar (b) asked whether 12.6 reads the SAME TOKEN as the query
and found only 6.0% against a 3.3% null. But that bar tested the
wrong thing, and the error is mine: an induction-style head does
not read the repeated token, it reads the token that FOLLOWED it
last time -- position p+1 where p is the previous occurrence. The
program's own head census calls that motif "induction-target".
Re-measure with the right probe, and characterise the head either
way.
At match positions, classify 12.6's top read as: the successor of
a previous occurrence (p+1), the previous occurrence itself (p),
local (offset >= -4), position 0, or other -- each against a
frequency-matched random-read null.
REGISTERED PREDICTIONS:
  (a) INDUCTION-TARGET: >= 25% of top reads land on the successor
      of a previous occurrence, against a null under 5%;
  (b) if (a) fails, the offset and class histogram is reported and
      the head is characterised by what it DOES read -- recorded
      either way;
  (c) CONTROL: the median layer-12 head shows successor-reading
      under 10%."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_12_6_reads_results.json'
NR=16
HEADS=[6,3]     # 12.6 and a median control head

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    stats={h:{'succ':0,'prev':0,'local':0,'pos0':0,'other':0,
              'n':0,'null_succ':0,'off':{}} for h in HEADS}
    g=torch.Generator().manual_seed(11)
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        hh=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hh.remove()
        X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        for hd in HEADS:
            def r3(w):
                return are(F.rms_norm(w(X).view(B,T,9,128),
                           (128,))[:,:,hd][:,:,None],cos,sin)[:,:,0]
            qf,kf=r3(at.c_q),r3(at.c_k)
            q2,k2=r3(at.c_q2),r3(at.c_k2)
            pat=((torch.einsum('bqd,bkd->bqk',qf.float(),
                               kf.float())/128)
                 *(torch.einsum('bqd,bkd->bqk',q2.float(),
                                k2.float())/128)) \
                *torch.tril(torch.ones(T,T,device=DEV))
            s=stats[hd]
            for b in range(B):
                toks=ROWS[i+b,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]
                    prev=last.get(t)
                    last[t]=q
                    if prev is None or prev+1>=q or q<8: continue
                    k=int(pat[b,q,:q].abs().argmax())
                    s['n']+=1
                    o=k-q
                    s['off'][o]=s['off'].get(o,0)+1
                    if k==prev+1: s['succ']+=1
                    elif k==prev: s['prev']+=1
                    elif o>=-4: s['local']+=1
                    elif k==0: s['pos0']+=1
                    else: s['other']+=1
                    kr=int(torch.randint(0,q,(1,),generator=g))
                    s['null_succ']+=int(kr==prev+1)
        print(f'batch {i} done',flush=True)
    out={}
    for hd in HEADS:
        s=stats[hd]; n=max(s['n'],1)
        out[f'12.{hd}']={
            'n':s['n'],
            'successor_rate':round(s['succ']/n,3),
            'previous_occurrence_rate':round(s['prev']/n,3),
            'local_rate':round(s['local']/n,3),
            'pos0_rate':round(s['pos0']/n,3),
            'other_rate':round(s['other']/n,3),
            'null_successor_rate':round(s['null_succ']/n,3),
            'top_offsets':sorted(s['off'].items(),
                                 key=lambda kv:-kv[1])[:6]}
        print(f"12.{hd}: {out[f'12.{hd}']}",flush=True)
    a=out['12.6']
    pa=(a['successor_rate']>=0.25 and
        a['null_successor_rate']<0.05)
    pc=out['12.3']['successor_rate']<0.10
    out.update({'pred_a':bool(pa),'pred_b':True,'pred_c':bool(pc),
                'runtime_s':time.time()-t0})
    for nm,v in (('a','12.6 is an induction-target reader'),
                 ('b','histogram reported either way'),
                 ('c','control head is not')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
