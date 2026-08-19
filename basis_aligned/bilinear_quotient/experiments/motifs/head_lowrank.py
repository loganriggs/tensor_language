"""HEAD LOW-RANK -- backlog item 5: is the pattern machinery of motif
heads a low-rank object at the WEIGHT level? Truncate per-head score
projections (c_q,c_k,c_q2,c_k2 rows) vs value machinery (c_v rows +
c_proj cols) to rank r in {8,16,32} of 128, per motif class (ind /
prev / self / diffuse, matched 9 heads each, seed-9 samples), measure
fresh CE. Weights-derived, transfer-free (the fold lesson). Prior:
double QK is a coincidence sharpener with one weak shared preference
(-> low-rank scores); OV subspaces sit at the random-spread floor
(-> value side should NOT compress).
REGISTERED PREDICTIONS (fresh, 120 never-seen rows):
  (a) ind-head QK rank-16 costs <= +0.02;
  (b) ind-head value rank-16 costs >= 3x its QK rank-16 cost;
  (c) diffuse QK rank-16 costs more than ind QK rank-16 (motif
      patterns compress; diffuse patterns resist); full rank curves
      for all four classes reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_lowrank_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    import tiktoken
    from datasets import load_dataset
    enc2=tiktoken.get_encoding('gpt2')
    ds=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    rows=[]
    for di in range(5000,10000):
        tk=enc2.encode_ordinary(ds[di]['text'])
        for s0 in range(0,len(tk)-513,513):
            row=tk[s0:s0+513]
            if tuple(row[:32]) in seen: continue
            rows.append(row)
            if len(rows)>=120: break
        if len(rows)>=120: break
    FR=torch.tensor(rows,dtype=torch.long)
    print(f'fresh rows {FR.shape[0]}',flush=True)
    def ceF():
        ces=[]
        for i in range(0,120,4):
            bb=FR[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        return float(torch.cat(ces).mean())
    base=ceF()
    print(f'base fresh CE {base:.4f}',flush=True)
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    byc={}
    for li,hd,mo,fr in mt: byc.setdefault(mo,[]).append((li,hd))
    g=torch.Generator().manual_seed(9)
    SAMP={}
    for mo in ('ind','prev','self','diffuse'):
        hs=byc[mo]
        if len(hs)>9:
            pi=torch.randperm(len(hs),generator=g)[:9].tolist()
            hs=[hs[i] for i in pi]
        SAMP[mo]=hs
        print(f'{mo}: {hs}',flush=True)
    def lowrank(Msl,r):
        U,S,Vh=torch.linalg.svd(Msl.float(),full_matrices=False)
        return (U[:,:r]*S[:r])@Vh[:r]
    def apply(heads,side,r):
        saves=[]
        for li,hd in heads:
            at=m.transformer.h[li].attn
            a=hd*128
            if side=='qk':
                for md_ in (at.c_q,at.c_k,at.c_q2,at.c_k2):
                    W=md_.weight
                    saves.append((W,'r',a,W[a:a+128].clone()))
                    W[a:a+128]=lowrank(W[a:a+128],r).to(W.dtype)
            else:
                W=at.c_v.weight
                saves.append((W,'r',a,W[a:a+128].clone()))
                W[a:a+128]=lowrank(W[a:a+128],r).to(W.dtype)
                W2=at.c_proj.weight
                saves.append((W2,'c',a,W2[:,a:a+128].clone()))
                W2[:,a:a+128]=lowrank(W2[:,a:a+128],r).to(W2.dtype)
        return saves
    def restore(saves):
        for W,kind,a,old in saves:
            if kind=='r': W[a:a+128]=old
            else: W[:,a:a+128]=old
    RES={}
    for mo in ('ind','prev','self','diffuse'):
        for side in ('qk','v'):
            for r in (8,16,32):
                sv=apply(SAMP[mo],side,r)
                d=ceF()-base
                restore(sv)
                RES[f'{mo}_{side}_r{r}']=round(d,4)
                print(f'{mo} {side} rank{r}: {d:+.4f}',flush=True)
    pa=RES['ind_qk_r16']<=0.02
    pb=RES['ind_v_r16']>=3*max(RES['ind_qk_r16'],1e-4)
    pc=RES['diffuse_qk_r16']>RES['ind_qk_r16']
    out={'base':round(base,4),'deltas':RES,
         'samples':{k:v for k,v in SAMP.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) ind qk r16 <= +0.02: {'HELD' if pa else 'FAILED'}")
    print(f"(b) ind v r16 >= 3x qk: {'HELD' if pb else 'FAILED'}")
    print(f"(c) diffuse qk > ind qk at r16: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
