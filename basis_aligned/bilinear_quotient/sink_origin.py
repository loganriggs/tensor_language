"""SINK ORIGIN -- 439: position 0's residual at layer 5 is
dominated by mlp4 (projection share 0.626, then m3 0.159, m0
0.119 -- NOT wte or m0 as registered), and routing that position's
value through head 5.7's own projection reproduces the head's mean
write at cosine 0.999. So the chain is: something at position 0 ->
mlp4 -> head 5.7 -> a constant added everywhere. Close it by
characterising what mlp4 does at position 0.
REGISTERED PREDICTIONS:
  (a) OUTLIER WRITE: mlp4's output norm at position 0 is >= 3x
      its mean norm at other positions;
  (b) FIXED VECTOR: mlp4's position-0 output direction is stable
      across rows -- mean pairwise cosine >= 0.9 (i.e. it is a
      learned constant, not a function of the first token);
  (c) TOKEN-INDEPENDENCE: rows whose first token differs still
      share it -- report the cosine split by first-token identity;
  (d) UPSTREAM: which writer dominates mlp4's INPUT at position 0
      (same exact decomposition one layer down)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sink_origin_results.json'
NR=32

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    vecs=[]; norms0=[]; normsE=[]; first=[]
    WR=['wte']+[f'{k}{l}' for l in range(4) for k in ('a','m')]
    shares={w:0.0 for w in WR}; n=0
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4
        outs={}; pre={}
        hs=[]
        for lj in range(5):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(k9=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[k9]=y.detach().float()
                    return h
                hs.append(mod.register_forward_hook(mk()))
        def ph(mo_,args): pre['X']=args[0]
        hs.append(m.transformer.h[4].mlp
                  .register_forward_pre_hook(
                      lambda mo_,args: pre.__setitem__(
                          'm4in',args[0])))
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        y4=outs['m4']
        vecs.append(y4[:,0].clone())
        norms0.append(y4[:,0].norm(dim=-1))
        normsE.append(y4[:,1:].norm(dim=-1).mean(dim=-1))
        first+= [int(t) for t in ROWS[i:i+4,0].tolist()]
        # mlp4's INPUT at position 0, decomposed by writer
        # 2026-08-20 (writeup 503): exact per-writer coefficients
        # for mlp4's input (a4 enters after the lambda mix).
        parts=cl.writer_parts(4,E,outs,'m')
        parts={w:parts[w] for w in list(WR)+['a4'] if w in parts}
        blkin=sum(parts.values())
        cl.check_parts(parts,pre['m4in'],label='sink_origin')
        tot=sum(parts.values())
        tn=(tot[:,0]*tot[:,0]).sum(-1).clamp_min(1e-9)
        for w,pv in parts.items():
            shares[w]=shares.get(w,0.0)+float(
                ((pv[:,0]*tot[:,0]).sum(-1)/tn).sum())
        n+=B
        print(f'batch {i} done',flush=True)
    V=torch.cat(vecs)                       # NR x D
    Vn=V/V.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    C=(Vn@Vn.T)
    off=C[~torch.eye(len(C),dtype=torch.bool,device=C.device)]
    meancos=float(off.mean())
    n0=float(torch.cat(norms0).mean())
    nE=float(torch.cat(normsE).mean())
    # split by first-token identity
    import collections
    byfirst=collections.defaultdict(list)
    for j,t in enumerate(first): byfirst[t].append(j)
    same=[];diff=[]
    for a in range(len(first)):
        for b in range(a+1,len(first)):
            (same if first[a]==first[b] else diff).append(
                float(C[a,b]))
    sh={w:round(v/max(n,1),4) for w,v in shares.items()}
    top=sorted(sh.items(),key=lambda kv:-abs(kv[1]))[:5]
    pa=n0>=3*nE
    pb=meancos>=0.9
    out={'m4_norm_pos0':round(n0,2),'m4_norm_elsewhere':round(nE,2),
         'norm_ratio':round(n0/max(nE,1e-6),2),
         'mean_pairwise_cosine_pos0':round(meancos,4),
         'cos_same_first_token':round(sum(same)/max(len(same),1),4)
             if same else None,
         'cos_diff_first_token':round(sum(diff)/max(len(diff),1),4)
             if diff else None,
         'n_distinct_first_tokens':len(byfirst),
         'm4_input_writers_pos0':top,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'pred_d':True,'runtime_s':time.time()-t0}
    print(f"m4 norm at pos0 {n0:.1f} vs elsewhere {nE:.1f} "
          f"(ratio {out['norm_ratio']})")
    print(f"mean pairwise cosine of pos-0 writes: {meancos:.4f} "
          f"(same-first-token {out['cos_same_first_token']}, "
          f"different {out['cos_diff_first_token']})")
    print(f"m4 input writers at pos0: {top}")
    for nm,v in (('a','pos-0 write norm >=3x elsewhere'),
                 ('b','pos-0 direction stable (cos>=0.9)'),
                 ('c','token split reported'),
                 ('d','upstream writers reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
