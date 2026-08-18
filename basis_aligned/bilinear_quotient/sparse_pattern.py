"""SPARSE-PATTERN ALIGNMENT -- if the weights are honestly sparse
(census v2/v3), WHERE are the hot entries? Two claims worth testing:
(1) within a family, the top-entry masks of different layers overlap
    beyond chance (a repeated wiring pattern, the decomposition-level
    motif); (2) hot INPUT coordinates concentrate on privileged stream
    coordinates shared across families (the flat track's tail-coords
    anatomy predicts yes).
Method: per matrix, top-1% entry mask; within-family pairwise mask
overlap (Jaccard) vs the 1% chance floor; per-family hot-column
histogram over stream coordinates; cross-family rank correlation of
hot-column mass.
REGISTERED PREDICTIONS:
  (a) within-family mean Jaccard >= 3x the 0.01 chance floor for >=5/7
      families;
  (b) cross-family hot-coordinate Spearman >= 0.3 for >= 15 of 21
      family pairs (privileged coordinates are global);
  (c) top-32 hot coordinates reported (for comparison with the known
      span/tail coordinates)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sparse_pattern_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    FAMS={'c_q':lambda b:b.attn.c_q.weight,
          'c_k':lambda b:b.attn.c_k.weight,
          'c_v':lambda b:b.attn.c_v.weight,
          'c_proj':lambda b:b.attn.c_proj.weight,
          'Left':lambda b:b.mlp.Left.weight,
          'Right':lambda b:b.mlp.Right.weight,
          'Down':lambda b:b.mlp.Down.weight}
    # input coordinate = stream side: columns for c_q/c_k/c_v/Left/Right
    # (shape *,D); for c_proj/Down the stream side is rows/output
    STREAM_IN={'c_q':1,'c_k':1,'c_v':1,'Left':1,'Right':1,
               'c_proj':0,'Down':0}
    jac={}; hot={}
    for fname,getw in FAMS.items():
        masks=[]
        hv=torch.zeros(D,device=DEV)
        for li in range(18):
            Wm=getw(m.transformer.h[li]).detach().float()
            n=Wm.numel(); k=n//100
            th=Wm.abs().flatten().kthvalue(n-k).values
            mk=(Wm.abs()>th)
            masks.append(mk)
            ax=STREAM_IN[fname]
            hv+=(Wm*mk).abs().sum(1-ax if Wm.shape[ax]!=D else 1-ax)\
                if False else 0
            # hot mass per stream coordinate:
            if ax==1 and Wm.shape[1]==D:
                hv+=(Wm*mk).abs().sum(0)
            elif ax==0 and Wm.shape[0]==D:
                hv+=(Wm*mk).abs().sum(1)
        hot[fname]=hv
        js=[]
        for a in range(18):
            for b in range(a+1,18):
                if masks[a].shape!=masks[b].shape: continue
                inter=(masks[a]&masks[b]).sum()
                uni=(masks[a]|masks[b]).sum()
                js.append(float(inter)/float(uni))
        jac[fname]=sum(js)/len(js)
        print(f'{fname:7s}: mean Jaccard {jac[fname]:.4f} '
              f'(chance ~0.005)',flush=True)
    def srank(v):
        s=sorted(range(len(v)),key=lambda i:v[i])
        r=[0]*len(v)
        for j,i in enumerate(s): r[i]=j
        return torch.tensor(r,dtype=torch.float)
    fams=list(FAMS)
    npair=0; nok=0; rhos={}
    for a in range(len(fams)):
        for b in range(a+1,len(fams)):
            ra=srank(hot[fams[a]].tolist()); rb=srank(hot[fams[b]].tolist())
            ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
            rho=float((ra*rb).mean())
            rhos[f'{fams[a]}-{fams[b]}']=round(rho,3)
            npair+=1
            if rho>=0.3: nok+=1
    tot=sum(hot.values())
    top32=torch.topk(tot,32).indices.tolist()
    ja=sum(1 for f in fams if jac[f]>=0.03)
    pa=ja>=5; pb=nok>=15
    out={'jaccard':{k:round(v,4) for k,v in jac.items()},
         'cross_family_rho':rhos,'top32_coords':top32,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'cross-family rho>=0.3: {nok}/{npair}')
    print(f'top-8 hot stream coords: {top32[:8]}')
    print(f"(a) Jaccard >= 3x chance ({ja}/7): {'HELD' if pa else 'FAILED'}")
    print(f"(b) global privileged coords ({nok}/21): "
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
