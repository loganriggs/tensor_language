"""PROJECTION LAW BATCH -- 360's "only subtraction bites" was
established on ONE circuit (r.0.0.1) plus a head-level echo (353/4).
Generalize: for the 10 highest-concentration census leaves, compare
three interventions on the leaf's own bundles: (i) DELETE (coords->
0 via projection), (ii) ASSIGN slice-mean (coords->constant mean),
(iii) ASSIGN a random natural donor value (coords->constant sample).
All three remove position-to-position variance; they differ only in
the written constant. Control: delete a random matched-dim subspace.
REGISTERED PREDICTIONS (per-leaf member-mean |dCE|):
  (a) LAW: the written constant is causally void -- across the 10
      leaves, median pairwise agreement of (i)/(ii)/(iii) >= 0.75
      (min/max ratio of member |dCE|);
  (b) SPECIFICITY: each leaf's own-bundle deletion >= 3x the
      random-subspace control (median across leaves);
  (c) per-leaf table reported; any leaf violating (a) by 2x is
      named (candidate value-sensitive circuit -- would refute the
      law's universality)."""
import json, time, torch
import census_lib as cl
from bilin18_joint_removal import orth, DEV
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'projection_law_batch_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    eb=json.load(open(PT+'explainer_batch_results.json'))
    leaves=sorted(eb['leaves'],key=lambda r:-r['conc'])[:10]
    bv=cl.base_ce()
    res=[]
    g=torch.Generator().manual_seed(11)
    for r in leaves:
        tag=r['tag']; lf=cl.leaf(tag)
        probes=[eval(p) for p in lf['top_probes']]
        mem=lf['member']; sl=lf['slice']
        PER={}
        for _,key,stag,blk in probes:
            PER.setdefault(key,[]).append(cl.pca_block(key,stag,blk))
        def hooks_mode(mode):
            hs=[]
            for key,vs in PER.items():
                P=orth(torch.cat(vs).T)   # D x k
                Y=cl.capture_out(key).float()
                if mode=='delete': val=torch.zeros(P.shape[1],device=DEV)
                elif mode=='mean': val=(Y[sl]@P.cpu()).mean(0).to(DEV)
                elif mode=='donor':
                    di=int(mem[int(torch.randint(len(mem),(1,),
                                                 generator=g))])
                    val=(Y[di]@P.cpu()).to(DEV)
                elif mode=='rand':
                    g2=torch.Generator().manual_seed(12)
                    R=torch.linalg.qr(torch.randn(P.shape[0],P.shape[1],
                                                  generator=g2))[0].to(DEV)
                    P=R; val=torch.zeros(P.shape[1],device=DEV)
                def fh(mo,i_,o_,P=P,val=val,isa=(key[0]=='a')):
                    if isa:
                        y,v1=o_
                        yf=y.float().reshape(-1,y.shape[-1])
                        yn=yf-((yf@P)-val[None,:])@P.T
                        return (yn.view(y.shape).to(y.dtype),v1)
                    yf=o_.float().reshape(-1,o_.shape[-1])
                    yn=yf-((yf@P)-val[None,:])@P.T
                    return yn.view(o_.shape).to(o_.dtype)
                hs.append(cl.MODS[key].register_forward_hook(fh))
            return hs
        row={'tag':tag}
        for mode in ('delete','mean','donor','rand'):
            d=cl.ce_sweep(hooks_mode(mode))-bv
            row[mode]=round(float(d[mem].abs().mean()),3)
        vals=[row['delete'],row['mean'],row['donor']]
        row['agree']=round(min(vals)/max(max(vals),1e-4),3)
        row['spec']=round(row['delete']/max(row['rand'],1e-4),2)
        res.append(row)
        print(row,flush=True)
    ag=sorted(r_['agree'] for r_ in res)[len(res)//2]
    sp=sorted(r_['spec'] for r_ in res)[len(res)//2]
    viol=[r_['tag'] for r_ in res if r_['agree']<0.5]
    pa=ag>=0.75; pb=sp>=3
    out={'leaves':res,'median_agree':ag,'median_spec':sp,
         'violators':viol,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':True}
    print(f'median agree {ag} | median specificity {sp}x | '
          f'violators {viol}')
    print(f"(a) constant void (agree>=0.75): {'HELD' if pa else 'FAILED'}")
    print(f"(b) specificity >=3x: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
