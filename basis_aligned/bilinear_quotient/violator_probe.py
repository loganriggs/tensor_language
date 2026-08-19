"""VIOLATOR PROBE -- r.6.0.0 was the one leaf breaking the
projection law (362: delete 1.64 vs mean-assign 0.67, agreement
0.32): somewhere in its machinery, the MEAN itself carries function.
Per-bundle sweep: for each of its 4 probe bundles alone, member-mean
|dCE| under delete-to-zero vs assign-slice-mean.
REGISTERED PREDICTIONS:
  (a) >=1 bundle is MEAN-CARRIED: deleting costs >=3x assigning the
      mean (the bundle works through its average value, a bias-like
      channel -- same object as the b5 junction's mean-carried
      regulator, 295-302);
  (b) >=1 bundle is variance-carried (delete ~= mean-assign within
      25%) -- the leaf mixes channel types, explaining the aggregate
      violation;
  (c) per-bundle table recorded in the circuit record."""
import json, time, torch
import census_lib as cl
from bilin18_joint_removal import orth, DEV
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'violator_probe_results.json'
TAG='r.6.0.0'

@torch.no_grad()
def main():
    t0=time.time()
    lf=cl.leaf(TAG)
    probes=[eval(p) for p in lf['top_probes']]
    mem=lf['member']; sl=lf['slice']; bv=cl.base_ce()
    res=[]
    for pi,(_,key,stag,blk) in enumerate(probes):
        P=orth(cl.pca_block(key,stag,blk).T.contiguous())
        Y=cl.capture_out(key).float()
        muv=(Y[sl]@P.cpu()).mean(0).to(DEV)
        row={'bundle':str(probes[pi])}
        for mode,val in (('delete',torch.zeros(P.shape[1],
                                               device=DEV)),
                         ('mean',muv)):
            def fh(mo,i_,o_,P=P,val=val,isa=(key[0]=='a')):
                if isa:
                    y,v1=o_
                    yf=y.float().reshape(-1,y.shape[-1])
                    yn=yf-((yf@P)-val[None,:])@P.T
                    return (yn.view(y.shape).to(y.dtype),v1)
                yf=o_.float().reshape(-1,o_.shape[-1])
                yn=yf-((yf@P)-val[None,:])@P.T
                return yn.view(o_.shape).to(o_.dtype)
            h=cl.MODS[key].register_forward_hook(fh)
            d=cl.ce_sweep([h])-bv
            row[mode]=round(float(d[mem].abs().mean()),3)
        row['ratio']=round(row['delete']/max(row['mean'],1e-3),2)
        res.append(row); print(row,flush=True)
    pa=any(r['ratio']>=3 for r in res)
    pb=any(abs(r['ratio']-1)<=0.25 for r in res)
    out={'tag':TAG,'bundles':res,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':True}
    print(f"(a) >=1 mean-carried (ratio>=3): {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=1 variance-carried: {'HELD' if pb else 'FAILED'}")
    cl.write_circuit(TAG,{'causal_bundles':{'mean_vs_delete':res},
        'certification':[{'test':'mean-carried channel present',
                          'value':max(r['ratio'] for r in res),
                          'verdict':'HELD' if pa else 'FAILED',
                          'source':'violator_probe',
                          'date':'2026-08-19'}]})
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
