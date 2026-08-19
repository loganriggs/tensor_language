"""Second linked pair for the composition law: does c generalise across edges?

§35-36 established excess = c * d16 * d17 with c = 22.9 on the 16->17 edge. One edge,
one constant. The 3->4 edge (§28, marginal flip -1.34) is the other verified link with
independently damageable endpoints: partial span deletions (top-k output mean-ablation)
of layers 3 and 4 give a 3x3 grid of (d3, d4) pairs. REGISTERED PREDICTIONS:
  (a) the product law holds on this edge too: excess/(d3*d4) constant within +/-35%
      across the grid (a looser bar than 16-17's 11%, since the damages are smaller
      and noisier);
  (b) the constant is NOT universal: c(3->4) differs from c(16->17)=22.9 by more than
      2x in either direction (edge-specific coupling strength -- the 3->4 edge is
      broadband (§31) while 16->17 is one axis, so the coupling geometry differs).
Control: the same grid for the UNLINKED pair (3, 14) -- §27's graph shows no edge --
where excess should be near zero at every cell."""
import json, sys, time, torch, statistics
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
KS=(8,32,128)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_product_law_pair2_results.json')

@torch.no_grad()
def outstats(li):
    outs=[]
    def hook(mod,inp,o): outs.append(o.detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    Y=torch.cat(outs); Ybar=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
    return Ybar, orth(Vh[:max(KS)].T)

def main():
    t0=time.time()
    base=held()
    out={'pairs':{}}
    stats={li: outstats(li) for li in (3,4,14)}
    def val(patches):
        for li,k in patches:
            Ybar,V=stats[li]
            PATCH[li]=(V[:,:k],Ybar@V[:,:k])
        try: return float((held()-base).mean())
        finally:
            for li,_ in patches: PATCH.pop(li)
    for pair in ((3,4),(3,14)):
        a,b=pair
        da={k: val([(a,k)]) for k in KS}
        db={k: val([(b,k)]) for k in KS}
        cells=[]
        print(f'\npair {pair}:')
        for ka in KS:
            for kb in KS:
                j=val([(a,ka),(b,kb)])
                exc=j-da[ka]-db[kb]
                cc=exc/max(da[ka]*db[kb],1e-9)
                cells.append({'ka':ka,'kb':kb,'da':da[ka],'db':db[kb],
                              'joint':j,'excess':exc,'c':cc})
                print(f'  k=({ka:3d},{kb:3d}) d{a} {da[ka]:+.4f} d{b} {db[kb]:+.4f} '
                      f'excess {exc:+.4f}  c={cc:+.1f}',flush=True)
        out['pairs'][str(pair)]=cells
    c34=[c['c'] for c in out['pairs']['(3, 4)']]
    exc314=[abs(c['excess']) for c in out['pairs']['(3, 14)']]
    mc=statistics.mean(c34); sc=statistics.stdev(c34)
    out['c_34_mean']=mc; out['c_34_rel_sd']=sc/abs(mc) if mc else None
    pa = sc/abs(mc) < 0.35 if mc else False
    pb = abs(mc)>45.8 or abs(mc)<11.45
    out['pred_a_held']=bool(pa); out['pred_b_held']=bool(pb)
    print(f"\nc(3->4): mean {mc:+.1f}, rel sd {100*sc/abs(mc):.0f}% "
          f"-> (a) {'HELD' if pa else 'FAILED'}")
    print(f"(b) c(3->4) vs c(16->17)=22.9 differs >2x: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"unlinked control (3,14): max |excess| {max(exc314):.4f}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
