"""v3 of the edge story: if tail writes are unaimed (section 92, alignment ~1.0),
the monotone decline of relative edge strength (section 91: 0.36 -> 0.04) should be
explained by DILUTION -- layer i's write shrinking relative to the residual stream
entering layer i+1, so the same transplant matters less because it is a smaller
fraction of what the reader sees.

REGISTERED PREDICTIONS: (a3) Spearman(dilution ratio r_i = E||mo_i - mean||^2 /
E||x_in(i+1) - mean||^2, measured relative effect) >= 0.7 over the ten edges;
(b3) r declines monotonically in depth (at most one inversion among the nine
adjacent comparisons). Effects reused from bilin18_blind_edges_results.json
(same rows, same measurement)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
EDGES=[(i,i+1) for i in range(5,15)]
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_edge_dilution_results.json')

def spearman(a,b):
    a=torch.tensor(a); b=torch.tensor(b)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    prev=json.load(open('/workspace/tensor_language/basis_aligned/'
                        'bilinear_quotient/bilin18_blind_edges_results.json'))
    effects=prev['effects']
    mos={}; ins={}
    hs=[]
    for li in range(5,16):
        def mko(li=li):
            return lambda mod,i_,o_: mos.setdefault(li,[]).append(
                o_.detach().reshape(-1,D).float())
        def mki(li=li):
            return lambda mod,inp: ins.setdefault(li,[]).append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mko()))
        hs.append(m.transformer.h[li].mlp.register_forward_pre_hook(mki()))
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    rs=[]
    for i,j in EDGES:
        Y=torch.cat(mos[i]); X=torch.cat(ins[j])
        wr=float((Y-Y.mean(0)).pow(2).sum(1).mean())
        res=float((X-X.mean(0)).pow(2).sum(1).mean())
        rs.append(wr/res)
        print(f'edge {i}->{j}: dilution ratio {wr/res:.4f} | effect {effects[len(rs)-1]:.4f}',
              flush=True)
    sD=spearman(rs,effects)
    inv=sum(1 for k in range(9) if rs[k+1]>rs[k])
    pa=sD>=0.7; pb=inv<=1
    out={'ratios':rs,'effects':effects,'spearman':sD,'inversions':inv,
         'pred_a3':bool(pa),'pred_b3_monotone':bool(pb)}
    print(f'\nSpearman(dilution, effect): {sD:+.2f} | inversions {inv}/9')
    print(f"(a3) dilution explains (>=0.7): {'HELD' if pa else 'FAILED'}")
    print(f"(b3) ratio ~monotone (<=1 inversion): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
