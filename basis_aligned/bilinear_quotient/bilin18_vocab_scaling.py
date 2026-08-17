"""v2, unit-normalised (v1 stacked raw matrices; large-norm forms dominated and
absolute eff-ranks were incomparable to section 58). Same registered predictions.
Does vocabulary dimension track writer complexity? (The gradient-coupling
hypothesis's scaling prediction.)

If readers converge onto a shared code because their gradients couple through the
writer, the code's dimension should track the writer's used output complexity. Build
the same 6-reader functional family with FOUR different writers -- layers 0, 1, 3, 16
-- (readers = the six next MLPs downstream of each writer), measure family eff-rank.
Writer output complexity on record (dims for 90% of variance): L0 430, L1 241,
L3 509, L16 15. REGISTERED PREDICTIONS:
  (a) Spearman(writer dims-90, vocab eff-rank) >= 0.8 across the four;
  (b) L16-writer vocabulary eff-rank <= 30 (a simple writer needs a small code)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_vocab_scaling_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

@torch.no_grad()
def main():
    t0=time.time()
    cases={0:(1,2,3,5,9,17),1:(2,3,5,9,13,17),3:(4,5,7,9,13,17),
           16:(17,)}
    dims90={0:430,1:241,3:509,16:15}
    def collect(li):
        outs=[]
        h=m.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    out={'writers':{}}
    ranks=[]; d90=[]
    for wl,readers in cases.items():
        Yw=collect(wl); Ywc=Yw-Yw.mean(0)
        _,_,Vh=torch.linalg.svd(Ywc, full_matrices=False)
        V=orth(Vh[:K].T)
        rows=[]
        nf = NF if len(readers)>1 else 240   # L16 has one reader; use more forms
        for j in readers:
            Yj=collect(j)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
            P=orth(Vhj[:min(nf,Vhj.shape[0])].T)
            mlp=m.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            for f in range(P.shape[1]):
                M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                Ms=0.5*(M+M.T)
                rows.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
        X=torch.stack(rows)
        sv=torch.linalg.svdvals(X); e=sv**2
        er=float(e.sum()**2/(e**2).sum())
        out['writers'][wl]={'n_functionals':X.shape[0],'effrank':er,
                            'dims90':dims90[wl]}
        ranks.append(er); d90.append(dims90[wl])
        print(f'writer L{wl:2d}: {X.shape[0]} functionals, family eff-rank '
              f'{er:.0f} (writer dims-90: {dims90[wl]})',flush=True)
    rr=spearman(torch.tensor(d90),torch.tensor(ranks))
    pa=rr>=0.8; pb=out['writers'][16]['effrank']<=30
    out['spearman']=rr; out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f'\nSpearman(dims-90, vocab eff-rank) = {rr:+.2f} -> (a) '
          f"{'HELD' if pa else 'FAILED'}")
    print(f"L16 vocab rank {out['writers'][16]['effrank']:.0f} <= 30: "
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
