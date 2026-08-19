"""Is the ~2x functional compression universal? The denominator check.

§78: trained family eff-ranks are ~80-110 for every writer tested; the shuffled-
weights value is 191 -- but that was measured only for the L1 writer. REGISTERED
PREDICTIONS: (a) shuffled-weights family eff-ranks for writers L0, L3, L16 all land
in [165, 215] (the generic value is writer-independent too); (b) the trained/shuffled
compression ratio lies in [1.6, 2.6] for all four writers (a universal constant of
training, not an L1 fact)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, FW, DEV
from tier2_model import load_elriggs
D=1152; K=48; NF=40
CASES={0:(1,2,3,5,9,17),1:(2,3,5,9,13,17),3:(4,5,7,9,13,17),16:(17,)}
TRAINED={0:85,1:79,3:98,16:112}
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_universal_compression_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    mdl,cfg=load_elriggs('bilin18', device=DEV)
    g=torch.Generator(device=DEV).manual_seed(0)
    for blk in mdl.transformer.h:
        for W in (blk.mlp.Left.weight, blk.mlp.Right.weight, blk.mlp.Down.weight):
            flat=W.data.flatten()
            W.data=flat[torch.randperm(flat.numel(),device=DEV,generator=g)]\
                .view_as(W.data)
    mdl.eval()
    def collect(li):
        outs=[]
        h=mdl.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    out={'writers':{}}
    ok_a=True; ok_b=True
    for wl,readers in CASES.items():
        Yw=collect(wl); Ywc=Yw-Yw.mean(0)
        _,_,Vh=torch.linalg.svd(Ywc, full_matrices=False)
        Q,_=torch.linalg.qr(Vh[:K].T); V=Q[:,:K]
        rows=[]
        nf=NF if len(readers)>1 else 240
        for j in readers:
            Yj=collect(j)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
            Pq,_=torch.linalg.qr(Vhj[:min(nf,Vhj.shape[0])].T)
            P=Pq[:,:min(nf,Vhj.shape[0])]
            mlp=mdl.transformer.h[j].mlp
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
        ratio=er/TRAINED[wl]
        out['writers'][wl]={'shuffled_effrank':er,'trained':TRAINED[wl],
                            'compression':ratio}
        if not (165<=er<=215): ok_a=False
        if not (1.6<=ratio<=2.6): ok_b=False
        print(f'writer L{wl:2d}: shuffled eff-rank {er:.0f} | trained '
              f'{TRAINED[wl]} | compression {ratio:.2f}x',flush=True)
    out['pred_a']=bool(ok_a); out['pred_b']=bool(ok_b)
    print(f"\n(a) generic value writer-independent [165,215]: "
          f"{'HELD' if ok_a else 'FAILED'}")
    print(f"(b) compression universal [1.6,2.6]: {'HELD' if ok_b else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
