"""Out-of-sample check of the universal compression constant: writer L9.

§79 established generic ~195-198 and trained compression 1.77-2.46x on four writers.
L9 was never used as a writer. REGISTERED PREDICTIONS: (a) trained family eff-rank in
[70, 120]; (b) shuffled in [165, 215]; (c) ratio in [1.6, 2.6]."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, FW, DEV
from tier2_model import load_elriggs
D=1152; K=48; NF=40
READERS=(10,11,12,13,15,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_vocab_writer9_results.json')

@torch.no_grad()
def family(mdl):
    def collect(li):
        outs=[]
        h=mdl.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Yw=collect(9); Ywc=Yw-Yw.mean(0)
    _,_,Vh=torch.linalg.svd(Ywc, full_matrices=False)
    Q,_=torch.linalg.qr(Vh[:K].T); V=Q[:,:K]
    rows=[]
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        Pq,_=torch.linalg.qr(Vhj[:NF].T); P=Pq[:,:NF]
        mlp=mdl.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T)
            rows.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
    X=torch.stack(rows)
    sv=torch.linalg.svdvals(X); e=sv**2
    return float(e.sum()**2/(e**2).sum())

def main():
    t0=time.time()
    mdl,cfg=load_elriggs('bilin18', device=DEV)
    er_t=family(mdl)
    g=torch.Generator(device=DEV).manual_seed(0)
    for blk in mdl.transformer.h:
        for W in (blk.mlp.Left.weight, blk.mlp.Right.weight, blk.mlp.Down.weight):
            flat=W.data.flatten()
            W.data=flat[torch.randperm(flat.numel(),device=DEV,generator=g)]\
                .view_as(W.data)
    er_s=family(mdl)
    ratio=er_s/er_t
    out={'trained':er_t,'shuffled':er_s,'compression':ratio}
    pa=70<=er_t<=120; pb=165<=er_s<=215; pc=1.6<=ratio<=2.6
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'writer L9: trained {er_t:.0f} | shuffled {er_s:.0f} | '
          f'compression {ratio:.2f}x')
    print(f"(a) trained in [70,120]: {'HELD' if pa else 'FAILED'} | "
          f"(b) shuffled in [165,215]: {'HELD' if pb else 'FAILED'} | "
          f"(c) ratio in [1.6,2.6]: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
