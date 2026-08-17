"""How far did training move the vocabulary subspace from the generic one?

§75: training compresses the functional family from eff-rank 191 to 80 and makes it
shared. Is the trained 80-dim subspace a subset of the generic (shuffled-weights)
structure, or somewhere new? Measure: energy of the trained top-80 principal basis
inside the shuffled model's top-80 span, vs (i) the shuffled top-191 span, (ii)
random 80-dim spans of matrix space. REGISTERED PREDICTIONS:
  (a) trained-in-shuffled-80 energy is 2-8x the random baseline (some inheritance)
      but < 0.5 absolute (training moved substantially);
  (b) trained-in-shuffled-191 < 0.7 (even the full generic structure does not
      contain the learned vocabulary)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import load_elriggs
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_vocab_vs_generic_results.json')

@torch.no_grad()
def family(model):
    def collect(li):
        outs=[]
        h=model.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            model(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    Q,_=torch.linalg.qr(Vh[:K].T); V=Q[:,:K]
    rows=[]
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        Pq,_=torch.linalg.qr(Vhj[:NF].T); P=Pq[:,:NF]
        mlp=model.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            rows.append((0.5*(M+M.T)).flatten())
    return torch.stack(rows)

def main():
    t0=time.time()
    mt,cfg=load_elriggs('bilin18', device=DEV)
    Xt=family(mt)
    _,_,Wt=torch.linalg.svd(Xt, full_matrices=False)
    Bt=Wt[:80]
    g=torch.Generator(device=DEV).manual_seed(0)
    for blk in mt.transformer.h:
        for W in (blk.mlp.Left.weight, blk.mlp.Right.weight, blk.mlp.Down.weight):
            flat=W.data.flatten()
            W.data=flat[torch.randperm(flat.numel(),device=DEV,generator=g)]\
                .view_as(W.data)
    Xs=family(mt)
    _,_,Ws=torch.linalg.svd(Xs, full_matrices=False)
    def energy(B, S):
        return float(((B@S.T)**2).sum())/B.shape[0]
    e80=energy(Bt,Ws[:80]); e191=energy(Bt,Ws[:191])
    dim=Bt.shape[1]
    rand80=80/dim
    out={'trained_in_shuffled80':e80,'trained_in_shuffled191':e191,
         'random_baseline_80':rand80}
    pa=(2*rand80<=e80<=0.5)
    pb=e191<0.7
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f'trained top-80 energy in shuffled top-80: {e80:.3f} '
          f'(random {rand80:.3f})')
    print(f'trained top-80 energy in shuffled top-191: {e191:.3f}')
    print(f"\n(a) inherited-but-moved: {'HELD' if pa else 'FAILED'} | "
          f"(b) not contained in generic: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
