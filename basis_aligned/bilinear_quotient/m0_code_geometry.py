"""M0 CODE GEOMETRY -- 415: the m0 identity code is the model's
universal comparison substrate (411: induction band + r.3.0 both
compare m0|m0). Characterize the substrate itself, weights-first:
the fold table C[t] = m0(rms(wte(t))) over the full vocab.
Questions: how compact is the code; do the match heads read a
SHARED subspace of it; is the read subspace small.
Method: C over all 50257 tokens (weights-only, no data); effective
rank of C; per induction head, the read operator is W_q[a:b] --
the head-relevant code covariance is (C W_q^T)(C W_q^T)^T's
spectrum; overlap between heads' top-16 read subspaces = mean
squared cosine of principal angles; null = 20 random non-band
heads' pairwise overlaps.
REGISTERED PREDICTIONS:
  (a) COMPACT: effective rank of C (participation ratio of
      singular values) <= 200 of 1152;
  (b) SHARED READ: mean pairwise subspace overlap among the 9
      band heads' q-side read subspaces >= 2x the random-head
      null;
  (c) SMALL READ: per band head, 16 dims capture >= 60% of its
      read energy (median across heads)."""
import json, sys, time, torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV, orth
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m0_code_geometry_results.json'
BAND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    V=m.transformer.wte.weight.shape[0]
    C=torch.zeros(V,D)
    for i in range(0,V,4096):
        tt=torch.arange(i,min(i+4096,V),device=DEV)
        E=F.rms_norm(m.transformer.wte(tt),(D,))
        C[i:i+4096]=m.transformer.h[0].mlp(E).float().cpu()
    Cc=C-C.mean(0)
    S=torch.linalg.svdvals(Cc.to(DEV).float())
    pr=float((S**2).sum()**2/((S**4).sum()))
    print(f'code effective rank (PR of spectrum) {pr:.1f}',
          flush=True)
    def read_basis(li,hd,side='q'):
        at=m.transformer.h[li].attn
        W=(at.c_q if side=='q' else at.c_k).weight.float()
        a9,b9=hd*128,(hd+1)*128
        P=Cc.to(DEV)@W[a9:b9].T.cpu().to(DEV)      # V x 128
        U,Sv,Vh=torch.linalg.svd(P,full_matrices=False)
        en=float((Sv[:16]**2).sum()/(Sv**2).sum())
        # basis in code space: top dirs of C^T reads
        B=orth((Cc.to(DEV).T@U[:,:16]))
        return B,en
    def overlap(B1,B2):
        sv=torch.linalg.svdvals(B1.T@B2)
        return float((sv**2).mean())
    bases={}; energies=[]
    for li,hd in BAND:
        B,en=read_basis(li,hd)
        bases[f'{li}.{hd}']=B; energies.append(en)
        print(f'{li}.{hd}: 16-dim read energy {en:.3f}',flush=True)
    ks=list(bases)
    ov=[overlap(bases[a],bases[b])
        for i,a in enumerate(ks) for b in ks[i+1:]]
    band_ov=sum(ov)/len(ov)
    g=torch.Generator().manual_seed(3)
    RAND=[]
    while len(RAND)<20:
        li=int(torch.randint(0,18,(1,),generator=g))
        hd=int(torch.randint(0,9,(1,),generator=g))
        if (li,hd) not in BAND and (li,hd) not in RAND:
            RAND.append((li,hd))
    rb={f'{li}.{hd}':read_basis(li,hd)[0] for li,hd in RAND}
    rk=list(rb)
    rov=[overlap(rb[a],rb[b])
         for i,a in enumerate(rk) for b in rk[i+1:]]
    null_ov=sum(rov)/len(rov)
    med_en=sorted(energies)[len(energies)//2]
    pa=pr<=200
    pb=band_ov>=2*null_ov
    pc=med_en>=0.6
    out={'code_eff_rank':round(pr,1),
         'band_overlap':round(band_ov,4),
         'null_overlap':round(null_ov,4),
         'read_energies':{k:round(e,3)
                          for k,e in zip(ks,energies)},
         'median_read_energy':round(med_en,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"eff rank {pr:.0f} | band overlap {band_ov:.4f} vs "
          f"null {null_ov:.4f} | median 16-dim energy {med_en:.3f}")
    for nm,v in (('a','eff rank <=200'),
                 ('b','band overlap >=2x null'),
                 ('c','median 16-dim read energy >=0.6')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
