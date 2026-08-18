"""OV MOTIFS -- the other half of the head: OV_h = Proj_h @ V_h (rank
<=128 linear map, weights-only). Two questions.
(1) SHARED REPERTOIRE: do the 162 heads' OV read subspaces (row space of
    the c_v slice) and write subspaces (column space of the c_proj
    slice) draw from a shared low-dim library? Leave-one-head-out: how
    much of head h's read energy lies in the top-r principal basis of
    the OTHER 161 heads' read spaces? Control: random r-dim basis.
    Random-overlap floor for two 128-dim subspaces in 1152-dim = 0.111.
(2) BINDINGS: within a pattern-motif family (prev/self/ind), do heads
    share OV structure more than across families? The dialect results
    (219-221) predict NO -- shared function, private bindings.
REGISTERED PREDICTIONS:
  (a) LOO r=256 read-energy >= 0.5 mean and >= 2x the random-basis
      control (a shared read library exists);
  (b) same for write side;
  (c) PRIVATE-BINDINGS: within-family mean pairwise subspace overlap
      exceeds across-family by < 1.15x for >= 2 of 3 families (motifs
      share pattern function, not OV bindings);
  (d) all pairwise-overlap distributions reported vs the 0.111 floor."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m, DEV
D=1152; HD=128
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ov_motifs_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    fam={}
    for li,hd,mo,fr in mt: fam[(li,hd)]=mo
    reads={}; writes={}
    for li in range(18):
        at=m.transformer.h[li].attn
        Wv=at.c_v.weight.detach().float()        # (D,D): rows -> hidden
        Wp=at.c_proj.weight.detach().float()     # (D,D)
        for hd in range(9):
            reads[(li,hd)]=Wv[hd*HD:(hd+1)*HD,:]          # (128,D)
            writes[(li,hd)]=Wp[:,hd*HD:(hd+1)*HD]         # (D,128)
    keys=list(reads.keys())
    def orthb(Mm):
        Q,_=torch.linalg.qr(Mm.T if Mm.shape[0]<Mm.shape[1] else Mm)
        return Q[:,:HD]
    QR={k:orthb(reads[k]) for k in keys}    # (D,128) orthonormal
    QW={k:orthb(writes[k]) for k in keys}
    g=torch.Generator(device=DEV).manual_seed(0)
    def loo_energy(mats,Qb):
        r=256
        es=[]; ectl=[]
        for k in keys:
            OTH=torch.cat([Qb[k2] for k2 in keys if k2!=k],1)  # (D,161*128)
            # top-r basis of the others via covariance eig
            C=OTH@OTH.T
            ev,evec=torch.linalg.eigh(C)
            U=evec[:,-r:]
            M=mats[k]
            Mn=(M@U).pow(2).sum()/M.pow(2).sum()
            es.append(float(Mn))
            R=torch.randn(D,r,device=DEV,generator=g)
            R,_=torch.linalg.qr(R)
            ectl.append(float((M@R).pow(2).sum()/M.pow(2).sum()))
        return sum(es)/len(es),sum(ectl)/len(ectl)
    # subsample LOO for speed: every 3rd head for the LOO stat
    sub=keys[::3]
    def loo_sub(mats,Qb):
        r=256; es=[]; ectl=[]
        ALL=torch.cat([Qb[k2] for k2 in keys],1)
        C=ALL@ALL.T
        for k in sub:
            Ck=C-Qb[k]@Qb[k].T
            ev,evec=torch.linalg.eigh(Ck)
            U=evec[:,-r:]
            M=mats[k]
            es.append(float((M@U).pow(2).sum()/M.pow(2).sum()))
            R=torch.randn(D,r,device=DEV,generator=g)
            R,_=torch.linalg.qr(R)
            ectl.append(float((M@R).pow(2).sum()/M.pow(2).sum()))
        return sum(es)/len(es),sum(ectl)/len(ectl)
    er,cr=loo_sub(reads,QR)
    ew,cw=loo_sub(writes,{k:QW[k] for k in keys})
    print(f'LOO r=256 read energy {er:.3f} (random {cr:.3f}) | '
          f'write {ew:.3f} (random {cw:.3f})',flush=True)
    # pairwise subspace overlaps (read side)
    N=len(keys)
    OV=torch.zeros(N,N)
    for i,k1 in enumerate(keys):
        for j in range(i+1,N):
            o=float((QR[k1].T@QR[keys[j]]).pow(2).sum())/HD
            OV[i,j]=o; OV[j,i]=o
    labs=[fam[k] for k in keys]
    def famstat(name):
        idx=[i for i,l in enumerate(labs) if l==name]
        if len(idx)<3: return None
        within=[]; across=[]
        for a in range(len(idx)):
            for b in range(a+1,len(idx)):
                within.append(float(OV[idx[a],idx[b]]))
        oth=[i for i in range(N) if i not in idx]
        for a in idx:
            for b in oth: across.append(float(OV[a,b]))
        return (sum(within)/len(within),sum(across)/len(across))
    ratios={}
    for name in ('prev','self','ind'):
        st=famstat(name)
        if st:
            ratios[name]=round(st[0]/max(st[1],1e-6),3)
            print(f'{name}: within {st[0]:.3f} across {st[1]:.3f} '
                  f'ratio {ratios[name]}',flush=True)
    meanov=float(OV[OV>0].mean())
    print(f'mean pairwise read overlap {meanov:.3f} (random floor 0.111)')
    pa=er>=0.5 and er>=2*cr
    pb=ew>=0.5 and ew>=2*cw
    pc=sum(1 for v in ratios.values() if v<1.15)>=2
    out={'loo_read':round(er,3),'loo_read_ctl':round(cr,3),
         'loo_write':round(ew,3),'loo_write_ctl':round(cw,3),
         'mean_pairwise_read_overlap':round(meanov,3),
         'family_ratios':ratios,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) shared read library: {'HELD' if pa else 'FAILED'}")
    print(f"(b) shared write library: {'HELD' if pb else 'FAILED'}")
    print(f"(c) private bindings (>=2/3 ratios < 1.15): "
          f"{'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
