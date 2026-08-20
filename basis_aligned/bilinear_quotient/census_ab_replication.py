"""CENSUS A/B REPLICATION -- the deferred (c) leg of census_diverse
(390) under the 381 identity rule: a leaf's identity is its
MACHINERY + causal profile, and certification requires the profile
to replicate on a disjoint window. Corpus halves: rows 0-499 (A)
vs 500-999 (B) of curated_rows.pt (docs never straddle halves
except possibly one boundary doc; <=2 rows/doc). For each sampled
leaf, the damage profile is the 4-vector of per-probe member-mean
dCE, computed separately on A-members and B-members (sampled rows,
<=40 per side). Machinery is reconstructed exactly as
census_diverse built it (comp/head probes at level 0; slice-PCA
probes below, slice = leaf['slice'], conditioning corpus-wide).
REGISTERED PREDICTIONS:
  (a) >=60% of eligible sampled leaves replicate: cosine of the
      4-probe profile A-vs-B >= 0.7;
  (b) specificity null: matched median cosine exceeds
      MISMATCHED-leaf median (A of leaf i vs B of leaf perm[i],
      seeded derangement) by >= 0.15;
  (c) depth holds: replication rate at depth>=2 within 20 points
      of depth<=1 (identity rule is scale-free); if FAILED,
      identity is coarse-grain only (fork, both branches
      informative)."""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'census_ab_replication_results.json'
NLEAF_PER_DEPTH=16; MAXROWS_SIDE=40; MINMEM=16

def safe_svd(X):
    X=torch.nan_to_num(X.float()).clamp(-6e4,6e4)
    try: return torch.linalg.svd(X,full_matrices=False)
    except Exception:
        U,S,Vh=torch.linalg.svd(X.double().cpu(),full_matrices=False)
        return (U.float().to(X.device),S.float().to(X.device),
                Vh.float().to(X.device))

@torch.no_grad()
def main():
    t0=time.time()
    st=torch.load(PT+'census_state_diverse.pt',map_location='cpu',
                  weights_only=False)
    rows=st['rows']; basev=st['basev'].float()
    leaves=st['leaves']
    cd=torch.load(PT+'curated_rows.pt',map_location='cpu',
                  weights_only=False)
    docid=cd['docid']
    if int(docid[499])==int(docid[500]):
        print('NOTE: one document straddles the A/B boundary')
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb

    # --- sample leaves, stratified by depth, seeded ---
    g=torch.Generator().manual_seed(7)
    bydep={}
    for lf in leaves:
        mem=lf['member']; arow=mem//256
        na=int((arow<500).sum()); nb=int((arow>=500).sum())
        if na<MINMEM or nb<MINMEM: continue
        bydep.setdefault(lf['tag'].count('.')-1,[]).append(lf)
    sample=[]
    for dep in sorted(bydep):
        pool=bydep[dep]
        pick=torch.randperm(len(pool),generator=g)[:NLEAF_PER_DEPTH]
        sample+=[pool[i] for i in pick.tolist()]
        print(f'depth {dep}: {len(pool)} eligible, '
              f'{min(len(pool),NLEAF_PER_DEPTH)} sampled',flush=True)

    # --- one full-corpus pass: capture outputs for all pca keys,
    #     accumulate means for all 36 comps ---
    import ast
    pspecs={}   # leaf tag -> list of parsed probes
    pcakeys=set()
    for lf in sample:
        ps=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
        pspecs[lf['tag']]=ps
        for p in ps:
            if p[0]=='pca': pcakeys.add(p[1])
    print(f'{len(sample)} leaves, {len(pcakeys)} pca keys to '
          f'capture',flush=True)
    sums={k:torch.zeros(D,device=DEV) for k in MODS}
    caps={k:[] for k in pcakeys}
    hs=[]
    for key,mod in MODS.items():
        def mk(key=key):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                yf=y.detach().reshape(-1,D)
                sums[key]+=yf.float().sum(0)
                if key in caps: caps[key].append(yf.half().cpu())
            return h
        hs.append(mod.register_forward_hook(mk()))
    for i in range(0,1000,4):
        bb=rows[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(),bb[:,1:].contiguous())
    for h in hs: h.remove()
    mus={k:(v/256000).cpu() for k,v in sums.items()}
    caps={k:torch.cat(v) for k,v in caps.items()}
    print(f'capture done ({time.time()-t0:.0f}s)',flush=True)

    # --- probe constructors (ported from census_diverse) ---
    PCACHE={}
    def pca_P(key,stag,blk,slice_idx):
        kk=(key,stag,tuple(blk))
        if kk not in PCACHE:
            Y=caps[key][slice_idx].float().to(DEV)
            _,_,Vh=safe_svd((Y-Y.mean(0))[:20000])
            s0,s1=blk
            PCACHE[kk]=orth(Vh[s0:s1].T)
        return PCACHE[kk]
    def hooks_for(p,slice_idx):
        kind=p[0]
        if kind=='comp':
            key=p[1]; mu=mus[key].to(DEV); mod=MODS[key]
            if key[0]=='a':
                def fh(mo,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,mu=mu):
                    return mu.expand_as(o_).to(o_.dtype)
            return [mod.register_forward_hook(fh)]
        if kind=='head':
            li,hd=p[1],p[2]; at=m.transformer.h[li].attn
            def fh(mo_,args,out,at=at,hd=hd):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                B=X.shape[0]
                v=at.c_v(X).view(B,T,9,128)
                vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2f=F.rms_norm(at.c_q2(X).view(B,T,9,128),(128,))
                k2f=F.rms_norm(at.c_k2(X).view(B,T,9,128),(128,))
                q2f,k2f=are(q2f,cos,sin),are(k2f,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                kf.float())/128
                sc2=torch.einsum('bqhd,bkhd->bhqk',q2f.float(),
                                 k2f.float())/128
                pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
                z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                z[:,hd]=0
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X.dtype))
                return (ynew,v1r)
            return [at.register_forward_hook(fh)]
        raise RuntimeError('pca probes go through pca_hooks')
    def pca_hooks(p,slice_idx):
        _,key,stag,blk=p
        P=pca_P(key,stag,blk,slice_idx)
        mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,P=P):
                y,v1=o_
                yf=y.float().reshape(-1,D)
                return ((yf-(yf@P)@P.T).view(y.shape).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,P=P):
                yf=o_.float().reshape(-1,D)
                return (yf-(yf@P)@P.T).view(o_.shape).to(o_.dtype)
        return [mod.register_forward_hook(fh)]

    def ce_rows(rowids,hooks):
        ces={}
        for i in range(0,len(rowids),4):
            rid=rowids[i:i+4]
            bb=rows[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)) \
                .float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(len(rid),T)
            for j,r in enumerate(rid.tolist()):
                ces[r]=ce[j].cpu()
        for h in hooks: h.remove()
        return ces

    # --- per-leaf A/B profiles ---
    results=[]
    for li9,lf in enumerate(sample):
        tag=lf['tag']; mem=lf['member']; sl=lf['slice']
        arow=mem//256
        memA=mem[arow<500]; memB=mem[arow>=500]
        gg=torch.Generator().manual_seed(11+li9)
        def pickrows(mm):
            rr=(mm//256).unique()
            if len(rr)>MAXROWS_SIDE:
                rr=rr[torch.randperm(len(rr),generator=gg)
                      [:MAXROWS_SIDE]].sort().values
            return rr
        rA,rB=pickrows(memA),pickrows(memB)
        fwd_rows=torch.cat([rA,rB])
        inA=torch.isin(memA//256,rA); inB=torch.isin(memB//256,rB)
        mA,mB=memA[inA],memB[inB]
        profA=[]; profB=[]
        for p in pspecs[tag]:
            hooks=(pca_hooks(p,sl) if p[0]=='pca'
                   else hooks_for(p,sl))
            ces=ce_rows(fwd_rows,hooks)
            def mmean(mm):
                vals=[float(ces[int(gi)//256][int(gi)%256]
                            -basev[int(gi)])
                      for gi in mm]
                return sum(vals)/len(vals)
            profA.append(mmean(mA)); profB.append(mmean(mB))
        vA=torch.tensor(profA); vB=torch.tensor(profB)
        cosm=float(F.cosine_similarity(vA,vB,dim=0))
        results.append({'tag':tag,'depth':tag.count('.')-1,
                        'nA':len(mA),'nB':len(mB),
                        'profA':[round(x,4) for x in profA],
                        'profB':[round(x,4) for x in profB],
                        'cos':round(cosm,3)})
        print(f"{tag}: cos {cosm:.3f} A{[round(x,3) for x in profA]}"
              f" B{[round(x,3) for x in profB]}",flush=True)

    # --- score predictions ---
    cs=torch.tensor([r['cos'] for r in results])
    rep=float((cs>=0.7).float().mean())
    # mismatched null: seeded derangement
    n=len(results)
    gp=torch.Generator().manual_seed(3)
    while True:
        perm=torch.randperm(n,generator=gp)
        if not (perm==torch.arange(n)).any(): break
    mis=[]
    for i in range(n):
        vA=torch.tensor(results[i]['profA'])
        vB=torch.tensor(results[int(perm[i])]['profB'])
        k=min(len(vA),len(vB))   # level-0 leaves may differ in probe count
        mis.append(float(F.cosine_similarity(vA[:k],vB[:k],dim=0)))
    mis=torch.tensor(mis)
    dep_lo=[r['cos'] for r in results if r['depth']<=1]
    dep_hi=[r['cos'] for r in results if r['depth']>=2]
    rlo=(sum(c>=0.7 for c in dep_lo)/max(len(dep_lo),1))
    rhi=(sum(c>=0.7 for c in dep_hi)/max(len(dep_hi),1))
    pa=rep>=0.6
    pb=float(cs.median()-mis.median())>=0.15
    pc=abs(rlo-rhi)<=0.20
    out={'n_sampled':n,'replication_rate':round(rep,3),
         'cos_median':round(float(cs.median()),3),
         'mismatch_median':round(float(mis.median()),3),
         'rate_depth_le1':round(rlo,3),'rate_depth_ge2':round(rhi,3),
         'leaves':results,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f"replication rate {rep:.2f} | matched med "
          f"{float(cs.median()):.3f} vs mismatched "
          f"{float(mis.median()):.3f} | depth<=1 {rlo:.2f} "
          f"depth>=2 {rhi:.2f}")
    for nm,v in (('a','>=60% replicate at cos>=0.7'),
                 ('b','matched-mismatched median gap >=0.15'),
                 ('c','depth rates within 20 points')):
        print(f"({nm}) {v}: {'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
