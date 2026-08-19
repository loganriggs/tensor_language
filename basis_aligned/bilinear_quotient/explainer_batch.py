"""EXPLAINER BATCH -- the line-break-circuit treatment (344) for every
census leaf at once: reproduce each leaf's exact causal operation
(joint ablation of its four slice-conditioned output-PCA blocks),
measure per-position dCE, record sign-split aggregates + mechanically
chosen examples (3 top-|score| + 3 seed-0 random). Component output
captures and PCA bases are shared across leaves, so each leaf costs
one ~2s sweep. Only pca-probe leaves qualify (roots use whole-comp
probes; skipped and counted).
REGISTERED PREDICTIONS:
  (a) >=70% of tested leaves show mean |dCE| on members >= 3x the
      off-slice background;
  (b) >=50% show two-signed structure (minority-sign member mass,
      by count, >= 0.15) -- sign-mixedness is typical, not a quirk
      of r.0.0.1;
  (c) examples recorded mechanically for every tested leaf (no
      selection beyond the fixed rules)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'explainer_batch_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    import tiktoken
    enc=tiktoken.get_encoding('gpt2')
    st=torch.load(PT+'census_state.pt',map_location='cpu')
    rows=st['rows']; basev=st['basev'].float()
    L={lf['tag']:lf for lf in st['leaves']}
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})
    OUTCAP={}
    def capture_out(key):
        if key in OUTCAP: return OUTCAP[key]
        capsX=[]
        def cap(mo,i_,o_):
            y=o_[0] if isinstance(o_,tuple) else o_
            capsX.append(y.detach().half().reshape(-1,D).cpu())
        h=MODS[key].register_forward_hook(cap)
        for i in range(0,212,4):
            bb=rows[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        h.remove()
        OUTCAP[key]=torch.cat(capsX)
        return OUTCAP[key]
    PCA={}
    def pca_rows(key,stag,blk):
        kk=(key,stag,blk)
        if kk in PCA: return PCA[kk]
        slice_idx=L[stag]['member'] if stag in L else torch.arange(54272)
        Y=capture_out(key)[slice_idx].float().to(DEV)
        _,_,Vh=torch.linalg.svd((Y-Y.mean(0))[:20000],full_matrices=False)
        s0,s1=blk
        PCA[kk]=Vh[s0:s1]
        return PCA[kk]
    def sweep(hooks):
        ces=[]
        for i in range(0,212,4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blkm in m.transformer.h:
                x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none').cpu())
        return torch.cat(ces).float()
    res=[]; skipped=0
    for lf in st['leaves']:
        try: probes=[eval(p) for p in lf['top_probes']]
        except Exception: skipped+=1; continue
        if not all(isinstance(p,tuple) and p[0]=='pca' for p in probes):
            skipped+=1; continue
        member=lf['member']; sl=lf['slice']; sc=lf['score']
        msc=sc[sc.abs()>=sc.abs().quantile(0.85)]
        if len(msc)!=len(member) or len(member)<40:
            skipped+=1; continue
        PER={}
        for _,key,stag,blk in probes:
            PER.setdefault(key,[]).append(pca_rows(key,stag,blk))
        hooks=[]
        for key,vs in PER.items():
            P=orth(torch.cat(vs).T)
            if key[0]=='a':
                def fh(mo,i_,o_,P=P):
                    y,v1=o_
                    yf=y.float().reshape(-1,D)
                    yn=yf-(yf@P)@P.T
                    return (yn.view(y.shape).to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,P=P):
                    yf=o_.float().reshape(-1,D)
                    yn=yf-(yf@P)@P.T
                    return yn.view(o_.shape).to(o_.dtype)
            hooks.append(MODS[key].register_forward_hook(fh))
        abl=sweep(hooks)
        for h in hooks: h.remove()
        d=abl-basev
        mm=torch.zeros(54272,dtype=torch.bool); mm[member]=True
        slm=torch.zeros(54272,dtype=torch.bool); slm[sl]=True
        am=float(d[mm].abs().mean()); ag=float(d[~slm].abs().mean())
        dm=float(d[mm].mean())
        npos=int((msc>0).sum()); nneg=int((msc<0).sum())
        dpos=float(d[member[msc>0]].mean()) if npos else 0.0
        dneg=float(d[member[msc<0]].mean()) if nneg else 0.0
        minshare=min(npos,nneg)/max(len(member),1)
        order=msc.abs().argsort(descending=True)
        g=torch.Generator().manual_seed(0)
        rest=member[order[3:]]
        rnd=rest[torch.randperm(len(rest),generator=g)[:3]]
        def ex(gi):
            gi=int(gi); r_,p_=gi//256,gi%256
            toks=rows[r_].tolist()
            ctx=enc.decode(toks[max(0,p_-12):p_+1])
            return {'context':ctx[-70:],'target':enc.decode([toks[p_+1]]),
                    'base_ce':round(float(basev[gi]),2),
                    'dce':round(float(d[gi]),2)}
        res.append({'tag':lf['tag'],'n_members':len(member),
                    'probes':lf['top_probes'],
                    'abs_dce_members':round(am,3),
                    'abs_dce_offslice':round(ag,3),
                    'conc':round(am/max(ag,1e-4),2),
                    'dce_members':round(dm,3),
                    'dce_pos':round(dpos,3),'dce_neg':round(dneg,3),
                    'n_pos':npos,'n_neg':nneg,
                    'min_sign_share':round(minshare,3),
                    'base_ce_member_mean':round(float(basev[mm].mean()),3),
                    'base_ce_frac_lt3':round(float((basev[mm]<3).float().mean()),3),
                    'top_examples':[ex(g_) for g_ in member[order[:3]]],
                    'random_examples':[ex(g_) for g_ in rnd]})
        print(f"{lf['tag']}: conc {res[-1]['conc']}x | signed "
              f"{dpos:+.2f}/{dneg:+.2f} | minority {minshare:.2f}",
              flush=True)
    nt=len(res)
    pa=sum(1 for r in res if r['conc']>=3)>=0.7*nt
    pb=sum(1 for r in res if r['min_sign_share']>=0.15)>=0.5*nt
    out={'n_tested':nt,'n_skipped':skipped,
         'frac_conc3':round(sum(1 for r in res if r['conc']>=3)/max(nt,1),3),
         'frac_twosigned':round(sum(1 for r in res
                                    if r['min_sign_share']>=0.15)/max(nt,1),3),
         'leaves':res,'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f'tested {nt} skipped {skipped} | conc>=3x: '
          f"{out['frac_conc3']:.0%} | two-signed: {out['frac_twosigned']:.0%}")
    print(f"(a) >=70% conc >=3x: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=50% two-signed: {'HELD' if pb else 'FAILED'}")
    print("(c) examples recorded: HELD")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
