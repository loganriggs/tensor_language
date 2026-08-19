"""CIRCUIT EXPLAINER -- standalone honest demo of r.0.0.1, the
strongest induction-grade circuit ("second piece of rare multi-token
words", mb3 lift 35.5x). Loads census_state.pt, reproduces the exact
census causal operation (project out the four slice-conditioned
output-PCA blocks of mlp0/mlp3 that define the neighborhood), and
reports per-position CE damage for 5 illustrative (top-|score|) and
5 RANDOM (seed 0) member positions, plus aggregate selectivity.
Examples are drawn mechanically -- no cherry-picking.
REGISTERED EXPECTATIONS:
  (a) member mean dCE >= 4x non-member slice mean dCE (selectivity
      replicates under the joint-4-block ablation);
  (b) the 5 random members show the same sign of effect as the 5
      illustrative ones (>=4/5 positive dCE)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_explainer2_results.json'
TAG='r.0.0.1'

@torch.no_grad()
def main():
    t0=time.time()
    import tiktoken
    enc=tiktoken.get_encoding('gpt2')
    st=torch.load(PT+'census_state.pt',map_location='cpu')
    rows=st['rows']; basev=st['basev'].float()
    L={lf['tag']:lf for lf in st['leaves']}
    lf=L[TAG]
    probes=[eval(p) for p in lf['top_probes']]
    print(f'{TAG}: n={lf["n_members"]} probes {probes}',flush=True)
    member=lf['member']; sl=lf['slice']; sc=lf['score']
    msc=sc[sc.abs()>=sc.abs().quantile(0.85)]
    assert len(msc)==len(member),(len(msc),len(member))
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
    # build the joint projection-removal per component, exactly as census
    PER={}
    for _,key,stag,blk in probes:
        slice_idx=L[stag]['member'] if stag in L else torch.arange(54272)
        Y=capture_out(key)[slice_idx].float().to(DEV)
        _,_,Vh=torch.linalg.svd((Y-Y.mean(0))[:20000],full_matrices=False)
        s0,s1=blk
        PER.setdefault(key,[]).append(Vh[s0:s1])
        print(f'PCA {key} {stag} {blk} done',flush=True)
    hooks=[]
    for key,vs in PER.items():
        P=orth(torch.cat(vs).T)
        def fh(mo,i_,o_,P=P):
            yf=o_.float().reshape(-1,D)
            yn=yf-(yf@P)@P.T
            return yn.view(o_.shape).to(o_.dtype)
        hooks.append(MODS[key].register_forward_hook(fh))
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
    for h in hooks: h.remove()
    abl=torch.cat(ces).float()
    d=abl-basev
    memb_mask=torch.zeros(54272,dtype=torch.bool); memb_mask[member]=True
    sl_mask=torch.zeros(54272,dtype=torch.bool); sl_mask[sl]=True
    dm=float(d[memb_mask].mean()); dn=float(d[sl_mask&~memb_mask].mean())
    dg=float(d[~sl_mask].mean())
    am=float(d[memb_mask].abs().mean())
    an=float(d[sl_mask&~memb_mask].abs().mean())
    ag=float(d[~sl_mask].abs().mean())
    print(f'|dCE| members {am:.3f} | slice non-members {an:.3f} | '
          f'off-slice {ag:.3f}',flush=True)
    print(f'dCE members {dm:+.4f} | slice non-members {dn:+.4f} | '
          f'off-slice {dg:+.4f}',flush=True)
    mb=json.load(open(PT+'mechanism_bootstrap3_results.json'))
    rec=[e for e in mb['results'] if e['tag']==TAG][0]
    trig=set(rec['trigger_top'])
    mb5=json.load(open(PT+'mechanism_bootstrap5_results.json'))
    trig5=set()
    for e in mb5['results']:
        if e.get('pass'): trig5|=set(e['trigger_top'])
    # suffix-style match: pair 'X|\n' where X ends with a place/venue suffix
    SUF=('ton','bourne','ington','bury',' Park',' House',' Hotel',
         ' Abbey',' Court',' Palace',' Hall',' Tower',' Bar')
    dpos=float(d[member[msc>0]].mean()) if (msc>0).any() else float('nan')
    dneg=float(d[member[msc<0]].mean()) if (msc<0).any() else float('nan')
    npos_=int((msc>0).sum()); nneg_=int((msc<0).sum())
    print(f'sign split: {npos_} pos-score dCE {dpos:+.3f} | '
          f'{nneg_} neg-score dCE {dneg:+.3f}',flush=True)
    order=msc.abs().argsort(descending=True)
    ill=member[order[:5]]
    rest=member[order[5:]]
    g=torch.Generator().manual_seed(0)
    rnd=rest[torch.randperm(len(rest),generator=g)[:5]]
    def ex(gi):
        gi=int(gi); r_,p_=gi//256,gi%256
        toks=rows[r_].tolist()
        lo=max(0,p_-14)
        ctx=enc.decode(toks[lo:p_+1])
        tgt=enc.decode([toks[p_+1]])
        pair=enc.decode([toks[p_]])+'|'+tgt
        sgn=float(sc[(lf['slice']==gi).nonzero()[0,0]]) if (lf['slice']==gi).any() else 0.0
        sufm=tgt=='\n' and any(enc.decode([toks[p_]]).endswith(s9) for s9 in SUF)
        return {'row':r_,'pos':p_,'context':ctx[-90:],'target':tgt,
                'pair':pair,'score':round(sgn,3),
                'in_mb5_triggers':pair in trig5,'suffix_newline':bool(sufm),
                'in_top8_triggers':pair in trig,
                'base_ce':round(float(basev[gi]),3),
                'ablated_ce':round(float(abl[gi]),3),
                'dce':round(float(d[gi]),3)}
    ILL=[ex(g_) for g_ in ill]; RND=[ex(g_) for g_ in rnd]
    for e in ILL: print('ILL',e,flush=True)
    for e in RND: print('RND',e,flush=True)
    pa=dm>=4*max(dn,1e-4)
    pb=sum(1 for e in RND if e['dce']>0)>=4
    out={'tag':TAG,'probes':lf['top_probes'],'n_members':lf['n_members'],
         'mb3':{'lift':rec['lift'],'precision':rec['precision'],
                'recall':rec['recall'],'trigger_top':rec['trigger_top']},
         'dce_members':round(dm,4),'dce_slice_nonmembers':round(dn,4),
         'dce_offslice':round(dg,4),
         'abs_dce_members':round(am,4),'abs_dce_slice_nonmembers':round(an,4),
         'abs_dce_offslice':round(ag,4),
         'frac_members_hit':round(float((d[memb_mask]>0.1).float()
                                        .mean()),3),
         'illustrative':ILL,'random':RND,
         'dce_pos_score':round(dpos,4),'dce_neg_score':round(dneg,4),
         'n_pos':npos_,'n_neg':nneg_,
         'member_dce':[round(float(x),3) for x in d[member]],
         'member_score':[round(float(x),3) for x in msc],
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) member dCE >= 4x slice non-member: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random members >=4/5 positive: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
