"""INTERCHANGE v2 -- resolve 350's null: local channel-setting did
nothing while global projection-removal moves nats. Hypothesis: the
b0 channel is read THROUGH ATTENTION from context positions (the
preceding list entries), not locally -- so a local patch misses the
causal path. Also fix mean-washout: patch with a single sampled
donor's coords, not the donor mean.
Conditions at 128 non-break recipients (newline logit shift vs base):
  local      -- set b0 coords at the recipient position only
  prefix     -- set b0 coords at recipient AND all prior positions
                in the row
  prefix_only-- set b0 coords at prior positions but NOT the
                recipient (pure context route)
  prefix_rand-- prefix patch in the random 4-dim control subspace
REGISTERED PREDICTIONS:
  (a) |prefix| >= 5x |local| (the read is contextual);
  (b) prefix_only >= 70% of prefix (the route is context, not
      local);
  (c) prefix_rand <= 40% of prefix (subspace-specific)."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; NL=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'interchange2_results.json'
TAG='r.0.0.1'

@torch.no_grad()
def main():
    t0=time.time()
    lf=cl.leaf(TAG); mem=lf['member']; msc=cl.member_scores(TAG)
    B0=cl.pca_block('m0','r.0.0',(12,16)).to(DEV)
    g=torch.Generator().manual_seed(2)
    R=torch.linalg.qr(torch.randn(D,4,generator=g))[0].T.to(DEV)
    Y=cl.capture_out('m0').float()
    donors=mem[msc>0]
    g4=torch.Generator().manual_seed(4)
    donor_one=donors[int(torch.randint(len(donors),(1,),generator=g4))]
    valB=(Y[donor_one]@B0.T.cpu()).to(DEV)
    valR=(Y[donor_one]@R.T.cpu()).to(DEV)
    slm=torch.zeros(54272,dtype=torch.bool); slm[lf['slice']]=True
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    tgt=cl.rows()[:,1:257].reshape(-1)
    cand=torch.nonzero(slm&~mm&(tgt!=NL)).squeeze(1)
    g1=torch.Generator().manual_seed(1)
    rec=cand[torch.randperm(len(cand),generator=g1)[:128]]
    ROWS=cl.rows()
    def nl_logits(positions,Bx=None,val=None,mode='local'):
        posset={}
        for gi in positions.tolist():
            posset.setdefault(gi//256,[]).append(gi%256)
        out={}; rws=sorted(posset)
        for i0 in range(0,len(rws),4):
            batch=rws[i0:i0+4]
            bb=torch.stack([ROWS[r_] for r_ in batch])[:,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            hooks=[]
            if Bx is not None:
                mask=torch.zeros(len(batch),256,dtype=torch.bool)
                for bi,r_ in enumerate(batch):
                    for p_ in posset[r_]:
                        if mode=='local': mask[bi,p_]=True
                        elif mode=='prefix': mask[bi,:p_+1]=True
                        elif mode=='prefix_only': mask[bi,:p_]=True
                mask=mask.to(DEV)
                def fh(mo,i_,o_,Bx=Bx,val=val,mask=mask):
                    yf=o_.float()
                    co=yf@Bx.T
                    yf=yf-(co-val[None,None,:]).masked_fill(
                        ~mask[:,:,None],0.0)@Bx
                    return yf.to(o_.dtype)
                hooks.append(cl.MODS['m0'].register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            for h in hooks: h.remove()
            for bi,r_ in enumerate(batch):
                for p_ in posset[r_]:
                    out[r_*256+p_]=float(lg[bi,p_,NL])
        return torch.tensor([out[int(gi)] for gi in positions.tolist()])
    # NOTE: one recipient per row for prefix modes (mask collisions):
    seen=set(); rec1=[]
    for gi in rec.tolist():
        if gi//256 not in seen: seen.add(gi//256); rec1.append(gi)
    rec1=torch.tensor(rec1[:96])
    base=nl_logits(rec1)
    res={}
    for nm9,(Bx,val,mode) in {
        'local':(B0,valB,'local'),
        'prefix':(B0,valB,'prefix'),
        'prefix_only':(B0,valB,'prefix_only'),
        'prefix_rand':(R,valR,'prefix')}.items():
        d=nl_logits(rec1,Bx,val,mode)-base
        res[nm9]={'mean':round(float(d.mean()),3),
                  'q':[round(float(d.quantile(q)),2)
                       for q in (0.1,0.5,0.9)]}
        print(f'{nm9}: {res[nm9]}',flush=True)
    L_,P_,PO_,PR_=[abs(res[k]['mean']) for k in
                   ('local','prefix','prefix_only','prefix_rand')]
    pa=P_>=5*max(L_,1e-3)
    pb=PO_>=0.7*max(P_,1e-3)
    pc=PR_<=0.4*max(P_,1e-3)
    out={'n_rec':len(rec1),'donor':int(donor_one),'conditions':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) prefix >=5x local: {'HELD' if pa else 'FAILED'}")
    print(f"(b) prefix_only >=70% prefix: {'HELD' if pb else 'FAILED'}")
    print(f"(c) rand <=40% prefix: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
