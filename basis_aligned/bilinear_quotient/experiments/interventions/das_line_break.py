"""DAS on the line-break push channel -- 352 closed value-transplants
for the PCA basis. Fork test: is the failure about basis choice
(PCA rotation mixes true variables -> a LEARNED basis fixes it) or
about the intervention class (bilinear circuits are
subtraction-defined -> NO basis works)? Learn an orthonormal 4-dim
basis B in m0-output space + a value v by gradient ascent on the
newline logit at held-in recipients under prefix patching; evaluate
on held-out recipients. Control: same optimization with a
shuffled objective (random off-slice positions) measures generic
logit-pumping by any m0 patch.
REGISTERED PREDICTIONS (explicit fork):
  (a) held-out recipients' newline-logit shift >= +0.5 -> basis
      choice was the problem; DAS unlocks value interventions;
  (b) DAS effect >= 2x the shuffled-objective control;
  (c) if (a) FAILS with training having converged (train shift
      >= +1.0), record: no 4-dim m0-output basis supports
      value-transplant at prefix scope; subtraction-defined
      deepens from PCA-specific to class-level."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; NL=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'das_line_break_results.json'
TAG='r.0.0.1'

def build_positions():
    lf=cl.leaf(TAG); mem=lf['member']
    slm=torch.zeros(54272,dtype=torch.bool); slm[lf['slice']]=True
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    tgt=cl.rows()[:,1:257].reshape(-1)
    cand=torch.nonzero(slm&~mm&(tgt!=NL)).squeeze(1)
    g1=torch.Generator().manual_seed(1)
    cand=cand[torch.randperm(len(cand),generator=g1)]
    seen=set(); rec=[]
    for gi in cand.tolist():
        if gi//256 not in seen: seen.add(gi//256); rec.append(gi)
    rec=rec[:96]
    # shuffled-objective control positions: off-slice, non-newline
    c2=torch.nonzero((~slm)&(tgt!=NL)).squeeze(1)
    g5=torch.Generator().manual_seed(5)
    c2=c2[torch.randperm(len(c2),generator=g5)]
    seen=set(); ctl=[]
    for gi in c2.tolist():
        if gi//256 not in seen: seen.add(gi//256); ctl.append(gi)
    return rec,ctl[:96]

def run(positions,label):
    ROWS=cl.rows()
    train=positions[0::2][:48]; hold=positions[1::2][:48]
    W=torch.randn(D,4,device=DEV)*0.02; W.requires_grad_(True)
    val=torch.zeros(4,device=DEV); val.requires_grad_(True)
    opt=torch.optim.Adam([W,val],lr=5e-3)
    g=torch.Generator().manual_seed(7)
    def nl_at(batch_gi,Bx,vv,grad):
        ctx=torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            tot=0
            for gi in batch_gi:
                r_,p_=gi//256,gi%256
                idx=ROWS[r_][None,:257][:,:-1].to(DEV)
                def fh(mo,i_,o_,Bx=Bx,vv=vv,p_=p_):
                    yf=o_.float()
                    co=yf@Bx
                    mask=torch.zeros(1,yf.shape[1],1,device=DEV)
                    mask[0,:p_+1]=1.0
                    yf=yf-((co-vv[None,None,:])*mask)@Bx.T
                    return yf.to(o_.dtype)
                h=cl.MODS['m0'].register_forward_hook(fh)
                x=F.rms_norm(m.transformer.wte(idx),(D,))
                x0=x; v1=None
                for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(
                    F.rms_norm(x,(D,)))/30)).float()
                h.remove()
                tot=tot+lg[0,p_,NL]
            return tot/len(batch_gi)
    base_h=float(nl_at(hold,torch.zeros(D,4,device=DEV),
                       torch.zeros(4,device=DEV),False))
    for step in range(120):
        bi=[train[int(i)] for i in
            torch.randperm(len(train),generator=g)[:6]]
        Bx=torch.linalg.qr(W)[0]
        loss=-nl_at(bi,Bx,val,True)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%30==0:
            print(f'{label} step {step}: train nl {-float(loss):+.3f}',
                  flush=True)
    with torch.no_grad(): Bx=torch.linalg.qr(W)[0].detach()
    tr=float(nl_at(train,Bx,val.detach(),False))
    base_t=float(nl_at(train,torch.zeros(D,4,device=DEV),
                       torch.zeros(4,device=DEV),False))
    ho=float(nl_at(hold,Bx,val.detach(),False))
    return {'train_shift':round(tr-base_t,3),
            'holdout_shift':round(ho-base_h,3)}

def main():
    t0=time.time()
    for p in m.parameters(): p.requires_grad_(False)
    rec,ctl=build_positions()
    das=run(rec,'DAS')
    print('DAS:',das,flush=True)
    sh=run(ctl,'CTL')
    print('CTL:',sh,flush=True)
    pa=das['holdout_shift']>=0.5
    pb=das['holdout_shift']>=2*max(sh['holdout_shift'],1e-3)
    conv=das['train_shift']>=1.0
    out={'das':das,'control':sh,'converged':bool(conv),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) holdout >=+0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=2x shuffled-objective: {'HELD' if pb else 'FAILED'}")
    if not pa and conv:
        print("(c) FORK: converged but no transfer -> "
              "subtraction-defined at class level")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
