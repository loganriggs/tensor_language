"""GATE FULL -- 405: the both-halves selectivity gate works
(94% SOP agreement, 79% fresh-leaf pass, shuffled null ~6%). Run
it over ALL 311 diverse-tree leaves to produce the swarm's real
production shortlist (replaces the voided 395 raw-cosine list).
REGISTERED PREDICTIONS:
  (a) full-tree pass rate within 15 points of the sampled 79%;
  (b) >=180 leaves pass (production pool);
  (c) DEPTH fork: pass rate at depth>=2 within 20 points of
      depth<=1 (selectivity, unlike raw-damage stability, is
      depth-uniform) -- report either way;
  (d) shuffled null <=8% (calibrated to the observed 6%)."""

import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'gate_full_results.json'
MAXROWS_SIDE=30; NEXTRA=48

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
    bytag={lf['tag']:lf for lf in st['leaves']}
    prev={}
    tags=[lf['tag'] for lf in st['leaves']]
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    import ast
    pspecs={}; pcakeys=set()
    for tg9 in tags:
        ps=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in bytag[tg9]['top_probes']]
        pspecs[tg9]=ps
        for p in ps:
            if p[0]=='pca': pcakeys.add(p[1])
    print(f'{len(tags)} leaves, {len(pcakeys)} pca keys',
          flush=True)
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
    PCACHE={}
    def pca_P(key,stag,blk,slice_idx):
        kk=(key,stag,tuple(blk))
        if kk not in PCACHE:
            Y=caps[key][slice_idx].float().to(DEV)
            _,_,Vh=safe_svd((Y-Y.mean(0))[:20000])
            s0,s1=blk
            PCACHE[kk]=orth(Vh[s0:s1].T)
        return PCACHE[kk]
    def hooks_for(ps,sl):
        out=[]
        for p in ps:
            if p[0]=='comp':
                key=p[1]; mu=mus[key].to(DEV); mod=MODS[key]
                if key[0]=='a':
                    def fh(mo,i_,o_,mu=mu):
                        y,v1=o_
                        return (mu.expand_as(y).to(y.dtype),v1)
                else:
                    def fh(mo,i_,o_,mu=mu):
                        return mu.expand_as(o_).to(o_.dtype)
                out.append(MODS[key].register_forward_hook(fh))
            elif p[0]=='head':
                li,hd=p[1],p[2]; at=m.transformer.h[li].attn
                def fh(mo_,args,o_,at=at,hd=hd):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    B=X.shape[0]
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                    kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                    qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                    q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),
                                  (128,))
                    k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),
                                  (128,))
                    q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',
                                     q2.float(),k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    z[:,hd]=0
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                out.append(at.register_forward_hook(fh))
            else:
                _,key,stag,blk=p
                P=pca_P(key,stag,blk,sl)
                if key[0]=='a':
                    def fh(mo,i_,o_,P=P):
                        y,v1=o_
                        yf=y.float().reshape(-1,D)
                        return ((yf-(yf@P)@P.T).view(y.shape)
                                .to(y.dtype),v1)
                else:
                    def fh(mo,i_,o_,P=P):
                        yf=o_.float().reshape(-1,D)
                        return (yf-(yf@P)@P.T).view(o_.shape) \
                            .to(o_.dtype)
                out.append(MODS[key].register_forward_hook(fh))
        return out
    def ce_rows(rowids,hooks):
        ces={}
        for i in range(0,len(rowids),4):
            rid=rowids[i:i+4]
            bb=rows[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(len(rid),T)
            for j,r in enumerate(rid.tolist()):
                ces[r]=ce[j].cpu()
        for h in hooks: h.remove()
        return ces
    results=[]
    for li9,tg9 in enumerate(tags):
        lf=bytag[tg9]; mem=lf['member']; sl=lf['slice']
        slm=torch.zeros(256000,dtype=torch.bool); slm[sl]=True
        mset=set(mem.tolist())
        gg=torch.Generator().manual_seed(19+li9)
        def pickrows(mm):
            rr=(mm//256).unique()
            if len(rr)>MAXROWS_SIDE:
                rr=rr[torch.randperm(len(rr),generator=gg)
                      [:MAXROWS_SIDE]].sort().values
            return rr
        arow=mem//256
        rA=pickrows(mem[arow<500]); rB=pickrows(mem[arow>=500])
        if len(rA)==0 or len(rB)==0: continue
        fwd_rows=torch.cat([rA,rB])
        ces=ce_rows(fwd_rows,hooks_for(pspecs[tg9],sl))
        def conc_of(rset,shufn=False):
            dm=[]; do=[]
            gsh=torch.Generator().manual_seed(29)
            for row in rset.tolist():
                dvec=ces[row]-basev[row*256:(row+1)*256]
                labels=[]
                for p in range(T):
                    gi=row*256+p
                    labels.append(1 if gi in mset else
                                  (0 if not slm[gi] else -1))
                if shufn:
                    lt=torch.tensor(labels)
                    perm=torch.randperm(T,generator=gsh)
                    lt=lt[perm]; labels=lt.tolist()
                for p in range(T):
                    if labels[p]==1: dm.append(float(dvec[p]))
                    elif labels[p]==0: do.append(float(dvec[p]))
            am=sum(abs(x) for x in dm)/max(len(dm),1)
            ao=sum(abs(x) for x in do)/max(len(do),1)
            return am/max(ao,1e-4)
        cA=conc_of(rA); cB=conc_of(rB)
        cAs=conc_of(rA,shufn=True)
        newgate=cA>=3 and cB>=3
        rec={'tag':tg9,'in_403':tg9 in prev,
             'sop_gate':prev.get(tg9,{}).get('gate'),
             'conc_A':round(cA,2),'conc_B':round(cB,2),
             'conc_A_shuf':round(cAs,2),
             'new_gate':'PASS' if newgate else 'FAIL'}
        results.append(rec)
        print(f"{tg9}: A {cA:.2f} B {cB:.2f} shuf {cAs:.2f} "
              f"{'PASS' if newgate else 'FAIL'}"
             +(f" (sop {rec['sop_gate']})" if rec['sop_gate']
               else ''),flush=True)
    npass=sum(r['new_gate']=='PASS' for r in results)
    rate=npass/max(len(results),1)
    dep=lambda t9:t9.count('.')-1
    lo=[r for r in results if dep(r['tag'])<=1]
    hi=[r for r in results if dep(r['tag'])>=2]
    rlo=sum(r['new_gate']=='PASS' for r in lo)/max(len(lo),1)
    rhi=sum(r['new_gate']=='PASS' for r in hi)/max(len(hi),1)
    shufpass=sum(r['conc_A_shuf']>=3 for r in results) \
        /max(len(results),1)
    pa=abs(rate-0.79)<=0.15; pb=npass>=180
    pc=abs(rlo-rhi)<=0.20; pd=shufpass<=0.08
    passtags=[r['tag'] for r in results if r['new_gate']=='PASS']
    json.dump(passtags,open(PT+'swarm_shortlist.json','w'))
    out={'results':results,'pass_rate':round(rate,3),
         'n_pass':npass,'rate_depth_le1':round(rlo,3),
         'rate_depth_ge2':round(rhi,3),
         'shuf_pass_rate':round(shufpass,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd)}
    print(f"pass {rate:.2f} ({npass}) | depth<=1 {rlo:.2f} "
          f"depth>=2 {rhi:.2f} | shuf {shufpass:.2f}")
    for nm,v in (('a','rate within 15pts of 0.79'),
                 ('b','>=180 pass'),
                 ('c','depth rates within 20 points'),
                 ('d','shuffled null <=8%')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
