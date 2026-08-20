"""SOP BATCH CERTIFIED -- 403: bridge the induction-arc close
back to the census campaign. Question: does the A/B certification
gate (395) predict the SOP's own step-1 concentration gate? Take
the top-24 CERTIFIED leaves (by A/B cosine) and a seeded sample
of 24 UNCERTIFIED leaves (cos<0.7) from the diverse tree; for
each, ablate its own 4-probe machinery jointly and measure
concentration (member |dCE| / off-slice |dCE| within the same
forwarded rows), sign split, and verbatim examples; write partial
circuit packs (SOP steps 1-2) for every passing leaf.
REGISTERED PREDICTIONS:
  (a) >=70% of the certified 24 pass the SOP gate
      (concentration >= 3);
  (b) certified leaves pass at >=1.5x the uncertified rate (the
      A/B gate predicts local selectivity, not just stability);
  (c) packs with examples written for every passing leaf (file
      sop_packs_certified.json)."""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m, DEV, orth
import tiktoken
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sop_batch_certified_results.json'
PACKS=PT+'sop_packs_certified.json'
NSEL=24; MAXROWS=60

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
    enc=tiktoken.get_encoding('gpt2')
    st=torch.load(PT+'census_state_diverse.pt',map_location='cpu',
                  weights_only=False)
    rows=st['rows']; basev=st['basev'].float()
    bytag={lf['tag']:lf for lf in st['leaves']}
    ab=json.load(open(PT+'census_ab_full_results.json'))
    cert=sorted([r for r in ab['leaves'] if r['cos']>=0.7],
                key=lambda r:-r['cos'])[:NSEL]
    unc_all=[r for r in ab['leaves'] if r['cos']<0.7]
    g=torch.Generator().manual_seed(5)
    unc=[unc_all[i] for i in
         torch.randperm(len(unc_all),generator=g)[:NSEL].tolist()]
    sel=[('cert',r) for r in cert]+[('unc',r) for r in unc]
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    import ast
    pspecs={}; pcakeys=set()
    for _,r in sel:
        lf=bytag[r['tag']]
        ps=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
        pspecs[r['tag']]=ps
        for p in ps:
            if p[0]=='pca': pcakeys.add(p[1])
    print(f'{len(sel)} leaves, {len(pcakeys)} pca keys',flush=True)
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
    packs={}; results=[]
    for grp,r in sel:
        tag=r['tag']; lf=bytag[tag]
        mem=lf['member']; sl=lf['slice']
        slm=torch.zeros(256000,dtype=torch.bool); slm[sl]=True
        gg=torch.Generator().manual_seed(13)
        rr=(mem//256).unique()
        if len(rr)>MAXROWS:
            rr=rr[torch.randperm(len(rr),generator=gg)[:MAXROWS]] \
                .sort().values
        ces=ce_rows(rr,hooks_for(pspecs[tag],sl))
        dmem=[]; doff=[]
        mset=set(mem.tolist())
        for row in rr.tolist():
            dvec=(ces[row]-basev[row*256:(row+1)*256])
            for p in range(T):
                gi=row*256+p
                if gi in mset: dmem.append(float(dvec[p]))
                elif not slm[gi]: doff.append(float(dvec[p]))
        am=sum(abs(x) for x in dmem)/max(len(dmem),1)
        ao=sum(abs(x) for x in doff)/max(len(doff),1)
        conc=am/max(ao,1e-4)
        npos=sum(x>0 for x in dmem); nneg=sum(x<0 for x in dmem)
        passed=conc>=3
        rec={'tag':tag,'group':grp,'ab_cos':r['cos'],
             'concentration':round(conc,2),
             'dce_members':round(sum(dmem)/max(len(dmem),1),4),
             'n_pos':npos,'n_neg':nneg,'n_scored':len(dmem),
             'gate':'PASS' if passed else 'FAIL'}
        results.append(rec)
        if passed:
            order=sorted(range(len(dmem)),
                         key=lambda j:-abs(dmem[j]))
            memf=[gi for row in rr.tolist()
                  for gi in range(row*256,(row+1)*256)
                  if gi in mset]
            exs=[]
            gex=torch.Generator().manual_seed(3)
            picks=order[:3]+[order[3:][j] for j in
                torch.randperm(max(len(order)-3,1),
                               generator=gex)[:3].tolist()
                if order[3:]]
            for j in picks[:6]:
                gi=memf[j]; row,p=gi//256,gi%256
                toks=rows[row].tolist()
                pre=enc.decode(toks[max(0,p-12):p+1])[-70:]
                exs.append({'gi':gi,'context':pre,
                            'target':enc.decode([toks[p+1]]),
                            'dce':round(dmem[j],2)})
            packs[tag]={'causal':rec,'examples':exs,
                        'probes':[str(p) for p in pspecs[tag]],
                        'provenance':'sop_batch_certified 403',
                        'ab_cos':r['cos']}
        print(f"{tag} [{grp}] cos {r['cos']} conc {conc:.2f} "
              f"{'PASS' if passed else 'FAIL'}",flush=True)
    cr=[x for x in results if x['group']=='cert']
    ur=[x for x in results if x['group']=='unc']
    ratec=sum(x['gate']=='PASS' for x in cr)/max(len(cr),1)
    rateu=sum(x['gate']=='PASS' for x in ur)/max(len(ur),1)
    pa=ratec>=0.7
    pb=ratec>=1.5*max(rateu,1e-6)
    json.dump(packs,open(PACKS,'w'),indent=1)
    pc=len(packs)==sum(x['gate']=='PASS' for x in results)
    out={'results':results,'rate_cert':round(ratec,3),
         'rate_unc':round(rateu,3),'n_packs':len(packs),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"cert pass {ratec:.2f} | unc pass {rateu:.2f} | "
          f"packs {len(packs)}")
    for nm,v in (('a','cert pass >=70%'),
                 ('b','cert >=1.5x unc'),
                 ('c','packs for all passing')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
