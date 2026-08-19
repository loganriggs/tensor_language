"""FAKE BATTERY v3 -- fixes from 330: (i) family-disjoint foreign sets;
(ii) per-candidate SIGN-MIXEDNESS (minority fraction of member damage
signs; a pure severity cone scores ~0); (iii) unmatchable controls ==
structural rejection.
REGISTERED PREDICTIONS:
  (a) >=4/5 reals pass (sel>=2, spec>=1.5, sign-minority>=0.15);
  (b) F1 by selectivity; F2 structural;
  (c) F3 not real-passing; (d) F5 sign-minority <=0.10;
  (e) F4 rejected. Fakes passing everything = holes, reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fake_battery3_results.json'
CA=300; NB=78
MHL=list(range(2,10))
T=256


def safe_svd(X):
    try:
        return torch.linalg.svd(X,full_matrices=False)
    except Exception:
        U,S,Vh=torch.linalg.svd(X.double().cpu(),full_matrices=False)
        return U.float().to(X.device),S.float().to(X.device),\
            Vh.float().to(X.device)

@torch.no_grad()
def main():
    t0=time.time()
    import tiktoken as tk3
    from datasets import load_dataset
    enc5=tk3.get_encoding('gpt2')
    dsf=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    fr=[]
    for di in range(3000,10000):
        tkn=enc5.encode_ordinary(dsf[di]['text'])
        for st0 in range(0,len(tkn)-513,513):
            row=tkn[st0:st0+513]
            if tuple(row[:32]) in seen: continue
            fr.append(row)
            if len(fr)>=100: break
        if len(fr)>=100: break
    rows=torch.cat([FW[300:512],torch.tensor(fr,dtype=torch.long)])
    g0=torch.Generator().manual_seed(7)
    rows=rows[torch.randperm(rows.shape[0],generator=g0)]
    assert rows.shape[0]==NB*4, rows.shape
    ntok=NB*4*256
    rowid=torch.arange(NB*4)[:,None].expand(-1,256).reshape(-1)
    def ce_vec(hooks):
        ces=[]
        for i in range(0,NB*4,4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return torch.cat(ces)
    base=ce_vec([])
    # component output means + output caches (for slice-PCA probes)
    sums={}; hs=[]
    outcache={}
    for li in range(18):
        for kind,mod in (('a',m.transformer.h[li].attn),
                         ('m',m.transformer.h[li].mlp)):
            key=f'{kind}{li}'; sums[key]=torch.zeros(D,device=DEV)
            def mk(key=key):
                def h(mo,i_,o_):
                    y=o_[0] if isinstance(o_,tuple) else o_
                    sums[key]+=y.detach().float().reshape(-1,D).sum(0)
                return h
            hs.append(mod.register_forward_hook(mk()))
    for i in range(0,NB*4,4):
        bb=rows[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    mus={k:v/ntok for k,v in sums.items()}
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})
    def comp_probe(key):
        mu=mus[key]; mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    def head_probe(li,hd):
        at=m.transformer.h[li].attn
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
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2f.float(),
                             k2f.float())/128
            pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
            z[:,hd]=0
            ynew=at.c_proj(z.transpose(1,2).contiguous()
                           .view(B,T,-1).to(X.dtype))
            return (ynew,v1r)
        return [at.register_forward_hook(fh)]
    OUTCAP={}; SLICES={}
    def capture_out(key):
        if key in OUTCAP: return OUTCAP[key]
        mod=MODS[key]; capsX=[]
        def cap(mo,i_,o_):
            y=o_[0] if isinstance(o_,tuple) else o_
            capsX.append(y.detach().float().clamp(-6e4,6e4)
                         .half().reshape(-1,D).cpu())
        h=mod.register_forward_hook(cap)
        for i in range(0,NB*4,4):
            bb=rows[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        h.remove()
        OUTCAP[key]=torch.cat(capsX)
        return OUTCAP[key]
    def slice_pca_probe(key,tag,block):
        # ablate output projection onto slice-conditioned PCA block
        slice_idx=SLICES[tag]
        Y=capture_out(key)[slice_idx].float().to(DEV)
        Y=torch.nan_to_num(Y,nan=0.0,posinf=0.0,neginf=0.0)
        _,_,Vh=safe_svd((Y-Y.mean(0))[:20000])
        s0,s1=block
        P=orth(Vh[s0:s1].T)
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
        return [MODS[key].register_forward_hook(fh)]
    def classify_rows(Tk):
        n=Tk.shape[0]
        Mid=torch.zeros(n,256,dtype=torch.long)
        for r in range(n):
            tt_=Tk[r,:257].tolist()
            for pos in range(256):
                t=tt_[pos+1]; p=tt_[pos]
                tg=enc.decode([t]); pv=enc.decode([p]); st=tg.strip()
                if st.isdigit() and not tg.startswith(' '): k=0
                elif st in (')',']') and any(b in enc.decode(
                    tt_[max(0,pos-60):pos+1]) for b in ('(','[')): k=1
                elif chr(10) in tg: k=2
                elif tg in ('.','!','?'): k=3
                elif tg==',': k=4
                elif (tg.startswith(' ') and st[:1].isupper() and
                      (pv.strip()[:1].isupper() if pv.strip()
                       else False)): k=5
                elif t==p: k=6
                elif (not tg.startswith(' ')) and st.isalpha(): k=7
                elif t in tt_[:pos+1]: k=8
                else: k=9
                Mid[r,pos]=k
        return Mid
    cls=classify_rows(rows).reshape(-1)
    Yoh=torch.zeros(ntok,10)
    Yoh[torch.arange(ntok),cls]=1.0
    toks=rows[:,:256].reshape(-1)
    dmg_cache={}
    def damage(pspec):
        key=str(pspec)
        if key in dmg_cache: return dmg_cache[key]
        kind=pspec[0]
        if kind=='comp': hooks=comp_probe(pspec[1])
        elif kind=='head': hooks=head_probe(pspec[1],pspec[2])
        else: hooks=slice_pca_probe(pspec[1],pspec[2],pspec[3])
        # (pca pspec: ('pca', key, slice_tag, (s0,s1)))
        d=(ce_vec(hooks)-base).cpu()
        dmg_cache[key]=d
        return d
    def factor(pspecs,slice_idx,nmodes,tag):
        cols=[]
        for ps in pspecs:
            cols.append(damage(ps)[slice_idx])
        M=torch.nan_to_num(torch.stack(cols,1),nan=0.0,posinf=0.0,neginf=0.0)
        sd=M.std(0,keepdim=True).clamp_min(1e-6)
        M=torch.clamp((M-M.mean(0))/sd,-3,3)
        rid=rowid[slice_idx]
        med=rid.median()
        ha=rid<=med; hb=~ha
        if ha.sum()<200 or hb.sum()<200: return []
        U,Sg,Vh=safe_svd(M)
        _,_,Va=safe_svd(M[ha]-M[ha].mean(0))
        _,_,Vb=safe_svd(M[hb]-M[hb].mean(0))
        out=[]
        for k in range(min(nmodes,len(Sg))):
            repl=abs(float(Va[k]@Vb[k]))
            sc=U[:,k]*Sg[k]
            r2b=0; cb=None
            for c in range(10):
                yc=Yoh[slice_idx,c]
                if yc.sum()<5: continue
                r=float(torch.corrcoef(torch.stack([sc,yc]))[0,1])**2
                if r>r2b: r2b=r; cb=CLS[c]
            loading=Vh[k]
            topi=loading.abs().argsort(descending=True)[:4].tolist()
            thr=sc.abs().quantile(0.85)
            member=slice_idx[sc.abs()>=thr]
            ii=sc.abs().argsort(descending=True)[:20]
            extok=[enc.decode([int(toks[slice_idx[i]])])
                   for i in ii.tolist()[:6]]
            ctxs=[]
            for i in ii.tolist()[:20]:
                gi=int(slice_idx[i])
                r_,p_=gi//256,gi%256
                lo=max(0,p_-10); hi=min(256,p_+3)
                seg=rows[r_,lo:hi].tolist()
                mark=p_-lo
                txt=enc.decode(seg[:mark])+' [['+enc.decode(
                    [seg[mark]])+']] '+enc.decode(seg[mark+1:])
                ctxs.append(txt)
            out.append({'tag':f'{tag}.{k}','repl':round(repl,3),
                        'class_r2':round(r2b,3),'best_class':cb,
                        'top_probes':[pspecs[i] for i in topi],
                        'n_members':int(len(member)),
                        'sample_tokens':extok,'contexts':ctxs,
                        '_member':member,'_loading':loading.clone(),
                        '_score':sc.clone(),'_slice':slice_idx})
        return out
    # ---- level 0 ----
    P0=[('comp',f'{k}{li}') for li in range(18) for k in ('a','m')]
    P0+=[('head',li,hd) for li in MHL for hd in range(9)]
    allidx=torch.arange(ntok)
    print(f'level 0: {len(P0)} probes',flush=True)
    roots=factor(P0,allidx,24,'r')
    leaves=[]; child_stats=[0,0]
    def fine_probes(md):
        fine=[]; seen=set()
        SLICES[md['tag']]=md['_member']
        for ps in md['top_probes']:
            if ps[0]=='comp':
                key=ps[1]
                if ('c',key) in seen: continue
                seen.add(('c',key))
                if key[0]=='m':
                    for blk in ((0,8),(8,16),(16,32),(32,64)):
                        fine.append(('pca',key,md['tag'],blk))
                else:
                    li=int(key[1:])
                    for hd in range(9):
                        fine.append(('head',li,hd))
            elif ps[0]=='head':
                li=ps[1]
                if ('h',li) in seen: continue
                seen.add(('h',li))
                for blk in ((0,4),(4,16)):
                    fine.append(('pca',f'a{li}',md['tag'],blk))
            else:
                _,key,tag,(s0,s1)=ps
                mid=(s0+s1)//2
                if mid>s0:
                    fine.append(('pca',key,md['tag'],(s0,mid)))
                    fine.append(('pca',key,md['tag'],(mid,s1)))
        return fine
    def recurse(md,depth):
        if depth>2: return
        fine=fine_probes(md)
        if len(fine)<4: return
        subs=factor(fine[:24],md['_member'],4,md['tag'])
        for sm in subs:
            child_stats[1]+=1
            okc=sm['repl']>=0.6
            print('  '*depth+f"{sm['tag']}: repl {sm['repl']} class "
                  f"{sm['best_class']} {sm['class_r2']} "
                  f"n={sm['n_members']} tokens {sm['sample_tokens']}",
                  flush=True)
            if okc:
                child_stats[0]+=1
                leaves.append(sm)
                recurse(sm,depth+1)
    for md in roots:
        ok=md['repl']>=0.6
        print(f"{md['tag']}: repl {md['repl']} class {md['best_class']} "
              f"{md['class_r2']} n={md['n_members']} "
              f"tokens {md['sample_tokens']}",flush=True)
        if not ok: continue
        leaves.append(md)
        recurse(md,1)
    print(f'tree built: {len(leaves)} leaves; fake battery',flush=True)
    basev=base.cpu()
    LBY={lf['tag']:lf for lf in leaves}
    reals=['r.8.0.0','r.1.1.0','r.1.1.2','r.8.0','r.7.2.1']
    reals=[t for t in reals if t in LBY]
    g9=torch.Generator().manual_seed(5)
    rowhalf=(torch.arange(rows.shape[0])%2==0)
    ho=~rowhalf[:,None].expand(-1,256).reshape(-1)   # held-out tokens
    fit=~ho
    def probeset(tag):
        comps=set()
        for ps in LBY[tag]['top_probes']:
            if ps[0]=='comp': comps.add(('comp',ps[1]))
            elif ps[0]=='head': comps.add(('head',ps[1],ps[2]))
            else: comps.add(('comp',ps[1]))
        return comps
    def match_members(target_mem,n):
        mb=basev[target_mem]
        lo,hi=mb.quantile(0.1),mb.quantile(0.9)
        cand=torch.nonzero((basev>=lo)&(basev<=hi)).squeeze(1)
        mset=set(target_mem.tolist())
        cand=torch.tensor([int(c) for c in cand.tolist()
                           if int(c) not in mset])
        return cand[torch.randperm(len(cand),generator=g9)[:n]]
    cands=[]
    for t in reals:
        cands.append({'name':t,'kind':'real','probes':probeset(t),
                      'mem':LBY[t]['_member']})
    mA=LBY[reals[0]]['_member']
    cands.append({'name':'F1_random','kind':'fake',
                  'probes':probeset(reals[0]),
                  'mem':match_members(mA,len(mA))})
    heavy={('comp','m1'),('comp','m0'),('comp','m17'),('comp','m3')}
    frag=basev.argsort(descending=True)[:len(mA)]
    cands.append({'name':'F2_severity','kind':'fake','probes':heavy,
                  'mem':frag})
    mB=LBY[reals[1]]['_member']
    fr=torch.cat([mA[:len(mA)//2],mB[:len(mB)//2]])
    cands.append({'name':'F3_franken','kind':'fake',
                  'probes':probeset(reals[0]),'mem':fr})
    topic=torch.arange(20*256,24*256)
    dmg_cols={}
    def joint_damage(probes):
        key=tuple(sorted(map(str,probes)))
        if key in dmg_cols: return dmg_cols[key]
        hooks=[]
        for c in probes:
            if c[0]=='comp': hooks+=comp_probe(c[1])
            else: hooks+=head_probe(c[1],c[2])
        d=(ce_vec(hooks)-base).cpu()
        dmg_cols[key]=d
        return d
    tprobes=[]
    for c in [('comp',f'{k}{li}') for li in range(18) for k in 'am']:
        dsing=joint_damage({c})
        tprobes.append((float(dsing[topic].mean()),c))
    tprobes.sort(reverse=True)
    cands.append({'name':'F4_topic','kind':'fake',
                  'probes':{c for _,c in tprobes[:4]},'mem':topic})
    gset={c for _,c in sorted([(float(joint_damage({c}).mean()),c)
          for c in [('comp',f'{k}{li}') for li in range(18)
                    for k in 'am']],reverse=True)[:4]}
    dG=joint_damage(gset)
    fitidx=torch.nonzero(fit).squeeze(1)
    selN=len(mA)
    adv=fitidx[dG[fitidx].argsort(descending=True)[:selN]]
    cands.append({'name':'F5_adversarial','kind':'fake','probes':gset,
                  'mem':adv})
    # battery
    results=[]
    for i,cd in enumerate(cands):
        own=joint_damage(cd['probes'])
        def rootof(nm2):
            return nm2.split('.')[1] if nm2.startswith('r.') else nm2
        myroot=rootof(cd['name'])
        far=[c2 for c2 in cands if c2['kind']=='real'
             and rootof(c2['name'])!=myroot]
        fcd=far[0] if far else cands[(i+1)%len(cands)]
        foreign=joint_damage(fcd['probes'])
        mem=cd['mem'].long()
        memho=mem[ho[mem]]
        if len(memho)<30: memho=mem
        ctl=match_members(mem,len(mem))
        if ctl is None:
            results.append({'name':cd['name'],'kind':cd['kind'],
                            'selectivity':-1.0,
                            'selectivity_insample':-1.0,
                            'specificity':-1.0,
                            'structural_reject':True})
            print(f"{cd['name']:16s} STRUCTURALLY REJECTED "
                  f"(unmatchable members)",flush=True)
            continue
        ctl=torch.as_tensor(ctl).long()
        ctlho=ctl[ho[ctl]]
        e_own=float(own[memho].abs().mean())
        e_ctl=float(own[ctlho].abs().mean())
        e_for=float(foreign[memho].abs().mean())
        selv=e_own/max(e_ctl,1e-4)
        spec=e_own/max(e_for,1e-4)
        memfit=mem[fit[mem]]
        ctlfit=ctl[fit[ctl]]
        selv_in=float(own[memfit].abs().mean())/            max(float(own[ctlfit].abs().mean()),1e-4)
        sg=torch.sign(own[memho])
        smix=float(min((sg>0).float().mean(),(sg<0).float().mean()))
        results.append({'name':cd['name'],'kind':cd['kind'],
                        'selectivity':round(selv,2),
                        'selectivity_insample':round(selv_in,2),
                        'specificity':round(spec,2),
                        'sign_minority':round(smix,3)})
        print(f"{cd['name']:16s} sel {selv:5.2f} (in-sample "
              f"{selv_in:5.2f}) spec {spec:5.2f} smix {smix:.2f}",flush=True)
    R={r['name']:r for r in results}
    def ok_real(r):
        return (r['selectivity']>=2 and r['specificity']>=1.5
                and r.get('sign_minority',0)>=0.15)
    realpass=sum(1 for t in reals if ok_real(R[t]))
    pa=realpass>=4
    pb=(R['F1_random']['selectivity']<=1.3 and
        R['F2_severity'].get('structural_reject',False))
    pc=not ok_real(R['F3_franken'])
    pdx=R['F5_adversarial'].get('sign_minority',1)<=0.10
    pe=R['F4_topic']['selectivity']<=1.5
    holes=[r['name'] for r in results if r['kind']=='fake'
           and ok_real(r)]
    out={'results':results,'real_pass':realpass,'holes':holes,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pdx),'pred_e':bool(pe)}
    print(f"(a) reals pass ({realpass}/5): {'HELD' if pa else 'FAILED'}")
    print(f"(b) F1 sel + F2 structural: {'HELD' if pb else 'FAILED'}")
    print(f"(c) F3 not real-passing: {'HELD' if pc else 'FAILED'}")
    print(f"(d) F5 sign-minority <=0.10: {'HELD' if pdx else 'FAILED'}")
    print(f"(e) F4 rejected: {'HELD' if pe else 'FAILED'}")
    if holes: print(f"** HOLES STANDING: {holes} **")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
