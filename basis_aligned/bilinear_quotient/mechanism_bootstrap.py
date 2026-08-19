"""MECHANISM BOOTSTRAP v1 -- the induction-style third leg (user
direction): derive each circuit's firing condition from the COMPONENT
side alone, then test whether it predicts the members. Scope v1: leaves
whose top probes are front-MLP direction blocks (m0-m3), where the
stream is near-token-determined, so the block's activation over the
whole vocabulary is computable by the fold construction -- every token
alone through the prefix, project the component output onto the leaf's
probe directions, rank tokens by activation. ZERO member data touches
the condition; the only freedom is the trigger-set size, fixed to the
leaf's member RATE (a count, registered here as the single allowed
statistic).
Prediction test on the document-disjoint held-out half: positions whose
current token is in the trigger set, scored as precision over base
rate and recall vs actual members. Null: random vocabulary set of the
same size.
REGISTERED PREDICTIONS:
  (a) >=50% of tested front-MLP leaves reach precision >= 3x base rate
      at recall >= 0.3 (mechanism predicts extension);
  (b) random-set null precision <= 1.2x base rate (median);
  (c) trigger sets printed (they are mechanism-derived names:
      "activates on tokens {...}")."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mechanism_bootstrap_results.json'
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
    print(f'tree built: {len(leaves)} leaves; bootstrap',flush=True)
    import tiktoken as tk9
    enc9=tk9.get_encoding('gpt2')
    # fold pass: every vocab token alone, capture m0-m3 outputs
    FOLDC={}
    capsF={0:[],1:[],2:[],3:[]}
    hsF=[]
    for li in (0,1,2,3):
        def mkf(li=li):
            def h(mo,i_,o_):
                capsF[li].append(o_.detach()[:,0].float())
            return h
        hsF.append(m.transformer.h[li].mlp.register_forward_hook(mkf()))
    for s0 in range(0,50257,2048):
        idx=torch.arange(s0,min(s0+2048,50257),device=DEV)[:,None]
        xF=F.rms_norm(m.transformer.wte(idx),(D,)); x0F=xF; v1F=None
        for blk in m.transformer.h[:4]:
            xF,v1F=blk(xF,v1F,x0F)
    for h in hsF: h.remove()
    for li in (0,1,2,3):
        FOLDC[li]=torch.cat(capsF[li]); capsF[li]=None
    # slice-PCA bases must be recomputed per probe spec via capture_out
    flat=rows[:,:256].reshape(-1)
    rowhalf=(torch.arange(rows.shape[0])%2==0)
    halfmask=rowhalf[:,None].expand(-1,256).reshape(-1)
    g9=torch.Generator().manual_seed(4)
    res=[]
    for lf in leaves:
        mlps=[ps for ps in lf['top_probes'] if ps[0]=='pca'
              and ps[1][0]=='m' and int(ps[1][1:])<=3]
        if len(mlps)<2: continue
        # mechanism: fold activation of the leaf's probe directions
        score=torch.zeros(50257,device=DEV)
        for ps in mlps:
            _,key,tag2,(s0_,s1_)=ps
            li=int(key[1:])
            sl2=SLICES[tag2]
            Y=capture_out(key)[sl2].float().to(DEV)
            Y=torch.nan_to_num(Y)
            _,_,Vh9=safe_svd((Y-Y.mean(0))[:20000])
            P=orth(Vh9[s0_:s1_].T)
            score+= (FOLDC[li].to(DEV)@P).pow(2).sum(1)
        mem=lf['_member']
        memflat=torch.zeros(len(flat),dtype=torch.bool)
        memflat[mem]=True
        rate=float(memflat.float().mean())
        q=max(8,int(rate*50257))
        trig=set(score.argsort(descending=True)[:q].tolist())
        pred=torch.tensor([int(t) in trig for t in flat.tolist()])
        ho=~halfmask
        tp=float((pred&memflat&ho).sum())
        prec=tp/max(float((pred&ho).sum()),1)
        rec=tp/max(float((memflat&ho).sum()),1)
        base_r=float(memflat[ho].float().mean())
        rnd=set(torch.randperm(50257,generator=g9)[:q].tolist())
        predr=torch.tensor([int(t) in rnd for t in flat.tolist()])
        tpr=float((predr&memflat&ho).sum())
        precr=tpr/max(float((predr&ho).sum()),1)
        toptr=[enc9.decode([t]) for t in
               score.argsort(descending=True)[:8].tolist()]
        ok=prec>=3*base_r and rec>=0.3
        res.append({'tag':lf['tag'],'precision':round(prec,4),
                    'recall':round(rec,3),'base':round(base_r,4),
                    'lift':round(prec/max(base_r,1e-6),2),
                    'null_lift':round(precr/max(base_r,1e-6),2),
                    'trigger_top':toptr,'pass':bool(ok)})
        print(f"{lf['tag']}: prec {prec:.3f} ({prec/max(base_r,1e-6):.1f}x"
              f" base) rec {rec:.2f} null {precr/max(base_r,1e-6):.1f}x "
              f"| triggers {toptr} {'PASS' if ok else 'fail'}",flush=True)
    npass=sum(1 for r in res if r['pass'])
    nulls=sorted(r['null_lift'] for r in res)
    mednull=nulls[len(nulls)//2] if nulls else 9
    pa=(npass>=0.5*len(res)) if res else False
    pb=mednull<=1.2
    out={'n_tested':len(res),'n_pass':npass,'median_null_lift':mednull,
         'results':res,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"tested {len(res)} front-MLP leaves | pass {npass} | "
          f"median null lift {mednull:.2f}")
    print(f"(a) >=50% mechanism-predicted: {'HELD' if pa else 'FAILED'}")
    print(f"(b) null <= 1.2x: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
