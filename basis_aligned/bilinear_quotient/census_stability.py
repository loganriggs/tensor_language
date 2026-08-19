"""CENSUS STABILITY -- diagnose 378's collapse (29 leaves at 2x, 9%
identity). Rebuild the tree on the fresh 212 rows ALONE: same size
as the original recipe, disjoint corpus.
REGISTERED PREDICTIONS (explicit fork):
  (a) >=50 leaves at matched size -> the 2x collapse was corpus
      MIXING (heterogeneity blurs damage modes), not size-scaling;
      if <50, the recipe itself is size-fragile;
  (b) identity between this tree and the 424-tree RESTRICTED TO THE
      SAME FRESH ROWS (same data, different clustering context):
      >=40% at J>=0.5 -> clustering is data-determined and the 424
      collapse was a thresholds problem; <40% -> the clustering
      itself is context-dependent and leaf identity needs
      window-replication as the production unit;
  (c) state saved (census_state_fresh212.pt)."""
import json, sys, time, torch

def safe_svd(X):
    X=torch.nan_to_num(X.float()).clamp(-6e4,6e4)
    try: return torch.linalg.svd(X,full_matrices=False)
    except Exception:
        U,S,Vh=torch.linalg.svd(X.double().cpu(),full_matrices=False)
        return U.float().to(X.device),S.float().to(X.device),                Vh.float().to(X.device)
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'census_stability_results.json'
CA=300; NB=53
MHL=list(range(2,10))
T=256

@torch.no_grad()
def main():
    t0=time.time()
    import census_lib as cl9
    rows=cl9.fresh_rows(212)
    assert rows.shape[0]==NB*4
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
            capsX.append(y.detach().half().reshape(-1,D).cpu())
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
    import tiktoken as tk9
    enc9=tk9.get_encoding('gpt2')
    def _cls_rows(R9):
        Mid=torch.zeros(R9.shape[0],256,dtype=torch.long)
        for r in range(R9.shape[0]):
            toks=R9[r,:257].tolist()
            for pos in range(256):
                t=toks[pos+1]; p=toks[pos]
                tg=enc9.decode([t]); pv=enc9.decode([p]); s=tg.strip()
                if s.isdigit() and not tg.startswith(' '): k=0
                elif s in (')',']') and any(b9 in enc9.decode(
                    toks[max(0,pos-60):pos+1]) for b9 in ('(','[')): k=1
                elif chr(10) in tg: k=2
                elif tg in ('.','!','?'): k=3
                elif tg==',': k=4
                elif (tg.startswith(' ') and s[:1].isupper() and
                      (pv.strip()[:1].isupper() if pv.strip()
                       else False)): k=5
                elif t==p: k=6
                elif (not tg.startswith(' ')) and s.isalpha(): k=7
                elif t in toks[:pos+1]: k=8
                else: k=9
                Mid[r,pos]=k
        return Mid
    cls=_cls_rows(rows).reshape(-1)
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
        M=torch.stack(cols,1)
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
                        '_member':member,'_score':sc,
                        '_slice':slice_idx})
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
    nleaf=len(leaves)
    nnew=sum(1 for lf in leaves if lf['class_r2']<=0.15)
    crate=child_stats[0]/max(child_stats[1],1)
    pa=nleaf>=40
    pb=nnew>=nleaf/2
    pc=crate>=0.5
    out={'n_leaves':nleaf,'n_new':nnew,'child_replication_rate':
         round(crate,3),'leaves':[{k:v for k,v in lf.items()
         if k!='top_probes' and not k.startswith('_')} | {'top_probes':[str(p) for p in
         lf['top_probes']]} for lf in leaves],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    packs=[{'tag':lf['tag'],'repl':lf['repl'],
            'class_r2':lf['class_r2'],'n_members':lf['n_members'],
            'top_probes':[str(p) for p in lf['top_probes']],
            'contexts':lf['contexts']} for lf in leaves]
    torch.save({'rows':rows,'basev':base.cpu(),
                'leaves':[{'tag':lf['tag'],'repl':lf['repl'],
                           'class_r2':lf['class_r2'],
                           'n_members':lf['n_members'],
                           'top_probes':[str(p) for p in
                                         lf['top_probes']],
                           'member':lf['_member'],
                           'score':lf['_score'],
                           'slice':lf['_slice']} for lf in leaves]},
               PT+'census_state_fresh212.pt')
    for lf in leaves:
        for k_ in ('_member','_score','_slice'): lf.pop(k_,None)
    for md in roots: md.pop('_member',None)
    json.dump(packs,open(PT+'circuit_tree_packs_fresh212.json','w'),indent=1)
    pd_=all(len(lf['contexts'])>=12 for lf in leaves)
    out_pd=pd_
    print(f"(d) all packs >=12 contexts: {'HELD' if pd_ else 'FAILED'}")
    print(f'\nleaves {nleaf} | new-structure {nnew} | child repl rate '
          f'{crate:.2f}')
    pa=nleaf>=50
    print(f"(a) >=50 leaves at matched size: {'HELD' if pa else 'FAILED'}")
    big=torch.load(PT+'census_state_424.pt',
                   map_location='cpu')['leaves']
    OLDGRID=212*256
    newst=torch.load(PT+'census_state_fresh212.pt',
                     map_location='cpu')['leaves']
    matched=0; pairs=[]
    for lf9 in newst:
        A=set(int(x) for x in lf9['member'])
        bestJ=0; bestT=None
        for lf2 in big:
            B=set(int(x)-OLDGRID for x in lf2['member']
                  if int(x)>=OLDGRID)
            if not B: continue
            J=len(A&B)/max(len(A|B),1)
            if J>bestJ: bestJ,bestT=J,lf2['tag']
        pairs.append({'fresh':lf9['tag'],'in424':bestT,
                      'J':round(bestJ,3)})
        if bestJ>=0.5: matched+=1
    frac=matched/max(len(newst),1)
    pb=frac>=0.4
    out['identity_pairs']=pairs
    out['identity_frac']=round(frac,3)
    print(f'same-data identity vs 424-tree: {matched}/{len(newst)} '
          f'({frac:.0%}) at J>=0.5')
    print(f"(b) >=40% same-data identity: {'HELD' if pb else 'FAILED'}")
    print(f"(c) state saved: HELD")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(PT+'circuit_tree_packs_summary.json','w'),indent=1)
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
