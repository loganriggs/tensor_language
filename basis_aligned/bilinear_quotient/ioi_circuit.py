"""IOI CIRCUIT -- the window opened (ioi_floor: 99% pair accuracy,
margin +2.41). Localize: per-component mean-ablation (36 comps) on
the 96 IOI prompts, margin drop each; then head-level deletion in
the top attention layers. Means computed on the census grid
(comp_means). Control: ablations on shuffled-name control margins
should stay ~0.
REGISTERED PREDICTIONS:
  (a) concentration: <=6 components account for >=70% of the total
      positive margin drop;
  (b) >=1 mid-or-late ATTENTION layer in the top-3 owners (name
      moving is attention work);
  (c) head-level: deleting the top-2 heads of the top attention
      owner drops margin >=50% of that component's full drop."""
import json, time, itertools, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ioi_circuit_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    enc=cl.enc()
    names=[' Mary',' John',' Anna',' Peter',' Sarah',' Tom',
           ' Alice',' Bob']
    ok=[n for n in names if len(enc.encode(n))==1]
    pairs=list(itertools.combinations(ok,2))[:8]
    TEMPL=['When{A} and{B} went to the store,{B} gave the drink to',
           'When{A} and{B} got home,{B} handed the keys to',
           'After{A} and{B} left the party,{B} gave the coat to',
           'Then{A} and{B} went to the park, and{B} threw the ball to',
           'While{A} and{B} were cooking,{B} passed the salt to',
           'When{A} and{B} finished lunch,{B} gave the bill to']
    prompts=[]
    for A,B in pairs:
        for a,b in ((A,B),(B,A)):
            for tpl in TEMPL:
                prompts.append((tpl.replace('{A}',a).replace('{B}',b),
                                enc.encode(a)[0],enc.encode(b)[0]))
    def margin(hooks=()):
        ms=[]
        for txt,ti,ts in prompts:
            ids=torch.tensor(enc.encode(txt))[None,:].to(DEV)
            x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(
                F.rms_norm(x,(D,)))/30)).float()[0,-1]
            ms.append(float(lg[ti]-lg[ts]))
        for h in hooks: h.remove()
        return sum(ms)/len(ms)
    base=margin()
    print(f'base margin {base:+.3f}',flush=True)
    drops={}
    for li in range(18):
        for kd in ('a','m'):
            key=f'{kd}{li}'
            drops[key]=round(base-margin(cl.mean_hooks([key])),3)
    top=sorted(drops.items(),key=lambda kv:-kv[1])
    print('top drops:',top[:8],flush=True)
    tot=sum(v for _,v in drops.items() if v>0)
    run=0; k6=0
    for _,v in top:
        if v<=0: break
        run+=v; k6+=1
        if run>=0.7*tot: break
    pa=k6<=6
    pb=any(k[0]=='a' and int(k[1:])>=4 for k,_ in top[:3])
    topa=next((k for k,_ in top if k[0]=='a'),None)
    hc={}
    if topa:
        li=int(topa[1:])
        mod2=__import__('sys').modules[
            type(m.transformer.h[0].attn).__module__]
        are=mod2.apply_rotary_emb
        def del_heads(li,hds):
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,at=at,hds=hds):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                Bb,T=X.shape[0],X.shape[1]
                v=at.c_v(X).view(Bb,T,9,128)
                vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                cos,sin=at.rotary(at.c_q(X).view(Bb,T,9,128))
                qf=F.rms_norm(at.c_q(X).view(Bb,T,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(Bb,T,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2=F.rms_norm(at.c_q2(X).view(Bb,T,9,128),(128,))
                k2=F.rms_norm(at.c_k2(X).view(Bb,T,9,128),(128,))
                q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                kf.float())/128
                s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                k2.float())/128
                pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
                z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                for hd in hds: z[:,hd]=0
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,T,-1).to(X.dtype))
                return (yn,v1r)
            return [at.register_forward_hook(fh)]
        per=[]
        for hd in range(9):
            per.append((hd,round(base-margin(del_heads(li,[hd])),3)))
        per.sort(key=lambda kv:-kv[1])
        hc={'layer':li,'per_head':per,
            'top2_drop':round(base-margin(del_heads(li,
                [per[0][0],per[1][0]])),3)}
        print(f'head drops a{li}: {per}',flush=True)
    pc=bool(hc) and hc['top2_drop']>=0.5*drops[topa]
    out={'base_margin':round(base,3),'drops':drops,
         'top':top[:10],'k70':k6,'heads':hc,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) <=6 comps for 70% ({k6}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) mid/late attn in top-3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) top-2 heads >=50% of {topa}: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
