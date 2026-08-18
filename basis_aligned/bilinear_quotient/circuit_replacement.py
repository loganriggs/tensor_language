"""The compression handoff: SLICE-CONDITIONED REPLACEMENT for the two
site-local certified circuits (section 239). Per the scan-first rule, rung
0 first: replace each owner's component content AT SLICE SITES with a
slice-conditioned CONSTANT (for an MLP owner, the top-8 span coefficients
fixed to their slice-mean -- an 8-number description; for an attention
owner, the slice-mean output vector). Constants fit on window A (rows
300-512) slice sites, applied on window C (rows 120-300): cross-window by
construction. Recovery = 1 - (standin slice damage / on-slice ablation
slice damage). Rung 1 for whatever fails: rank-8 slice-linear refit
(coefficients predicted from the component input, ridge, fit window A).

REGISTERED PREDICTIONS: (a) digit: slice-constant recovers >= 40% (digit
continuation may be mostly a fixed 'boost digits' write); (b) subword:
ALTERNATIVE registered -- slice-constant recovers < 40% (completion needs
token-specific content; its 1.18-nat local damage suggests rich content),
and rung-1 slice-linear recovers >= 50%; (c) control: a random constant of
matched norm recovers <= 10% on both; (d) floors measured in-run per the
standing rule."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_replacement_results.json'
CA,CB=300,512     # fit window A
R0,R1=120,300     # eval window C

def slicemask(name,r0,r1):
    M=torch.zeros(r1-r0,256,dtype=torch.bool)
    for r in range(r0,r1):
        toks=FW[r,:257].tolist()
        for pos in range(256):
            tg=enc.decode([toks[pos+1]]); s=tg.strip()
            if name=='digit': v=s.isdigit() and not tg.startswith(' ')
            else: v=(not tg.startswith(' ')) and s.isalpha()
            M[r-R0 if r0==R0 else r-CA,pos]=v
    return M

@torch.no_grad()
def mlp_span(li):
    accs=[]
    for i in range(0,120,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
        accs.append(acc[0])
    Y=torch.cat(accs); Yb=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
    return orth(Vh[:8].T), Yb.float()

@torch.no_grad()
def capture_fit(name, owners):
    """Window A: capture owners' outputs & inputs at slice sites."""
    Ms=slicemask(name,CA,CB).to(DEV)
    caps={o:[[],[]] for o in owners}   # [outputs, inputs(normed x)]
    hs=[]
    for o in owners:
        if o.startswith('mlp'):
            li=int(o[3:])
            def mk(o=o,li=li):
                def hook(mod,i_,o_):
                    caps[o][0].append(o_.detach())
                    caps[o][1].append(i_[0].detach())
                return hook
            hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
        else:
            li=int(o[4:])
            def mka(o=o):
                return lambda mo_,i_,oo_: caps[o][0].append(
                    (oo_[0] if isinstance(oo_,tuple) else oo_).detach())
            hs.append(m.transformer.h[li].attn.register_forward_hook(mka()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    fit={}
    flat=Ms[:, :256].reshape(-1)
    for o in owners:
        Y=torch.cat([y.reshape(-1,D) for y in caps[o][0]]).float()
        Ys=Y[flat]
        if o.startswith('mlp'):
            Q,mu=mlp_span(int(o[3:]))
            cbar_slice=(Ys@Q).mean(0)
            X=torch.cat([x.reshape(-1,D) for x in caps[o][1]]).float()[flat]
            C=Ys@Q
            lam=1e-2*len(X)
            A=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),
                                 X.T@C)          # D x 8 ridge map
            fit[o]=('mlp',int(o[3:]),Q,mu,(Ys@Q).mean(0),A)
        else:
            fit[o]=('attn',int(o[4:]),None,attn_mean(int(o[4:])),
                    Ys.mean(0),None)
    return fit

@torch.no_grad()
def pertok(arms, maskC):
    """arms: list of (kind, fitspec) per owner. kind in
    ablate|const|linear|randconst."""
    hs=[]; cur={'b0':0}
    g=torch.Generator(device=DEV).manual_seed(0)
    for kind,spec in arms:
        typ=spec[0]
        if typ=='mlp':
            _,li,Q,mu,cslice,A=spec
            if kind=='randconst':
                r=torch.randn(8,device=DEV,generator=g)
                cs=r/r.norm()*cslice.norm()
            else: cs=cslice
            def mk(li=li,Q=Q,mu=mu,cs=cs,A=A,kind=kind):
                def hook(mod,i_,o_):
                    B,T,_=o_.shape
                    c=o_.float().reshape(-1,D)@Q
                    if kind=='ablate': tgtc=(mu@Q).expand_as(c)
                    elif kind=='linear':
                        x=i_[0].float().reshape(-1,D)
                        tgtc=x@A
                    else: tgtc=cs.expand_as(c)
                    delta=((c-tgtc)@Q.T).view(B,T,D)
                    mm=maskC[cur['b0']:cur['b0']+B,:T].to(DEV)
                    delta=delta*mm[:,:,None]
                    return o_-delta.to(o_.dtype)
                return hook
            hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
        else:
            _,li,_,amu,yslice,_=spec
            if kind=='randconst':
                r=torch.randn(D,device=DEV,generator=g)
                ys=r/r.norm()*yslice.norm()
            elif kind=='ablate': ys=amu
            else: ys=yslice
            def mka(li=li,ys=ys):
                def hook(mod,i_,o_):
                    out=o_[0] if isinstance(o_,tuple) else o_
                    B,T,_=out.shape
                    mm=maskC[cur['b0']:cur['b0']+B,:T].to(DEV)
                    new=torch.where(mm[:,:,None],
                                    ys[None,None,:].to(out.dtype)
                                    .expand_as(out),out)
                    if isinstance(o_,tuple): return (new,)+o_[1:]
                    return new
                return hook
            hs.append(m.transformer.h[li].attn.register_forward_hook(mka()))
    ces=[]
    for i in range(R0,R1,4):
        cur['b0']=i-R0
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    for h in hs: h.remove()
    return torch.cat(ces)

@torch.no_grad()
def main():
    t0=time.time()
    CIRCUITS={'digit':('attn8','mlp15'),'subword':('mlp16','mlp15')}
    base=pertok([],torch.zeros(R1-R0,256,dtype=torch.bool,device=DEV))
    res={}
    for name,owners in CIRCUITS.items():
        fit=capture_fit(name,owners)
        Mc=slicemask(name,R0,R1).to(DEV)
        flat=Mc.reshape(-1)
        def dmg(kind):
            arms=[(kind,fit[o]) for o in owners]
            return float((pertok(arms,Mc)-base)[flat].mean())
        d_abl=dmg('ablate'); d_con=dmg('const'); d_rnd=dmg('randconst')
        rec_c=1-d_con/max(d_abl,1e-6); rec_r=1-d_rnd/max(d_abl,1e-6)
        row={'ablate':round(d_abl,4),'const':round(d_con,4),
             'const_recovery':round(rec_c,2),
             'randconst_recovery':round(rec_r,2)}
        if rec_c<0.4 and any(o.startswith('mlp') for o in owners):
            d_lin=dmg('linear')
            row['linear']=round(d_lin,4)
            row['linear_recovery']=round(1-d_lin/max(d_abl,1e-6),2)
        res[name]=row
        print(f'{name:8s}: ablate {d_abl:+.3f} const {d_con:+.3f} '
              f'(rec {rec_c:.0%}) rnd-const rec {rec_r:.0%}'
              +(f" linear rec {row.get('linear_recovery','-')}"
                if 'linear' in row else ''),flush=True)
    pa=res['digit']['const_recovery']>=0.40
    pb_alt=res['subword']['const_recovery']<0.40
    pb_lin=res['subword'].get('linear_recovery',0)>=0.50 if pb_alt else None
    pc=all(r['randconst_recovery']<=0.10 for r in res.values())
    out={'circuits':res,'pred_a':bool(pa),'alt_b':bool(pb_alt),
         'pred_b_linear':None if pb_lin is None else bool(pb_lin),
         'pred_c':bool(pc)}
    print(f"\n(a) digit const rec >=40%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) subword const <40% (alt): {'YES' if pb_alt else 'no'}"
          f"{' | linear >=50%: '+('HELD' if pb_lin else 'FAILED') if pb_lin is not None else ''}")
    print(f"(c) random-const rec <=10%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
