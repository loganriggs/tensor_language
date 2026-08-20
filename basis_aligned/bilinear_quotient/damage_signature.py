"""DAMAGE SIGNATURE -- 466: the surviving effect is scoped (464):
ablating r.13.2.1's bundle spares punctuation (-0.025) and digits
(-0.018) while damaging space-words (+0.014), capitalised (+0.006)
and newlines most (+0.027). Four leaves have now independently
"discovered" the punctuation half of it -- the signature of a
population-level phenomenon rather than a circuit.
Final scope question, and the deepest available deflation: is this
class profile a property of THIS bundle, or of DAMAGE ITSELF?
Identical class breakdown, four interventions, same fresh rows:
  bundle : r.13.2.1's 16-dim probe bundle (reference)
  random : a rank-matched random subspace in the same components
  head   : deleting one mid-stack attention head (6.1)
  mlp    : mean-ablating one MLP (m9)
REGISTERED PREDICTIONS:
  (a) UNIVERSAL: the random-subspace arm reproduces the profile
      (punct < 0, digit < 0, newline > 0);
  (b) ACROSS TYPES: the head and MLP arms reproduce it too;
  (c) if (a) and (b) hold the survivor is a UNIVERSAL DAMAGE
      SIGNATURE of this model and not a circuit property; if they
      fail the bundle's profile is specific after all -- recorded
      either way."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'damage_signature_results.json'
TAG='r.13.2.1'; NFRESH=48
KINDS=['punct','newline','digit','subword','space_word',
       'capitalized']

def classify(tok):
    s=cl.d1(int(tok)); st=s.strip()
    return {'punct':bool(st) and not any(c.isalnum() for c in st),
            'newline':chr(10) in s,'digit':st.isdigit(),
            'subword':(not s.startswith(' ')) and st.isalpha(),
            'space_word':s.startswith(' ') and st.isalpha(),
            'capitalized':s.startswith(' ') and bool(st)
                          and st[:1].isupper()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    import ast as A
    probes=[A.literal_eval(p) if isinstance(p,str) else p
            for p in cl.leaf(TAG)['top_probes']]
    rk={}
    for p in probes:
        key=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
        n=(p[3][1]-p[3][0]) if p[0]=='pca' else 8
        rk[key]=rk.get(key,0)+n
    def mk_bundle():
        return cl.proj_hooks(cl.leaf(TAG)['top_probes'])
    def mk_random():
        hs=[]
        for key,n in rk.items():
            gg=torch.Generator(device=DEV).manual_seed(707)
            P=orth(torch.randn(D,n,generator=gg,device=DEV))
            mod=MODS[key]
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
            hs.append(mod.register_forward_hook(fh))
        return hs
    def mk_head():
        at=m.transformer.h[6].attn
        def fh(mo_,args,o_,at=at):
            y,v1r=o_
            X=args[0]; B=X.shape[0]
            v1=args[1] if args[1] is not None else v1r
            v=at.c_v(X).view(B,T,9,128)
            vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
            cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
            def r2(w):
                return are(F.rms_norm(w(X).view(B,T,9,128),
                           (128,)),cos,sin)
            qf,kf=r2(at.c_q),r2(at.c_k)
            q2,k2=r2(at.c_q2),r2(at.c_k2)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
            z[:,1]=0
            return (at.c_proj(z.transpose(1,2).contiguous()
                    .view(B,T,-1).to(X.dtype)),v1r)
        return [at.register_forward_hook(fh)]
    def mk_mlp():
        mu=mus['m9'].to(DEV)
        def fh(mo,i_,o_,mu=mu):
            return mu.expand_as(o_).to(o_.dtype)
        return [MODS['m9'].register_forward_hook(fh)]
    ARMS={'bundle':mk_bundle,'random':mk_random,'head':mk_head,
          'mlp':mk_mlp}
    fresh=cl.fineweb_rows(NFRESH)
    masks={k:torch.zeros(NFRESH,T,dtype=torch.bool) for k in KINDS}
    for r in range(NFRESH):
        for q in range(T):
            c=classify(int(fresh[r,q+1]))
            for k in KINDS: masks[k][r,q]=c[k]
    print({k:int(masks[k].sum()) for k in KINDS},flush=True)
    ALL={}
    for arm,mk in ARMS.items():
        dce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            def fwd(use):
                hs=mk() if use else []
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
                v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                                  /30)).float()
                c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                  reduction='none').view(B,T).cpu()
                for h in hs: h.remove()
                return c
            dce[i:i+4]=fwd(True)-fwd(False)
        dd=dce.reshape(-1)
        prof={}
        for k in KINDS:
            mk2=masks[k].reshape(-1)
            if int(mk2.sum())<20: prof[k]=None; continue
            prof[k]=round(float(dd[mk2].mean()
                                -dd[~mk2].mean()),4)
        ALL[arm]=prof
        print(f'{arm}: '+', '.join(f'{k} {prof[k]}'
                                   for k in KINDS),flush=True)
    def ok(a):
        d=ALL[a]
        return ((d.get('punct') or 0)<0 and (d.get('digit') or 0)<0
                and (d.get('newline') or 0)>0)
    pa=ok('random'); pb=ok('head') and ok('mlp')
    out={'profiles':ALL,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':True,'runtime_s':time.time()-t0}
    for nm,v in (('a','random subspace reproduces the profile'),
                 ('b','head and MLP arms reproduce it'),
                 ('c','verdict recorded either way')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
