"""The regularity's decisive test: is even the rho=0.95 axis causally token-selective?

'Readable but not steerable' (§69) has held four-for-four. Layer 0's leader is the
program's crispest name -- a punctuation-vs-content axis verified at rho 0.95 against
a 0.14 null (§20). Steer L0's MLP output along d0 at +/-2 sigma; measure the full
swing of mean logprob for the punctuation set vs frequency-matched content controls.
REGISTERED PREDICTIONS:
  (a) per the regularity: the swing ratio punct/control < 1.5 (fails selectivity like
      the other four);
  (b) the alternative outcome -- ratio >= 2 -- would bound the regularity instead
      ('steerable only where naming is near-perfect'). Bar (a) is the prediction;
      (b) is the named alternative."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_word5_causal import full_logits
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import torch.nn.functional as F
D=1152
enc=tiktoken.get_encoding('gpt2')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_punct_causal_results.json')

@torch.no_grad()
def full_logits_l0(idx, steer=None):
    # same as full_logits but injecting at layer 0's MLP output
    from tier2_model import rope_tables, apply_rot
    NH,HD=9,128
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    for li in range(len(m.transformer.h)):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        a=blk.attn
        hcur=F.rms_norm(x,(D,))
        def qk(l):
            z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
            return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
        s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==0 and steer is not None:
            vec,mag=steer; mo=mo+(mag*vec).to(mo.dtype)
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    return F.log_softmax(logits.float(),-1)

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs); Y0c=(Y0-Y0.mean(0)).float()
    _,_,Vh0=torch.linalg.svd(Y0c, full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    d0=orth(Vh0[:32].T)[:,int(phi0.argmax())].float()
    s0=float((Y0c@d0).std())
    punct=[enc.encode(t)[0] for t in ('.',',','!','?',';',':',')','(','"',"'")]
    ctrl=[enc.encode(t)[0] for t in (' people',' world',' story',' house',
          ' morning',' friend',' road',' music',' game',' door')]
    rows=FW[300:312,:257].to(DEV)
    lp0=full_logits_l0(rows)
    sw={}
    for sgn in (+1,-1):
        lp=full_logits_l0(rows,steer=(d0,sgn*2*s0))
        sw[sgn]={'punct':float((lp[...,punct]-lp0[...,punct]).mean()),
                 'ctrl':float((lp[...,ctrl]-lp0[...,ctrl]).mean())}
        print(f'{"+" if sgn>0 else "-"}steer: punct {sw[sgn]["punct"]:+.4f} | '
              f'control {sw[sgn]["ctrl"]:+.4f}',flush=True)
    swing_p=sw[1]['punct']-sw[-1]['punct']
    swing_c=sw[1]['ctrl']-sw[-1]['ctrl']
    ratio=abs(swing_p)/max(abs(swing_c),1e-9)
    out={'swings':{'punct':swing_p,'ctrl':swing_c},'ratio':ratio,'per_sign':sw}
    pa=ratio<1.5
    out['regularity_held']=bool(pa); out['alternative_bound']=bool(ratio>=2)
    print(f'\nfull swing: punct {swing_p:+.4f} vs control {swing_c:+.4f} '
          f'({ratio:.1f}x)')
    print(f"regularity (ratio<1.5, five-for-five): {'HELD' if pa else 'FAILED'}"
          f"{' -- BOUNDED: steerable where naming is near-perfect' if ratio>=2 else ''}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
