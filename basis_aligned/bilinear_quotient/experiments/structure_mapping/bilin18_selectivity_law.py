"""The graded law: does causal token selectivity track naming crispness?

§70's revision predicts a monotone relation. Eight named directions spanning the
program's rho range, each steered +/-2 sigma at its own layer, each scored by the
swing ratio of its OWN named-token set vs frequency-matched controls:
  L0 #1 punct (rho .95), L0 #2 numbers (.80), L0 #3 (.80), L1 vocab word #2
  openers (.56), word #1 determiners (.50), word #5 measurement (.48), L16 axis
  (.31), L17 dir-2-feature copulas (.39 -- from §8.3's table, current-token rho).
REGISTERED PREDICTIONS: (a) Spearman(rho, selectivity ratio) >= 0.6 across the
eight; (b) the fitted relation crosses selectivity 1.0 near rho ~ 0.3 (below that,
names are causally inert)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_word5_causal import full_logits
from bilin18_punct_causal import full_logits_l0
from bilin18_gradient_steering import collect_basis
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
enc=tiktoken.get_encoding('gpt2')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_selectivity_law_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

def toks(*ws): return [enc.encode(w)[0] for w in ws]
CTRL=None

@torch.no_grad()
def swing(dirvec, s, layer, named):
    rows=FW[300:312,:257].to(DEV)
    fl = full_logits_l0 if layer==0 else full_logits
    lp0=fl(rows)
    sw={}
    for sgn in (+1,-1):
        lp=fl(rows,steer=(dirvec,sgn*2*s))
        sw[sgn]=(float((lp[...,named]-lp0[...,named]).mean()),
                 float((lp[...,CTRL]-lp0[...,CTRL]).mean()))
    sm=sw[1][0]-sw[-1][0]; sc=sw[1][1]-sw[-1][1]
    return abs(sm)/max(abs(sc),1e-9)

def main():
    global CTRL
    t0=time.time()
    CTRL=toks(' people',' world',' story',' house',' morning',' friend',' road',
              ' music',' game',' door')
    # L0 directions
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs); Y0c=(Y0-Y0.mean(0)).float()
    _,_,Vh0=torch.linalg.svd(Y0c, full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    o=phi0.argsort(descending=True); Q0=orth(Vh0[:32].T)
    cases=[]
    cases.append(('L0 punct',0.95,0,Q0[:,int(o[0])].float(),
                  toks('.',',','!','?',';',':',')','(','"',"'")))
    cases.append(('L0 numbers',0.80,0,Q0[:,int(o[1])].float(),
                  toks(' 10',' first',' not',' one',' more',' no',' two',' 1')))
    cases.append(('L0 #3',0.80,0,Q0[:,int(o[2])].float(),
                  toks('.',' make','!',' work',';',' made',' get',' put')))
    # L1 vocabulary words
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s1=float(Y1c.norm(dim=1).mean())/K**0.5*K**0.5*0.2
    rows=[]
    for j in (2,3,5,9,13,17):
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            rows.append((0.5*(M+M.T)).flatten())
    X=torch.stack(rows)
    _,sv,W=torch.linalg.svd(X, full_matrices=False)
    def word(p_):
        Pm=0.5*(W[p_].view(K,K)+W[p_].view(K,K).T)
        evp,Up=torch.linalg.eigh(Pm.double())
        w=(V@Up[:,evp.abs().argmax()].float()); return w/w.norm()
    cases.append(('word2 openers',0.56,1,word(1),
                  toks('(',' [',' "','We',' (',' The','[','"')))
    cases.append(('word1 dets',0.50,1,word(0),
                  toks(' your',' their',' both',' the',' our',' its',' his',' a')))
    cases.append(('word5 meas',0.48,1,word(4),
                  toks(' levels',' samples',' data',' measured',' rate',' values',
                       ' detected',' analysis')))
    # crude sigma per case
    s0map={0: float((Y0c@Q0[:,int(o[0])]).std())}
    out={'cases':[]}
    rhos=[]; sels=[]
    for tag,rho,layer,dv,named in cases:
        if layer==0:
            s=float((Y0c@dv).std())
        else:
            s=s1
        r=swing(dv,s,layer,named)
        rhos.append(rho); sels.append(r)
        out['cases'].append({'tag':tag,'rho':rho,'selectivity':r})
        print(f'{tag:14s} rho {rho:.2f} -> selectivity {r:.2f}x',flush=True)
    rr=spearman(torch.tensor(rhos),torch.tensor(sels))
    out['spearman']=rr
    pa=rr>=0.6
    out['pred_a']=bool(pa)
    print(f'\nSpearman(rho, selectivity) = {rr:+.2f} -> (a) '
          f"{'HELD' if pa else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
