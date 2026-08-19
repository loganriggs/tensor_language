"""Blind confirmation of the direct-write predictor.

§72: |DW| predicts steering selectivity at rho 0.77, retrodictively. Blind protocol:
STEP 1 computes |DW| for four never-steered named directions (L0 causal dirs #4 and
#5 with their §20-style fires-on sets; vocabulary words #3 and #4 with their §68
fires-on sets) and REGISTERS the predicted selectivity ordering before any steering.
STEP 2 measures. REGISTERED PREDICTIONS: (a) Spearman(predicted order, measured) >=
0.6 on the four; (b) the max-DW direction's measured selectivity >= 1.5x."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_selectivity_law import toks, spearman, swing
import bilin18_selectivity_law as SL
from bilin18_gradient_steering import collect_basis
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
enc=tiktoken.get_encoding('gpt2')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_direct_write_blind_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    SL.CTRL=toks(' people',' world',' story',' house',' morning',' friend',' road',
                 ' music',' game',' door')
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs); Y0c=(Y0-Y0.mean(0)).float()
    _,_,Vh0=torch.linalg.svd(Y0c, full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    o=phi0.argsort(descending=True); Q0=orth(Vh0[:32].T)
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
    cases=[('L0 #4',0,Q0[:,int(o[3])].float(),
            toks(' the','The',' The',' entire',' specific','the',' this',' that')),
           ('L0 #5',0,Q0[:,int(o[4])].float(),
            toks(' seeing',' fixed',' recommend',' requires',' allows',' Because',
                 ' discussed',' new')),
           ('word3',1,word(2),
            toks('�',' Hep',' Tru','em','(',' problem',' going',' work')),
           ('word4',1,word(3),
            toks(' er','�',' h',' problem',' going',' get',' make',' time'))]
    wte=m.transformer.wte.weight.detach().float()
    print('STEP 1 -- registered predictions (weights only):')
    dws=[]
    for tag,layer,d,named in cases:
        d=d/d.norm(); a=wte@d
        dw=abs(float(a[named].mean())-float(a[SL.CTRL].mean()))
        dws.append(dw)
        print(f'  {tag:8s} |DW| {dw:.1f}')
    order=sorted(range(4),key=lambda i:-dws[i])
    print(f'  predicted selectivity order: {[cases[i][0] for i in order]}\n')
    out={'dws':dws,'predicted_order':[cases[i][0] for i in order],'cases':[]}
    sels=[]
    for i,(tag,layer,d,named) in enumerate(cases):
        d=d/d.norm()
        s=float((Y0c@d).std()) if layer==0 else s1
        r=swing(d,s,layer,named)
        sels.append(r)
        out['cases'].append({'tag':tag,'DW':dws[i],'selectivity':r})
        print(f'{tag:8s} measured selectivity {r:.2f}x',flush=True)
    rr=spearman(torch.tensor(dws),torch.tensor(sels))
    imax=max(range(4),key=lambda i:dws[i])
    pa=rr>=0.6; pb=sels[imax]>=1.5
    out['spearman']=rr; out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f'\nSpearman(DW, measured) = {rr:+.2f} -> (a) '
          f"{'HELD' if pa else 'FAILED'}")
    print(f"max-DW ({cases[imax][0]}) measured {sels[imax]:.2f}x -> (b) "
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
