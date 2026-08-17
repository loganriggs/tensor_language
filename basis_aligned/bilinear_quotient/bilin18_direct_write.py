"""The direct-write hypothesis: selectivity = how much the direction writes its named
tokens' logits directly.

§71: selectivity is direction-idiosyncratic. Mechanistic candidate: a steered write at
layer L rides the residual bypass to the unembedding; its FIRST-ORDER logit effect on
token t is proportional to wte_t . d (times the final-norm scaling). Selectivity
should track the contrast
    DW(d) = mean_{t in named} (wte_t . d)  -  mean_{t in ctrl} (wte_t . d)
normalised by the unembedding row scale -- computable from weights alone. REGISTERED
PREDICTIONS: (a) Spearman(|DW|, measured selectivity) >= 0.7 across the six §71
directions; (b) the numbers axis has the largest |DW| (its 3.82x is direct logit
writing, not semantic routing)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_selectivity_law import toks, spearman
from bilin18_gradient_steering import collect_basis
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
enc=tiktoken.get_encoding('gpt2')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_direct_write_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    meas=json.load(open('bilin18_selectivity_law_results.json'))['cases']
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
    dirs={'L0 punct':Q0[:,int(o[0])].float(),
          'L0 numbers':Q0[:,int(o[1])].float(),
          'L0 #3':Q0[:,int(o[2])].float(),
          'word2 openers':word(1),'word1 dets':word(0),'word5 meas':word(4)}
    named={'L0 punct':toks('.',',','!','?',';',':',')','(','"',"'"),
           'L0 numbers':toks(' 10',' first',' not',' one',' more',' no',' two',' 1'),
           'L0 #3':toks('.',' make','!',' work',';',' made',' get',' put'),
           'word2 openers':toks('(',' [',' "','We',' (',' The','[','"'),
           'word1 dets':toks(' your',' their',' both',' the',' our',' its',' his',' a'),
           'word5 meas':toks(' levels',' samples',' data',' measured',' rate',
                             ' values',' detected',' analysis')}
    CTRL=toks(' people',' world',' story',' house',' morning',' friend',' road',
              ' music',' game',' door')
    wte=m.transformer.wte.weight.detach().float()
    out={'cases':[]}
    dws=[]; sels=[]
    for c in meas:
        tag=c['tag']; d=dirs[tag]; d=d/d.norm()
        a=wte@d
        dw=abs(float(a[named[tag]].mean())-float(a[CTRL].mean()))
        dws.append(dw); sels.append(c['selectivity'])
        out['cases'].append({'tag':tag,'DW':dw,'selectivity':c['selectivity']})
        print(f"{tag:14s} |DW| {dw:.3f} -> selectivity {c['selectivity']:.2f}x",
              flush=True)
    rr=spearman(torch.tensor(dws),torch.tensor(sels))
    top=max(out['cases'],key=lambda c:c['DW'])['tag']
    pa=rr>=0.7; pb=top=='L0 numbers'
    out['spearman']=rr; out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f'\nSpearman(|DW|, selectivity) = {rr:+.2f} -> (a) '
          f"{'HELD' if pa else 'FAILED'}")
    print(f"largest DW: {top} -> (b) {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
