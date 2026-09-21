"""Verify Schur exchange ranking against exhaustive direct solves, five families."""
import json
from pathlib import Path
import torch
from sparse_support_exchange import fitted_score,best_exchange,exchange_path
from sparse_quartic_bank import entries
P=Path(__file__).resolve().parent

def controls():
    torch.set_num_threads(2);rows=[];dtype=torch.float64
    for seed in range(5):
        torch.manual_seed(11600+seed);U=torch.randn(6,1,3,dtype=dtype);V=torch.randn_like(U)
        if seed==1:U[1]=U[0]
        if seed==2:V[1]=V[0]
        if seed==3:V=U.clone()
        if seed==4:U[1]=U[0];V[1]=-V[0]
        pairs=torch.triu_indices(6,6);indices=torch.cartesian_prod(*[torch.arange(3) for _ in range(4)])
        phi=entries(U,V,pairs,indices);G=phi.T@phi
        true=torch.randn(2,21,dtype=dtype);target=phi@true.T;X=target.T@phi
        selected=torch.arange(7);protected=[0,1];ridge=.01
        original,_=fitted_score(G,X,selected,ridge);proposal=best_exchange(G,X,selected,protected,ridge)
        direct=[]
        for i,old in enumerate(selected.tolist()):
            if old in protected:continue
            for new in range(21):
                if new in selected:continue
                take=selected.clone();take[i]=new;score,_=fitted_score(G,X,take,ridge)
                direct.append((float(score-original),old,new))
        best=max(direct)
        gap=abs(best[0]-proposal['predicted_score_gain'])/(1+abs(float(original)))
        assert gap<1e-8,(seed,gap)
        take,C,info=exchange_path(G,X,selected,protected,ridge,steps=5)
        assert len(torch.unique(take))==len(selected) and all(i in take for i in protected)
        assert info['final_score']>=info['initial_score']
        rows.append(dict(seed=seed,ranking_gap=gap,chosen_pair=[proposal['remove'],proposal['add']],bruteforce_pair=list(best[1:]),accepted_edits=len(info['history']),relative_score_gain=(info['final_score']-info['initial_score'])/(1+abs(info['initial_score']))))
    return rows
if __name__=='__main__':
    rows=controls();(P/'SPARSE_SUPPORT_EXCHANGE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
