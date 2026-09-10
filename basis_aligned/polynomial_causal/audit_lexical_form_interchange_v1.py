"""Paired semantic-factor audit and saved-state composition diagnostic.

No checkpoint load. State addition is a report-only diagnostic of where the
interaction occurs, not an independent upstream program or new passing gate.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;source=p/'LEXICAL_FORM_INTERCHANGE_V1_RESULT.json';r=json.loads(source.read_text())
    state=p/'LEXICAL_FORM_INTERCHANGE_V1_STATES.pt';assert hashlib.sha256(state.read_bytes()).hexdigest()==r['artifact_sha256']
    saved=torch.load(state,map_location='cpu',weights_only=True)
    H=torch.tensor([[1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],[1,-1,-1,1]],dtype=torch.float64);reports={}
    center=lambda x:x-x.mean(-1,keepdim=True)
    for j,name in enumerate(['A1','A2']):
        panel=r['reports'][name];s=saved[name];bs=PairedPanelBootstrap(16,9111561+j)
        interval=lambda m:bs.relative_l2(m['error_squared_per_row'],m['reference_squared_per_row'])
        U=torch.einsum('ba,nbd->nad',H,s['reader_components']);h={k:v['h'].double() for k,v in s['states'].items()};z={k:v['selected_scores'].double() for k,v in s['states'].items()}
        def read(x):return 30*torch.tanh(torch.einsum('nkd,nd->nk',U,x)/(30*torch.sqrt(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps)))
        bridge=max(float((read(h[k])-z[k]).abs().max()) for k in h);assert bridge<=1e-3
        target=center(z['J']-z['B']);summed=read(h['LB']+h['FB']-h['B']);error=center(summed-z['J'])
        state_error=h['J']-h['LB']-h['FB']+h['B'];state_ref=h['J']-h['B']
        additive_score=center(z['LB']+z['FB']-z['B']-z['J'])
        reports[name]=dict(transfers={k:dict(mean=v['mean'],ci95=None if v['per_row'] is None else bs.mean(v['per_row'])) for k,v in panel['transfers'].items()},
            preservation_ci95={k:interval(v) for k,v in panel['preservation'].items()},joint_error_ci95=interval(panel['joint']),full_addition_error_ci95=interval(panel['addition']),
            failed_native_rows={k:[i for i,v in enumerate(panel['selected_scores'][k]) if int(np.argmax(v))!=idx] for k,idx in [('B',0),('X',2),('Y',1),('Z',3)]},
            score_components={k:(v@H.T/4).tolist() for k,v in z.items()},
            saved_readout_bridge_max_abs=bridge,state_addition_selected_effect_error=float(error.norm()/target.norm()),state_addition_error_ci95=bs.relative_l2(error.square().sum(-1).tolist(),target.square().sum(-1).tolist()),
            score_addition_selected_effect_error=float(additive_score.norm()/target.norm()),raw_state_addition_effect_error=float(state_error.norm()/state_ref.norm()))
    g=r['reports']['G'];bs=PairedPanelBootstrap(16,9111563)
    reports['G']=dict(mean_absolute_margin=g['mean_absolute_margin'],margin_abs_ci95=bs.mean(np.abs(g['margin_change_per_row'])),ce_abs_ci95=bs.mean(np.abs(g['ce_change_per_row'])),
        natural_mean_absolute_margin=float(np.mean(np.abs(g['natural_margin_change_per_row']))),common_score_mean_absolute_change=float(np.mean(np.abs(g['common_score_change_per_row']))))
    out=dict(schema='lexical.form_interchange.audit.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,
        native_checkpoint_loaded=False,native_forwards=0,gpu_accessed=False,scope='All opened groups,4000 paired draws. State-addition diagnostic retains exact selected readout normalizer/softcap; native singleton state generation remains required. Original registered verdicts unchanged.')
    with (p/'LEXICAL_FORM_INTERCHANGE_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({n:{k:v for k,v in report.items() if k!='score_components'} for n,report in reports.items()},indent=2))


if __name__=='__main__':main()
