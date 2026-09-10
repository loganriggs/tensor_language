"""Paired factor effects and saved-state replay; no additional model access."""
import hashlib,json
from pathlib import Path
import torch
from paired_panel_bootstrap_v1 import PairedPanelBootstrap
from quadratic_readout_state_v1 import evaluate,advance


def main():
    p=Path(__file__).resolve().parent;source=p/'GERUND_READOUT_FACTORIAL_V2_RESULT.json';r=json.loads(source.read_text());reports={}
    for index,name in enumerate(('A1','A2','G','C')):
        bs=PairedPanelBootstrap(16,9111481+index);panel=r['reports'][name]
        reports[name]=dict(interaction_ci95=bs.relative_l2(panel['interaction_squared_per_row'],panel['effect_squared_per_row']),
            arms={k:dict(effect_error_ci95=bs.relative_l2(v['error_squared_per_row'],v['effect_squared_per_row']),mean_ce_ci95=bs.mean(v['ce_per_row'])) for k,v in panel['arms'].items()})
    old=torch.load(p/'GERUND_READOUT_FACTORIAL_V1_STATES.pt',map_location='cpu',weights_only=True)
    new=torch.load(p/'GERUND_READOUT_FACTORIAL_V2_STATES.pt',map_location='cpu',weights_only=True)
    comparisons=[]
    def compare(a,b):
        if isinstance(a,torch.Tensor):comparisons.append(dict(equal=bool(torch.equal(a,b)),max_abs=float((a.double()-b.double()).abs().max())))
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        else:assert a==b
    compare(old,new)
    compose={k:float((evaluate(advance(v['token_state'],v['delta']*.3),v['delta']*.7)-evaluate(v['token_state'],v['delta'])).abs().max()) for k,v in new.items()}
    assert max(compose.values())<=1e-10
    output=dict(schema='gerund.readout_factorial.audit.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,
                failed_vs_repaired_state_exact=all(x['equal'] for x in comparisons),failed_vs_repaired_state_max_abs=max(x['max_abs'] for x in comparisons),reloaded_token_program_composition_max_abs=compose,
                native_checkpoint_loaded=False,native_forwards=0,gpu_accessed=False,
                scope='4000 paired resamples on reused authored rows. V1 saved states preserved and compared with V2 after publisher-only fix. Token program reloaded from state cache; initial-state production is not extracted.')
    with (p/'GERUND_READOUT_FACTORIAL_AUDIT_V1_RESULT.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
