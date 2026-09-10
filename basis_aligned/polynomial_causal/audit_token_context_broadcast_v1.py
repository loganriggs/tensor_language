"""Paired analysis of the fixed first-value factorial; no model access."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;source=p/'TOKEN_CONTEXT_BROADCAST_V1_RESULT.json'
    r=json.loads(source.read_text());reports={}
    for j,(name,panel) in enumerate(r['reports'].items()):
        bs=PairedPanelBootstrap(16,9111541+j);arms={}
        for label,a in panel['arms'].items():
            cross=np.asarray(a['gate_cross_per_row']);ref=np.asarray(a['gate_reference_squared_per_row']);den=ref[bs.indices].sum(1);assert (den>0).all()
            arms[label]=dict(gate_transfer_ci95=np.quantile(cross[bs.indices].sum(1)/den,[.025,.975]).tolist(),
                gate_error_ci95=bs.relative_l2(a['gate_error_squared_per_row'],ref),gate_magnitude_ci95=bs.relative_l2(a['gate_change_squared_per_row'],ref),
                mean_absolute_ce_ci95=bs.mean(np.abs(a['ce_change_per_row'])),full_effect_error_ci95=bs.relative_l2(a['full_error_squared_per_row'],panel['full_reference_squared_per_row']))
        reports[name]=dict(arms=arms,interaction_ci95=bs.relative_l2(panel['interaction_squared_per_row'],panel['full_reference_squared_per_row']))
        if 'lexical' in panel:
            l=panel['lexical'];reports[name]['lexical']=dict(both_correct=l['both_correct'],value_mean_recovery=None if l['value_recovery'] is None else float(np.mean(l['value_recovery'])),
                value_recovery_ci95=None if l['value_recovery'] is None else bs.mean(l['value_recovery']),other_mean_recovery=None if l['other_recovery'] is None else float(np.mean(l['other_recovery'])),
                failed_row_indices=[i for i,(a,b) in enumerate(zip(l['base_margin'],l['donor_margin'])) if not (a>0 and b<0)])
    out=dict(schema='token.context_broadcast.audit.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,
        native_checkpoint_loaded=False,native_forwards=0,gpu_accessed=False,scope='4000 paired resamples of all16 authored groups per opened panel; no outcome filtering or universal OOD claim. Failed lexical endpoints preserved.')
    with (p/'TOKEN_CONTEXT_BROADCAST_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
