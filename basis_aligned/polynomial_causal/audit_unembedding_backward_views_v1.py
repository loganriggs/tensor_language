"""Post-result paired uncertainty audit; no threshold changes or new fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;src=p/'UNEMBEDDING_BACKWARD_VIEWS_V1_RESULT.json'
    r=json.loads(src.read_text());reports={}
    for j,(name,row) in enumerate(r['reports'].items()):
        bs=PairedPanelBootstrap(16,9111341+j);arms={}
        gold=np.asarray(row['live_ce_change_per_row'])
        for key,arm in row['arms'].items():
            errors=np.asarray(arm['ce_change_per_row'])-gold
            arms[key]={'live_effect_error_ci95':bs.relative_l2(arm['live_error_squared_per_row'],row['live_effect_squared_per_row']),
                       'ce_mae_ci95':bs.mean(abs(errors)),
                       'mean_signed_ce_error':float(errors.mean())}
        reports[name]={'arms':arms,'live_ce_change_ci95':bs.mean(gold),
                       'live_effect_norm':row['live_effect_norm'],
                       'mlp16_swap_recovery':row['live_mlp16_swap_recovery']}
    result={'schema':'unembedding.backward_views.audit.v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reports':reports,
            'scope':'4000 paired group resamples per existing authored panel. Conditional finite-panel uncertainty; no fresh OOD or universal guarantee. Frozen predicates unchanged.'}
    with (p/'UNEMBEDDING_BACKWARD_VIEWS_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['reports'],indent=2))


if __name__=='__main__':main()
