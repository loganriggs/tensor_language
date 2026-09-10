"""Paired finite-panel uncertainty for native head-space split interventions."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap
P=Path(__file__).resolve().parent


def main():
    path=P/'CORRELATIVE_COMPLEMENT_SPLIT_V1_RESULT.json';r=json.loads(path.read_text());assert r['predictions']['pred_a_instrument']
    reports={}
    for index,name in enumerate(('A1','A2','C')):
        row=r['reports'][name];b=PairedPanelBootstrap(16,9111314+index)
        base=np.array(row['native_base_margin']);den=base+np.array(row['native_donor_margin']);assert (den>1e-6).all()
        ref=row['effects']['full']['effect_squared_per_row'];arms={}
        for arm,e in row['effects'].items():
            damage=np.array(row['removal'][arm]['ce_damage_per_row'])
            arms[arm]={'recovery_ci95':b.mean((base-np.array(e['margin']))/den),
                'full_effect_error_ci95':b.relative_l2(e['full_error_squared_per_row'],ref),
                'mean_ce_damage_ci95':b.mean(damage),'mean_absolute_ce_change_ci95':b.mean(np.abs(damage)),
                'positive_damage_rows':int((damage>0).sum())}
        reports[name]={'arms':arms,'interaction_ci95':b.relative_l2(row['interaction_squared_per_row'],ref)}
    result={'schema':'correlative.complement_split.audit.v1','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'bootstrap_helper_sha256':hashlib.sha256((P/'paired_panel_bootstrap_v1.py').read_bytes()).hexdigest(),
        'seeds':[9111314,9111315,9111316],'draws_per_panel':4000,'reports':reports,
        'scope':'Post-result paired resampling of16authored groups perpanel; no population/training OOD inference, threshold adjustment or rescue.'}
    with (P/'CORRELATIVE_COMPLEMENT_SPLIT_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
