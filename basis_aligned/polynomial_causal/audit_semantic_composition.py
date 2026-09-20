"""Verify primitive replays and decompose joint error without changing gates."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'basis_aligned/bilinear_quotient/circuits/followups'

def main():
    result={};checks=[];primitive_checks=[]
    for version,prior in [('698r1','696'),('699','697')]:
        r=json.loads((ART/f'subject_attention_freeze_v{version}_result.json').read_text())
        old=json.loads((ART/f'subject_attention_freeze_v{prior}_result.json').read_text());records=[]
        for source,cells in r['two_site_reports'].items():
            for family,c in cells.items():
                for key in ['target','prediction','native_modal','predicted_modal']:
                    primitive_checks.append(float(np.max(np.abs(np.array(r['attractor_reports'][source][family][key])-old['attractor_reports'][source][family][key]))))
                    primitive_checks.append(float(np.max(np.abs(np.array(r['reuse_reports'][source][family][key])-old['reuse_reports'][source][family][key]))))
                budget=c['source_budget']
                total=np.array(c['prediction'])-c['target']
                primitive=np.array(c['subject_prediction'])-c['subject_effect']+np.array(c['attractor_prediction'])-c['attractor_effect']
                interaction=np.array(c['predicted_cross'])-c['native_cross']
                checks.append(float(np.max(np.abs(total-primitive-interaction))))
                subject=r['tangent_reports']['pruned'][source][family];attractor=r['attractor_reports'][source][family]
                modal_total=np.array(c['predicted_modal'])-c['native_modal']
                modal_primitive=np.array(subject['predicted_modal'])-subject['native_modal']+np.array(attractor['predicted_modal'])-attractor['native_modal']
                modal_interaction=modal_total-modal_primitive
                records.append(dict(source=source,family=family,number_total_over_budget=float(np.linalg.norm(total)/budget),
                    number_primitive_error_sum_over_budget=float(np.linalg.norm(primitive)/budget),number_interaction_error_over_budget=float(np.linalg.norm(interaction)/budget),
                    number_total_absolute_rms=float(np.sqrt(np.mean(total**2))),
                    modal_total_over_budget=(np.linalg.norm(modal_total,axis=0)/budget).tolist(),
                    modal_primitive_sum_over_budget=(np.linalg.norm(modal_primitive,axis=0)/budget).tolist(),
                    modal_interaction_error_over_budget=(np.linalg.norm(modal_interaction,axis=0)/budget).tolist(),
                    gate_failure=c['number_error_over_budget']>.10 or max(c['modal_error_over_budget'])>.05))
        result[version]=records
    assert max(primitive_checks)==0 and max(checks)<1e-12
    output=dict(primitive_replay_max_difference=max(primitive_checks),number_error_identity_max_difference=max(checks),records=result,
                scope='Vector identity separates primitive prediction errors from cross-effect errors; norms are not additive and this is not unique causal attribution.')
    Path(__file__).with_name('SEMANTIC_COMPOSITION_CPU_AUDIT_2026-09-20.json').write_text(json.dumps(output,indent=2)+'\n')
    print('primitive replay',max(primitive_checks),'identity',max(checks))
    for version,rs in result.items():
        for row in rs:
            if row['gate_failure']:print(version,row)
if __name__=='__main__':main()
