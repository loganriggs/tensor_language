"""Post-hoc combination of registered v686 output branches, no refitting."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'basis_aligned/bilinear_quotient/circuits/followups'

def main():
    r=json.loads((ART/'subject_attention_freeze_v686_result.json').read_text())
    report={}
    for source,cs in r['tangent_reports']['corrected'].items():
        report[source]={}
        for family,c in cs.items():
            linear=r['tangent_reports']['linear'][source][family]
            assert c['target']==linear['target'] and c['native_modal']==linear['native_modal']
            y=np.asarray(c['target']);den=np.linalg.norm(y)
            target_error=float(np.linalg.norm(np.array(c['prediction'])-y)/den)
            modal_error=(np.linalg.norm(np.array(linear['predicted_modal'])-linear['native_modal'],axis=0)/den).tolist()
            report[source][family]=dict(target_error=target_error,modal_error=modal_error,passes=target_error<=.10 and max(modal_error)<=.05)
    result=dict(posthoc=True,formula='number=sparse number prediction minus baseline gradient dotted with discarded initial delta; modals=negative baseline gradient dotted with full initial delta',
                passes_all_sources=all(c['passes'] for cs in report.values() for c in cs.values()),report=report,
                reader_values_per_batch=r['reader_values_per_batch'],scope='Output predictor at native prepared-context/full input-response ports; all reader generation and full residual input charged. No fresh evidence, no shared residual-state realization, no model replacement claim.')
    Path(__file__).with_name('SUBJECT_HYBRID_OPENED_AUDIT_2026-09-20.json').write_text(json.dumps(result,indent=2)+'\n')
    print('opened all-source gates',result['passes_all_sources'])
    print('reader values',result['reader_values_per_batch'])
if __name__=='__main__':main()
