"""Paired native-context response errors; no altered hypotheses or directions."""
import json,hashlib
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;source=p/'NATIVE_RESPONSE_GATE_V1_RESULT.json';r=json.loads(source.read_text());reports={}
    for index,name in enumerate(('A1','A2','G','C')):
        bs=PairedPanelBootstrap(16,9111460+index);panel=r['reports'][name]
        reports[name]={'native_ce_ci95':bs.mean(panel['native_ce_per_row']),'schemes':{}}
        for label,scheme in panel['schemes'].items():
            reports[name]['schemes'][label]=dict(
                selected_error_ci95=bs.relative_l2(scheme['selected_error_squared_per_row'],scheme['selected_reference_squared_per_row']),
                full_effect_error_ci95=bs.relative_l2(scheme['full_error_squared_per_row'],scheme['full_effect_squared_per_row']),
                mean_ce_ci95=bs.mean(scheme['ce_change_per_row']),
                ce_prediction_mae_ci95=bs.mean(np.abs(np.asarray(scheme['ce_change_per_row'])-panel['native_ce_per_row'])))
    output=dict(schema='native.response_gate.audit.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,
                scope='4000 paired resamples of16 reused contexts; cyclic donors remain fixed. No changed gates, ranks, thresholds or pristine OOD claim.')
    with (p/'NATIVE_RESPONSE_GATE_AUDIT_V1_RESULT.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
