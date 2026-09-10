"""Paired CPU uncertainty for the fixed scalar-read consumer experiment."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    directory=Path(__file__).resolve().parent
    source=directory/'GERUND_MLP_CONSUMER_V1_RESULT.json'; result=json.loads(source.read_text()); reports={}
    for index,name in enumerate(('A1','A2','G','C')):
        bootstrap=PairedPanelBootstrap(16,9111421+index); arms=result['reports'][name]['arms']
        live=np.asarray(arms['live_swap']['raw_recovery_per_row']); blocked=np.asarray(arms['blocked_swap']['raw_recovery_per_row'])
        live_ce=np.asarray(arms['live_zero']['ce_change_per_row']); blocked_ce=np.asarray(arms['blocked_zero']['ce_change_per_row'])
        reports[name]=dict(recovery_loss_ci95=bootstrap.mean(live-blocked),
            zero_ce_reduction_ci95=bootstrap.mean(live_ce-blocked_ce),
            half_damage_bar_margin_ci95=bootstrap.mean(.5*live_ce-blocked_ce),
            arms={key:dict(recovery_ci95=bootstrap.mean(value['raw_recovery_per_row']),
                           mean_ce_ci95=bootstrap.mean(value['ce_change_per_row']),
                           mean_absolute_ce_ci95=bootstrap.mean(np.abs(value['ce_change_per_row'])))
                  for key,value in arms.items()})
        for key in ('live_swap','blocked_swap'):
            reports[name]['arms'][key]['scalar_prediction_error_ci95']=bootstrap.relative_l2(arms[key]['error_squared_per_row'],arms[key]['effect_squared_per_row'])
    cosines=[x['context_reader_cosine'] for x in result['context_readers']]
    out=dict(schema='gerund.mlp_consumer.audit.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,
             context_reader_cosines=dict(minimum=min(cosines),median=float(np.median(cosines)),maximum=max(cosines)),
             scope='4000 paired resamples of16 reused authored contexts. No threshold changes, layer selection or independent OOD claims. Weight cosine is diagnostic, not causal attribution.')
    with (directory/'GERUND_MLP_CONSUMER_AUDIT_V1_RESULT.json').open('x') as stream:
        json.dump(out,stream,indent=2);stream.write('\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
