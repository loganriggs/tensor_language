"""Paired source-screen intervals and descriptive shared/remainder accounting.

Reuses saved folded readers and native states; no checkpoint or GPU access.
Panel means are report-only summaries of already opened rows, not new circuits.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;source=p/'TOKEN_CONTEXT_SOURCE_V1_RESULT.json'
    r=json.loads(source.read_text());state_path=p/'TOKEN_CONTEXT_SOURCE_V1_ARTIFACT.pt'
    assert hashlib.sha256(state_path.read_bytes()).hexdigest()==r['artifact_sha256']
    states=torch.load(state_path,map_location='cpu',weights_only=True);reports={};structure={}
    for j,name in enumerate(('A1','A2','G')):
        bs=PairedPanelBootstrap(16,9111521+j);reports[name]={}
        for site,a in r['reports'][name]['arms'].items():
            cross=np.asarray(a['gate_cross_per_row']);den=np.asarray(a['gate_reference_squared_per_row'])
            denominators=den[bs.indices].sum(1);assert (denominators>0).all()
            reports[name][site]=dict(gate_transfer_ci95=np.quantile(cross[bs.indices].sum(1)/denominators,[.025,.975]).tolist(),
                gate_error_ci95=bs.relative_l2(a['gate_error_squared_per_row'],den),
                mean_absolute_ce_ci95=bs.mean(np.abs(a['ce_change_per_row'])))
        s=states[name];k=s['context_reader'];u=s['u17'];du=u.roll(-1,0)-u
        mean=k.mean(0);remainder=k-mean
        exact=(k*du).sum(-1);shared=du@mean;individual=(remainder*du).sum(-1)
        recorded=torch.tensor(r['reports'][name]['gate_difference'],dtype=torch.float64)
        closure=float((exact-shared-individual).abs().max());replay=float((exact-recorded).abs().max())
        assert closure<=1e-8 and replay<=1e-8
        structure[name]=dict(shared_plus_remainder_max_abs=closure,saved_gate_replay_max_abs=replay,
            mean_only_gate_error=float((exact-shared).norm()/exact.norm()),
            remainder_only_gate_error=float((exact-individual).norm()/exact.norm()),
            shared_gate_delta=shared.tolist(),token_remainder_gate_delta=individual.tolist(),
            shared_remainder_cross=float(shared@individual),
            mean_only_error_ci95=bs.relative_l2(individual.square().tolist(),exact.square().tolist()))
    out=dict(schema='token.context_source.audit.v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,structure=structure,
        nominees=r['nominees'],native_checkpoint_loaded=False,native_forwards=0,gpu_accessed=False,
        scope='4000 paired resamples per opened panel; intervals are descriptive and not simultaneous selection guarantees. Shared reader is the panel mean after a linear weight pullback; exact remainder retained. Not independently validated structure, behavioral recovery, or extracted context production.')
    with (p/'TOKEN_CONTEXT_SOURCE_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(dict(nominees=out['nominees'],structure={n:{k:v for k,v in s.items() if not isinstance(v,list)} for n,s in structure.items()},
        example_intervals={n:{site:reports[n][site] for site in ('attn_11','mlp_16')} for n in reports}),indent=2))


if __name__=='__main__':main()
