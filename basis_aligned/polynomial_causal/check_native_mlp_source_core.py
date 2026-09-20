"""Replay exported fixed-context local source core on CPU without model weights."""
import json
from pathlib import Path
import torch
from normalized_mlp_source_core import evaluate
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    x=torch.load(A/'native_mlp_source_core_v1.pt',map_location='cpu',weights_only=True)
    errors=[float((evaluate(x['core'],a[None,:]).sum(0)-ref).abs().max()) for a,ref in zip(x['amplitudes'],x['references'])]
    assert max(errors)<1e-8 and all(t.device.type=='cpu' for t in x['core'].values())
    r=json.loads((A/'native_mlp_source_core_v1_result.json').read_text())
    out=dict(cpu_replay_max_abs=max(errors),runtime_values=sum(v.numel() for v in x['core'].values()),artifact_bytes=(A/'native_mlp_source_core_v1.pt').stat().st_size,scope=x['scope'],max_number_error=max(c['number_error'] for c in r['records']),max_modal_error=max(max(c['modal_error']) for c in r['records']),max_local_gradient_replay=max(c['local_gradient_replay'] for c in r['checks']),max_local_hessian_replay=max(c['local_hessian_replay'] for c in r['checks']),max_local_output_replay=max(c['local_output_replay'] for c in r['checks']))
    (P/'NATIVE_MLP_SOURCE_CORE_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
