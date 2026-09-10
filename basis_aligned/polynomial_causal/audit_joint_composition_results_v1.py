"""Saved-state coefficient lower bounds and paired joint-routing statistics."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from paired_panel_bootstrap_v1 import PairedPanelBootstrap
P=Path(__file__).resolve().parent

def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    receipt=json.loads((P/'UNEMBEDDING_JOINT_FAMILY_V1_RESULT.json').read_text());ap=P/'UNEMBEDDING_JOINT_FAMILY_V1_PROGRAM.pt';assert hashlib.sha256(ap.read_bytes()).hexdigest()==receipt['artifact_sha256']
    program=torch.load(ap,map_location='cpu',weights_only=True);S=program['selected_native_quadratics'];bounds={}
    for token,s in zip(program['tokens'],S):
        vals=torch.linalg.eigvalsh(s);energy=vals.square().sort(descending=True).values;total=energy.sum();needed=int(torch.searchsorted(energy.cumsum(0),.99*total))+1
        bounds[token]=dict(rank_needed_at_10pct_frobenius_error=needed,real_product_count_lower_bound=(needed+1)//2,coefficient_norm=float(total.sqrt()),minimum_rank6_relative_error=float((energy[6:].sum()/total).sqrt()),scope='Every sum of k real linear-form products has rank<=2k. Necessary condition on this full-input coefficient norm, not an activation-domain or behavioral bound.')
    flat=S.flatten(1);e=torch.linalg.eigvalsh(flat@flat.T).clamp_min(0).sort(descending=True).values
    span_rank=int(torch.searchsorted(e.cumsum(0),.99*e.sum()))+1
    q=json.loads((P/'CORRELATIVE_JOINT_QK_SUBSPACES_V1_RESULT.json').read_text());paired={}
    for i,(name,r) in enumerate(q['reports'].items()):
        b=PairedPanelBootstrap(8,9113200+i);paired[name]={}
        for fam,m in r['families'].items():
            paired[name][fam]=dict(relative_error=m['relative_error'],error_interval=b.relative_l2(m['error_squared'],m['reference_squared']),relative_effect=m['relative_effect'],effect_interval=b.relative_l2(m['effect_squared'],m['reference_squared']))
    out=dict(product_lower_bounds=bounds,shared_output_span_rank_needed_at_10pct_family_frobenius_error=span_rank,joint_routing_paired=paired,position_scope='V1 bases fit after rotary transforms: A1 query position5 versus A2 position8. Cross-frame failure can mix semantic failure with coordinate rotation; it does not prove raw-input subspaces absent. Next must identify pre-rotary joint spaces or transport the bases with the native position maps.',native_forwards=0,checkpoint_loaded=False,cpu_seconds=time.perf_counter()-tic,audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),program_sha256=receipt['artifact_sha256'])
    target=P/'JOINT_COMPOSITION_RESULTS_V1_AUDIT.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
