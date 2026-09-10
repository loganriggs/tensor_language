#!/usr/bin/env python3
# BQGATE: 0forwards0seq; equal-token weight varimax240s.
import os,sys,json,time,signal,hashlib,shutil
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from orthogonal_output_varimax_v1 import fit,sparsity
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    return v
def main():
    binding=json.loads((P/'OUTPUT_VARIMAX_NORMALIZED_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'OUTPUT_VARIMAX_V1_RESULT.json').read_text())['predictions']['pred_a_instrument']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,seconds=240,criterion='equal_token_row_varimax')));return
    out=P/'OUTPUT_VARIMAX_NORMALIZED_V1_RESULT.json';assert not out.exists();assert shutil.disk_usage(P).free>10_000_000
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    source=P/'OUTPUT_VARIMAX_V1_CHECKPOINT.pt';s=torch.load(source,weights_only=False,map_location='cuda')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda()
    a=(u-s['unembedding_mean'])@s['writer'];norm=a.norm(dim=1,keepdim=True);assert float(norm.min())>1e-12
    normalized=a/norm;unit_error=float((normalized.norm(dim=1)-1).abs().max())
    before=sparsity(a);before_equal=sparsity(normalized)
    state=fit(normalized,seconds=240);rotation=state['rotation'];rotated=a@rotation;after=sparsity(rotated);after_equal=sparsity(normalized@rotation)
    raw_reference=sparsity(a@s['rotation']);equal_reference=sparsity(normalized@s['rotation'])
    torch.manual_seed(561);probe=torch.randn(4608,16,device=a.device,dtype=a.dtype);original=a@(s['core']@probe);replayed=rotated@((rotation.T@s['core'])@probe)
    replay=float((original-replayed).norm()/original.norm());energy_bridge=abs(float(rotated.square().sum()/a.square().sum())-1)
    final=state['history'][-1];valid=max(unit_error,replay,energy_bridge,final['orthogonality_error'],state['maximum_score_decrease'])<=1e-10
    ap=P/'OUTPUT_VARIMAX_NORMALIZED_V1_CHECKPOINT.pt';assert not ap.exists();saved=cpu(state);saved.update(factor_source=str(source),factor_source_sha256=digest(source),config=dict(output_functions=128,metric='centered_U_weight',criterion='equal_token_row_varimax'),next_chunk=1);torch.save(saved,ap)
    reduction=after['median_token_factor_participation']<=.75*before['median_token_factor_participation'];top4=after_equal['token_top4_energy_fraction']>=.5
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_converged':valid and state['converged'],'pred_c_equal_token_sparse_usage':valid and reduction and top4},before=before,before_equal_token=before_equal,after=after,after_equal_token=after_equal,raw_varimax_reference=raw_reference,raw_varimax_reference_equal_token=equal_reference,median_reduction_clause=reduction,equal_token_top4_clause=top4,final=final,terminal_reason=state['terminal_reason'],bridges=dict(unit_rows=unit_error,rotation_replay=replay,raw_energy=energy_bridge),price=dict(body_forwards=0,fit_seconds=state['chunk_seconds'],iterations=state['iteration'],checkpoint_bytes=ap.stat().st_size,required_factor_source_bytes=source.stat().st_size),checkpoint_sha256=digest(ap),wall_seconds=time.perf_counter()-tic,scope='Equal-token rotation objective, samecenteredrank128tensorprojection. Rownormalizationonlyduringdiscovery, unnormalizedfunctionspreserved. No circuit/globaloptimum/stabilityclaim.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
