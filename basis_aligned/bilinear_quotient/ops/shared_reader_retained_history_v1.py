"""Matched controller comparison: retain L-BFGS history until solver termination."""
import os,sys,json,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(P),str(ROOT)]
import torch
from bounded_shared_reader_fit_v1 import fit
from shared_reader_variable_projection_v2 import Objective
from ll1_joint_parent_graph_v2 import build,execute,factors,price
from symmetric_ll1_objective_v1 import cp
PREFIX='SHARED_READER_RETAINED_HISTORY_V1'
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def main(label,family):
    assert label == 'spectral' and family == 'original'
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    preflight=json.loads((P/'SHARED_READER_PROJECTED_PREFLIGHT_V1_RESULT.json').read_text())
    assert all(preflight['pred_'+letter] for letter in ('a','b','c'))
    control=json.loads((P/'BOUNDED_SHARED_READER_FIT_V1_CONTROL.json').read_text())
    assert control['pred_'+'a']  # Independent global recovery bar missed; retained explicitly.
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=0,corpus_access=False,
                              label=label,family=family,groups=64,rank=16,seconds=1200,
                              raw_entry_bounds=[-1,1],reencode_iteration_limit=1000000,penalty=.01,
                              global_planted_recovery='1/4; registered2/4bar missed')));return
    stem=f'{PREFIX}_{label.upper()}_{family.upper()}'
    out=P/f'{stem}.json';artifact=P/f'{stem}.pt';assert not out.exists() and not artifact.exists()
    signal.alarm(1800);torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.set_grad_enabled(True)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
    left,right,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    source=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt')
    saved=torch.load(source,weights_only=True,map_location='cpu')
    parts=tuple(v.double() for v in saved['parts']);wh=saved['output_whitener'].double()
    if family=='original':graph=build(*parts,torch.zeros(0,1152),[])
    else:graph=torch.load(P/'LL1_JOINT_CORE_SOLVE_V1_GRAPHS.pt',weights_only=True,map_location='cpu')[label]
    graph.update(output_whitener=wh,bias=sd['transformer.h.17.mlp.Down_bias'].clone(),source_sha256=digest(source))
    target=(left,right,wh.cuda()@down);total=99245061353.47293
    reference=json.loads((P/'LL1_JOINT_CORE_SOLVE_V1_AUDIT.json').read_text())
    expected=next(row for row in reference['rows'] if row['label']==label)['arms'][family]['after_objective']
    check=Objective(target,graph,total);initial=check.evaluate(check.initial)[0]
    initial_replay=abs(initial-expected);assert initial_replay<=1e-8
    del check
    fitted,stats=fit(target,graph,total,seconds=1200,penalty=.01,max_cycles=100,iterations_per_cycle=1000000)
    with torch.no_grad():
        bank=cp(*factors(fitted));torch.manual_seed(3101);x=torch.randn(13,1152,device='cuda')
        baseline=((x@bank[0].T)*(x@bank[1].T))@bank[2].T
        replay=float((execute(fitted,x)-baseline).norm()/baseline.norm())
    details=stats['details']
    numeric=(max(initial_replay,replay,details['reduced_identity_error'],stats['maximum_reencoding_objective_error'],stats['maximum_objective_increase'])<=1e-8
             and details['inner']['normal_residual']<=1e-10 and stats['maximum_raw_entry']<=1+1e-10)
    baseline=json.loads((P/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_ORIGINAL.json').read_text())
    baseline_objective=baseline['optimization']['final']
    predictions={'pred_a':bool(numeric),'pred_b':bool(stats['local_converged']),
                 'pred_c':bool(baseline_objective-stats['final']>=1e-4)}
    final=cpu(fitted);torch.save(final,artifact)
    result=dict(**predictions,label=label,family=family,capture=stats['capture'],
                baseline_objective=baseline_objective,objective_advantage=baseline_objective-stats['final'],
                iterations_per_cycle=1000000,baseline_iterations_per_cycle=200,initial_cpu_replay=initial_replay,
                executor_replay=replay,price=price(final,parts),optimization=stats,
                artifact=dict(path=str(artifact),sha256=digest(artifact)),
                scope='Matched20minute spectral-original optimization with retained L-BFGS history; fresh checks after solver termination. Same initial family and objective as completed200step restart arm. Full folded weight metric; no global recovery, corpus fitting, or behavioral circuit claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='optimization'}),flush=True)
    return result
