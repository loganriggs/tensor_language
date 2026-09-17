#!/usr/bin/env python3
# BQGATE:832bodyforwards;8prefixes;300seconds;no fitting.
"""pred_a exact state/anchor controls; pred_b joint from singles<=.10.
pred_c pairwise interaction<=.25minimumsingle; pred_d <=.5randomsplitmedian.
Opened-panel physical-write screen; prior two-input composition failure preserved.
"""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import typed_face_write_atoms_v1 as atoms
from regional_paired_write_runtime_v2 import measure
from regional_endpoint_batching_v1 import group_rows,expand
from run_even_value_factorial_native_v1 import setup
STEM='TYPED_FACE_WRITE_COMPOSITION_V1';D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
ARMS=['native','full']+[f'{split}:{mask}' for split in range(17) for mask in range(1,7)]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
    torch.set_num_threads(2);torch.manual_seed(17092130)
    total=torch.randn(1,32,128,dtype=torch.float64)
    cpu_error=float((atoms.split(total,atoms.random_basis(17092130)).sum(0)-total).norm()/total.norm());assert cpu_error<=1e-12
    assert len(ARMS)==104 and len(groups)==8
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('832bodyforwards;8prefixes;17three-writepartitions;CPU split closure',cpu_error);return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    start=time.perf_counter();signal.alarm(300);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda')
    program={k:v.to('cuda') for k,v in torch.load(D/'program.pt',weights_only=True).items()}
    bases=[atoms.random_basis(17092130+i).to('cuda') for i in range(16)]
    cache={};closure=[];native_errors=[]
    def write(arm,row,donor_row,current,donor,mask):
        key=(row['context_id'],row['cue']);city=row['city_position']
        if key not in cache:
            args=(program,current,donor[:,city],row['ids'][city],donor_row['ids'][city],city)
            channel_parts=atoms.channels(*args);total=channel_parts.sum(0)
            canonical=atoms.physical_writes(channel_parts,program,mask)
            randoms=[atoms.physical_writes(atoms.split(total,basis),program,mask) for basis in bases]
            reference=atoms.native.execute(*args,mask)
            native_errors.append(float((canonical.sum(0)-reference).norm()/reference.norm().clamp_min(1e-8)))
            closure.extend(float((parts.sum(0)-canonical.sum(0)).norm()/canonical.sum(0).norm().clamp_min(1e-8)) for parts in randoms)
            cache[key]=[canonical]+randoms
        if arm=='full':return cache[key][0].sum(0).to(current.dtype)
        split_id,bitmask=map(int,arm.split(':'))
        pieces=cache[key][split_id]
        return sum(pieces[j] for j in range(3) if bitmask&(1<<j)).to(current.dtype)
    measured=measure(model,graph,groups,ARMS,write);v=expand(measured['values'],mapping,6)
    parent=torch.load(P/'REGIONAL_ENDPOINT_BATCHING_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    anchor=v[:2]-parent[[0,6]];anchor_abs=float(anchor.abs().max());anchor_rel=float(anchor.norm()/parent[[0,6]].norm())
    effects=v-v[:1];target=effects[1,:,0]
    cells={cue:[i for i,r in enumerate(rows) if r['cue']==cue] for cue in ['British','American']}
    cells.update({f'endpoint_{e}':[i for i,r in enumerate(rows) if r['endpoint']==e] for e in range(6)})
    reports=[]
    for split_id in range(17):
        corner={0:effects[0,:,0],7:target}
        corner.update({mask:effects[ARMS.index(f'{split_id}:{mask}'),:,0] for mask in range(1,7)})
        per_cell={}
        for name,ids in cells.items():
            singles=[corner[m][ids] for m in [1,2,4]]
            pair_ratios={}
            for a,b in [(1,2),(1,4),(2,4)]:
                interaction=corner[a|b][ids]-corner[a][ids]-corner[b][ids]
                denominator=min(float(corner[a][ids].norm()),float(corner[b][ids].norm()))
                pair_ratios[f'{a}+{b}']=float(interaction.norm())/max(denominator,1e-30)
            triple=corner[7][ids]-corner[3][ids]-corner[5][ids]-corner[6][ids]+sum(singles)
            per_cell[name]={'joint_from_singles_error':float((sum(singles)-target[ids]).norm()/target[ids].norm().clamp_min(1e-8)),
                'pair_interaction_ratios':pair_ratios,'third_order_over_joint':float(triple.norm()/target[ids].norm().clamp_min(1e-8))}
        reports.append({'split':split_id,'cells':per_cell,'worst_pair_ratio':max(max(c['pair_interaction_ratios'].values()) for c in per_cell.values()),
                        'singleton_target_rms':[float(corner[m].square().mean().sqrt()) for m in [1,2,4]]})
    canonical=reports[0];null_stats=torch.tensor([r['worst_pair_ratio'] for r in reports[1:]],dtype=torch.float64)
    null_median=float(null_stats.quantile(.5));specificity=canonical['worst_pair_ratio']/max(null_median,1e-30)
    pred_a=cpu_error<=1e-12 and max(closure)<=1e-5 and max(native_errors)<=1e-5 and anchor_abs<=1e-5 and anchor_rel<=1e-6 and min(canonical['singleton_target_rms'])>=1e-5
    pred_b=all(c['joint_from_singles_error']<=.1 for c in canonical['cells'].values())
    pred_c=canonical['worst_pair_ratio']<=.25
    pred_d=specificity<=.5
    result={'pred_a':pred_a,'pred_b':pred_b,'pred_c':pred_c,'pred_d':pred_d,'canonical':canonical,'random_splits':reports[1:],
       'canonical_over_random_median':specificity,'random_worst_pair_median':null_median,'anchor_max_abs':anchor_abs,'anchor_relative':anchor_rel,
       'max_native_write_error':max(native_errors),'max_split_write_closure':max(closure),'cpu_random_split_closure':cpu_error,
       'body_forwards':measured['body_forwards'],'seconds':time.perf_counter()-start,'panel_status':'opened four-context physical-write screen',
       'source_shas':binding,'scope':'Three physical writes including cross term; tests a different explicit object from the preserved failed two-input-port additive hypothesis. No fresh promotion or whole-model extraction.'}
    torch.save({'values':v,'arms':ARMS},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_shas','random_splits']},indent=2));signal.alarm(0)
if __name__=='__main__':main()
