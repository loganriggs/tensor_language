"""Known radial projection applied to three frozen native reader dictionaries."""
import hashlib
import json
from pathlib import Path
import torch
from native_reader_msp_generalization_v1 import P,CK
from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram
from radial_corrected_reader_v1 import RadialCorrectedReader

def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double();dim=l.shape[1]
    meta=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    metric=torch.load(meta['cache']['path'],weights_only=True,map_location='cpu')['unembedding_gram']
    total=meta['native_total'];mean=d@(l*r).sum(1)
    energy=lambda x:float(x@metric@x)
    mean_energy=energy(mean)
    sphere_quad=dim/(dim+2)*(2*total+mean_energy)
    sphere_with_bias=sphere_quad+2*float(mean@metric@bias)+energy(bias)
    harmonic=dim/(dim+2)*2*(total-mean_energy/dim)
    parents=[('orthogonal0','FULL_READER_DICTIONARY_MSP_V1_SEED_0.json'),
        ('orthogonal937','FULL_READER_DICTIONARY_MSP_V1_SEED_937.json'),
        ('ordinary0','OBLIQUE_READER_DICTIONARY_V1_ordinary_covariance_SEED_0.json')]
    rows=[]
    for name,filename in parents:
        parent=json.loads((P/filename).read_text());path=Path(parent['cache']['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==parent['cache']['sha256']
        saved=torch.load(path,weights_only=True,map_location='cpu')
        base=RectangularSparseReaderProgram.from_artifact(saved,d,bias)
        reads=torch.sparse.mm(base.codes,base.analysis_basis);a,b=reads.split(4608)
        fitted_mean=d@(a*b).sum(1);delta=mean-fitted_mean
        program=RadialCorrectedReader(base,delta)
        x=torch.randn(5,dim,generator=torch.Generator().manual_seed(842))
        direct=((x@a.T)*(x@b.T))@d.T+bias+x.square().sum(1,keepdim=True)/dim*delta
        execution=float((program(x)-direct).norm()/direct.norm())
        trace_replay=float((fitted_mean+delta-mean).norm()/mean.norm())
        old_error=(1-parent['scores']['coefficient_capture'])*total
        delta_energy=energy(delta)
        new_error=old_error-delta_energy/dim
        assert new_error>=0
        before_sphere_error=dim/(dim+2)*(2*old_error+delta_energy)
        after_sphere_error=dim/(dim+2)*2*new_error
        cache=Path(f'/dev/shm/bilin18_frozen_reader_radial_v1_{name}.pt');assert not cache.exists()
        torch.save(dict(delta=delta,base=parent['cache'],retained_down_key='transformer.h.17.mlp.Down.weight',
            retained_bias_key='transformer.h.17.mlp.Down_bias'),cache)
        rows.append(dict(name=name,base=parent['cache'],cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
            execution_replay=execution,trace_replay=trace_replay,
            coefficient_capture_before=1-old_error/total,coefficient_capture_after=1-new_error/total,
            coefficient_gain=delta_energy/(dim*total),uniform_sphere_capture_before=1-before_sphere_error/sphere_with_bias,
            uniform_sphere_capture_after=1-after_sphere_error/sphere_with_bias,
            uniform_sphere_capture_gain=(before_sphere_error-after_sphere_error)/sphere_with_bias,
            traceless_coefficient_capture=1-new_error/(total-mean_energy/dim),additional_coefficients=dim))
    toy=json.loads((P/'RADIAL_CORRECTED_READER_V1_CONTROL.json').read_text())
    predictions=dict(pred_a_execution=all(toy['predictions'].values()) and all(max(r['execution_replay'],r['trace_replay'])<=1e-10 for r in rows),
        pred_b_sphere_gain=all(r['uniform_sphere_capture_gain']>=.10 for r in rows),
        pred_c_small_coefficient_gain=all(r['coefficient_gain']<=.01 for r in rows))
    result=dict(predictions=predictions,arms=rows,uniform_sphere_target_energy_with_bias=sphere_with_bias,
        radial_plus_bias_only_uniform_sphere_capture=1-harmonic/sphere_with_bias,
        radial_only_coefficient_capture=mean_energy/(dim*total),
        scope='Exact radial projection of frozen weight fits; ideal sphere reference, no natural-input validation or new dictionary fit')
    (P/'FROZEN_READER_RADIAL_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
