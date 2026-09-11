"""CPU program execution, removals/composition and nonlinear-readout controls.
These establish algebra, not native behavioral sufficiency or selectivity.
"""
import hashlib
import io
import json
from pathlib import Path
import torch
from sparse_reader_program_v1 import SparseReaderProgram
from quadratic_readout_state_v1 import compile_state,evaluate
from calibration_two_readers_v1 import EPS32


def relative(actual,expected):
    return float((actual-expected).norm()/expected.norm().clamp_min(1e-30))


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(710)
    dim,products,outputs,k=12,17,9,3
    basis=torch.linalg.qr(torch.randn(dim,dim)).Q
    indices=torch.stack([torch.randperm(dim)[:k] for _ in range(2*products)])
    values=torch.randn(2*products,k)
    down,bias=torch.randn(outputs,products),torch.randn(outputs)
    artifact=dict(analysis_basis=basis,code_indices=indices.to(torch.int16),code_values=values,
                  retained_down=down,retained_bias=bias)
    buffer=io.BytesIO();torch.save(artifact,buffer);buffer.seek(0)
    loaded=torch.load(buffer,weights_only=True)
    program=SparseReaderProgram.from_artifact(loaded,loaded['retained_down'],loaded['retained_bias'])
    codes=torch.zeros(2*products,dim).scatter_(1,indices,values)
    dense_left,dense_right=(codes@basis).split(products)
    x,donor=torch.randn(2,7,dim),torch.randn(2,7,dim)
    expected=((x@dense_left.T)*(x@dense_right.T))@down.T+bias
    actual=program(x)
    errors=dict(execution=relative(actual,expected))
    first,second=[0,2,4],[1,3,5]
    f0=actual
    fs=program.remove_features(x,first)
    ft=program.remove_features(x,second)
    fst=program.remove_features(x,first+second)
    interaction=program.disjoint_interaction(x,first,second)
    errors['removal_composition']=relative(f0-fs-ft+fst,interaction)
    interaction_fraction=float(interaction.norm()/f0.norm())
    h=program.features(x)
    delta=program.selected(program.features(donor)-h,first)
    y0,b,a=program.response_coefficients(x,delta)
    errors['interchange']=relative(y0+b+a,program.interchange_features(x,donor,first))
    errors['empty_removal']=relative(program.remove_features(x,[]),f0)
    errors['all_removal']=relative(program.remove_features(x,list(range(dim))),bias.expand_as(f0))
    background=torch.randn_like(y0)
    readers=torch.randn(14,5,outputs)
    state=compile_state((background+y0).reshape(14,outputs),b.reshape(14,outputs),a.reshape(14,outputs),readers)
    dose_errors=[]
    for dose in (-.5,0.,.25,1.,2.):
        edited=background+program.forward_features(h+dose*delta)
        direct=30*torch.tanh(torch.einsum('btd,bd->bt',readers,edited.reshape(14,outputs))
            /(30*(edited.reshape(14,outputs).square().mean(-1)+EPS32).sqrt()[:,None]))
        dose_errors.append(dict(dose=dose,max_absolute_error=float((evaluate(state,dose)-direct).abs().max()),
            relative_error=relative(evaluate(state,dose),direct)))
    result=dict(predictions={
        'pred_a_sparse_execution':max(v for k,v in errors.items() if k!='removal_composition')<=1e-10,
        'pred_b_live_interaction_identity':errors['removal_composition']<=1e-10 and interaction_fraction>=.05,
        'pred_c_terminal_dose_readout':max(r['max_absolute_error'] for r in dose_errors)<=1e-10},
        errors=errors,interaction_fraction=interaction_fraction,dose_errors=dose_errors,
        price=program.price(),serialized_artifact_bytes=buffer.getbuffer().nbytes,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('sparse_reader_program_v1.py'),Path(__file__).with_name('quadratic_readout_state_v1.py'))},
        gpu_access=False,corpus_access=False,
        scope='Synthetic exact program algebra and analytic FP64 readout; not native float32 rounding, circuit selectivity, extraction or OOD evidence.')
    with Path(__file__).with_name('SPARSE_READER_PROGRAM_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
