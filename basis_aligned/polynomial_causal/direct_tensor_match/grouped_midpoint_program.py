"""Compile contiguous output-shared product groups into an executable residual program.
No model-specific files or global state; compilation is an offline exact rewrite.
"""
import torch

def compile_program(program, reduced_output_map):
    a,b=program['A'],program['B']
    outputs=program['offset'].numel()
    products=a.shape[1]
    if products % outputs:
        raise ValueError('Unequal product groups require an explicit sparse grouping map')
    rank=products//outputs
    expected=torch.zeros_like(program['readout'])
    expected[torch.arange(products),torch.arange(products)//rank]=1
    if not torch.equal(expected,program['readout']):
        raise ValueError('Only contiguous unit-weight product groups are supported')
    writer=torch.linalg.solve(reduced_output_map,program['reduced_writers'])
    return dict(A=a,B=b,writer=writer,offset=program['offset'],rank=rank)

def scalar_features(compiled,n,m):
    products=(n@compiled['A'])*(m@compiled['B'])
    return products.unflatten(-1,(-1,compiled['rank'])).sum(-1)-compiled['offset']

def residual_value(compiled,n,m):
    return scalar_features(compiled,n,m)@compiled['writer'].T

def toy_check():
    gen=torch.Generator().manual_seed(261201)
    rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    outputs,rank,d=3,4,7;products=outputs*rank
    readout=torch.zeros(products,outputs,dtype=torch.float64)
    readout[torch.arange(products),torch.arange(products)//rank]=1
    program=dict(A=rand(d,products),B=rand(d,products),readout=readout,offset=rand(outputs),reduced_writers=rand(d,outputs))
    transform=rand(d,d)+5*torch.eye(d,dtype=torch.float64)
    n,m=rand(2,5,d),rand(2,5,d)
    original=(((n@program['A'])*(m@program['B']))@readout-program['offset'])@program['reduced_writers'].T
    original=torch.linalg.solve(transform,original.unsqueeze(-1)).squeeze(-1)
    compact=compile_program(program,transform);actual=residual_value(compact,n,m)
    error=float((actual-original).norm()/original.norm())
    assert error<1e-12
    bad=dict(program);bad['readout']=readout.clone();bad['readout'][0,0]=2
    rejected=False
    try:compile_program(bad,transform)
    except ValueError:rejected=True
    assert rejected
    return dict(residual_replay=error,unsupported_grouping_rejected=rejected,scope='Exact grouped summation and offline residual writer contraction; arbitrary batch dimensions. No approximation or speed measurement.')

if __name__=='__main__':
    import json
    from pathlib import Path
    result=toy_check();out=Path(__file__).with_name('GROUPED_MIDPOINT_ORACLE_V1.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
