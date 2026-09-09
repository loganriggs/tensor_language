"""Exact joint edit observables for symmetric normalized quadratic readers.

The initialization still needs native forms and input. This is an edit-response
engine, not an independently extracted circuit or a learned-weight reduction.
"""
import torch


def compile_edits(forms,writers):
    return {'forms':forms,'writers':writers,
            'hessians':torch.einsum('ia,kij,jb->kab',writers,forms,writers),
            'writer_gram':writers.T@writers}


def encode(x,program):
    q,w=program['forms'],program['writers']
    return {'p':torch.einsum('bi,kij,bj->bk',x,q,x),
            'g':torch.einsum('ia,kij,bj->bka',w,q,x),
            'h':x@w,'rho':x.square().sum(-1)}


def edit(state,z,program):
    hh=program['hessians'];gg=program['writer_gram']
    return {'p':state['p']+2*torch.einsum('bka,ba->bk',state['g'],z)+torch.einsum('ba,kac,bc->bk',z,hh,z),
            'g':state['g']+torch.einsum('kac,bc->bka',hh,z),
            'h':state['h']+z@gg,
            'rho':state['rho']+2*(state['h']*z).sum(-1)+torch.einsum('ba,ac,bc->b',z,gg,z)}


def evaluate(state,width,epsilon,bias):
    return state['p']/(state['rho'][:,None]/width+epsilon)+bias


def controls():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60914)
        width,k,r=96,8,2
        q=torch.randn(k,width,width,dtype=torch.float64);q=(q+q.transpose(-1,-2))/2
        w=torch.randn(width,r,dtype=torch.float64)
        x=torch.randn(17,width,dtype=torch.float64);a=torch.randn(17,r,dtype=torch.float64);b=torch.randn(17,r,dtype=torch.float64)
        bias=torch.randn(k,dtype=torch.float64)
    eps=torch.finfo(torch.float32).eps;program=compile_edits(q,w);state=encode(x,program)
    actual=edit(state,a,program);expected=encode(x+a@w.T,program)
    errors={key:float((actual[key]-expected[key]).abs().max()) for key in state}
    sequential=edit(actual,b,program);joint=edit(state,a+b,program)
    roundtrip=edit(actual,-a,program)
    output_error=float((evaluate(actual,width,eps,bias)-evaluate(expected,width,eps,bias)).abs().max())
    base_only=dict(state,p=state['p']+2*torch.einsum('bka,ba->bk',state['g'],a))
    missing_cross=dict(program,hessians=program['hessians']*torch.eye(r,dtype=q.dtype)[None])
    bad=edit(state,a,missing_cross)
    checks={'full_state_closure':max(errors.values())<1e-8,'full_output_closure':output_error<1e-8,
            'independent_then_joint':max(float((sequential[key]-joint[key]).abs().max()) for key in state)<1e-8,
            'edit_inverse':max(float((roundtrip[key]-state[key]).abs().max()) for key in state)<1e-8,
            'missing_mixed_writer_negative':float((bad['p']-expected['p']).abs().max())>1,
            'linear_frozen_norm_negative':float((evaluate(base_only,width,eps,bias)-evaluate(expected,width,eps,bias)).abs().max())>1,
            'full_rank_forms':all(int(torch.linalg.matrix_rank(v))==width for v in q)}
    return {'passed':all(checks.values()),'checks':checks,'state_errors':errors,'output_error':output_error,
            'input_width':width,'readers':k,'writer_amplitudes':r,'state_scalars_per_input':k+k*r+r+1,
            'scope':'Random full-rank module; exact specified additive-edit family. Initialization retains full forms and native input; no trained structural savings.'}
