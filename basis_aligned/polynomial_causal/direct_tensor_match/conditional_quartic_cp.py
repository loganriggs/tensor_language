"""Exact Gaussian conditional mean of an affine quartic CP program.
Each atom reuses its six pair products; the quartic adds one multiplication.
"""
import itertools
import torch
PAIRS=list(itertools.combinations(range(4),2))

def construct(factors,C,S,mu,V):
    fw=[f@S for f in factors]
    retained=[f@V for f in fw]
    discarded=[f-a@V.T for f,a in zip(fw,retained)]
    cov={pair:(discarded[pair[0]]*discarded[pair[1]]).sum(1) for pair in PAIRS}
    read=torch.linalg.solve(S.T,V).T
    offset=read@mu
    bias=[f@mu-a@offset for f,a in zip(factors,retained)]
    # Pair ij is multiplied by covariance of the complementary two slots.
    weights=torch.stack([cov[tuple(i for i in range(4) if i not in pair)] for pair in PAIRS])
    constant=C@(cov[0,1]*cov[2,3]+cov[0,2]*cov[1,3]+cov[0,3]*cov[1,2])
    return dict(input_projection=read,factors=retained,biases=bias,pair_weights=weights,coefficients=C,constant=constant)

def evaluate(program,x,corrections=True):
    z=x@program['input_projection'].T
    a=[z@f.T+b for f,b in zip(program['factors'],program['biases'])]
    products=[a[i]*a[j] for i,j in PAIRS]
    phi=products[0]*products[-1]
    if corrections:
        phi=phi+sum(p*w for p,w in zip(products,program['pair_weights']))
    y=phi@program['coefficients'].T
    return y+program['constant'] if corrections else y

def price(program):
    d=program['input_projection'].shape[1];r=program['input_projection'].shape[0];k=program['factors'][0].shape[0];v=program['coefficients'].shape[0]
    return dict(variable_products=7*k,stored_floats=d*r+4*k*r+4*k+6*k+v*k+v,
                scalar_coefficient_multiplications=d*r+4*k*r+6*k+v*k,
                additions=r*(d-1)+4*k*r+6*k+v*k,
                input_rank=r,quartic_atoms=k,
                convention='Dense generic nonzero coefficients; projection then2048affineforms; six shared pairs per atom and onequarticproduct; sixweightedcorrections; outputconstant. Finalfixedvocabularywriter excluded equally.')
