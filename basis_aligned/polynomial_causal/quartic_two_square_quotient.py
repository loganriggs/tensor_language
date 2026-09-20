"""Reduce a PSD Gram norm on (1,t,t²) to two squares of quadratic features."""
import numpy as np
from scipy.linalg import solve_triangular


def polynomial_coefficients(gram):
    return np.array([gram[0,0],2*gram[0,1],2*gram[0,2]+gram[1,1],
                     2*gram[1,2],gram[2,2]])


def reduce_gram(gram):
    """Return five coefficients for (a+b*t+c*t²)²+(d*t+e*t²)².

    Floating-point identity checked in coefficient space; no exact rational
    certificate or lower bound on arbitrary arithmetic circuits is implied.
    """
    gram=(np.asarray(gram,dtype=np.float64)+np.asarray(gram,dtype=np.float64).T)/2
    scale=float(np.linalg.norm(gram))
    if scale==0:
        return np.zeros(5),dict(coefficient_error=0.,rank=0,shift=0.)
    g=gram/scale
    eigenvalues=np.linalg.eigvalsh(g)
    assert eigenvalues[0]>=-1e-12
    if eigenvalues[0]<=1e-14:
        candidates=[(0.,g)]
    else:
        null=np.array([[0.,0.,1.],[0.,-2.,0.],[1.,0.,0.]])
        l=np.linalg.cholesky(g)
        left=solve_triangular(l,null,lower=True)
        transformed=solve_triangular(l,left.T,lower=True).T
        mu=np.linalg.eigvalsh((transformed+transformed.T)/2)
        lower=max(-1/x for x in mu if x>0)
        upper=min(-1/x for x in mu if x<0)
        candidates=sorted([(lower,g+lower*null),(upper,g+upper*null)],key=lambda x:abs(x[0]))
    original=polynomial_coefficients(gram)
    best=None
    for shift,boundary in candidates:
        values,vectors=np.linalg.eigh((boundary+boundary.T)/2)
        if values[0]<-1e-10:
            continue
        factor=np.sqrt(np.maximum(values[-2:],0.))[:,None]*vectors[:,-2:].T
        constant=factor[:,0];length=np.linalg.norm(constant)
        if length>0:
            a,b=constant/length
            factor=np.array([[a,b],[-b,a]])@factor
        factor[1,0]=0.
        factor*=np.sqrt(scale)
        coefficients=np.r_[factor[0],factor[1,1:]]
        error=float(np.linalg.norm(polynomial_coefficients(factor.T@factor)-original)/max(np.linalg.norm(original),1e-30))
        receipt=dict(coefficient_error=error,rank=int(np.sum(values>1e-12)),shift=float(shift*scale))
        if best is None or error<best[1]['coefficient_error']:
            best=coefficients,receipt
    assert best is not None and best[1]['coefficient_error']<=1e-10, best
    return best


def evaluate_norm(coefficients,t):
    a,b,c,d,e=np.asarray(coefficients).T
    return (a+b*t+c*t*t)**2+(d*t+e*t*t)**2


def controls():
    rng=np.random.default_rng(21)
    cases=[np.eye(3),np.diag([0.,1.,1.]),np.zeros((3,3)),np.ones((3,3))]
    for rank in [1,2,3]:
        for _ in range(20):
            matrix=rng.normal(size=(rank,3));cases.append(matrix.T@matrix)
    errors=[]
    for gram in cases:
        coefficients,receipt=reduce_gram(gram)
        assert receipt['coefficient_error']<=1e-10
        for t in [-3.,-1.,0.,.25,1.,3.]:
            phi=np.array([1.,t,t*t]);expected=phi@gram@phi
            errors.append(abs(evaluate_norm(coefficients,t)-expected)/max(1.,abs(expected)))
    assert max(errors)<1e-10
    return dict(cases=len(cases),max_norm_error=float(max(errors)))


if __name__=='__main__':
    print(controls())
