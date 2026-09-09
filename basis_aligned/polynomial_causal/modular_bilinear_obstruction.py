"""Finite-field certificate primitives for exact dyadic quadratic readers."""
import numpy as np

PRIME=65521


def dyadic_mod(values,prime=PRIME):
    """Exact FP32/FP64 binary numbers, with mantissas fitting 53 bits."""
    x=np.asarray(values,dtype=np.float64);assert np.isfinite(x).all()
    mant,exp=np.frexp(x)
    # Native inputs are FP32 dyadics: require this, rather than silently round.
    integers=(mant*2**24).astype(np.int64)
    assert np.array_equal(integers.astype(np.float64),mant*2**24)
    powers=np.array([pow(2,int(e)-24,prime) for e in np.unique(exp)],dtype=np.int64)
    table=dict(zip(np.unique(exp).tolist(),powers.tolist()))
    scale=np.empty_like(exp,dtype=np.int64)
    for e,p in table.items():scale[exp==e]=p
    return ((integers%prime)*scale)%prime


def determinant_mod(matrix,prime=PRIME):
    a=np.asarray(matrix,dtype=np.int64).copy()%prime;n=len(a);assert a.shape==(n,n)
    determinant=1
    for i in range(n):
        pivots=np.flatnonzero(a[i:,i])
        if not len(pivots):return 0
        j=i+int(pivots[0])
        if j!=i:a[[i,j]]=a[[j,i]];determinant=-determinant
        pivot=int(a[i,i]);determinant=determinant*pivot%prime
        multipliers=(a[i+1:,i]*pow(pivot,-1,prime))%prime
        a[i+1:,i+1:]=(a[i+1:,i+1:]-multipliers[:,None]*a[i,i+1:])%prime
        a[i+1:,i]=0
    return int(determinant%prime)


def forms_mod_numpy(c,left,right,down,prime=PRIME):
    c,l,r,d=[dyadic_mod(v,prime) for v in (c,left,right,down)]
    coefficient=(c@d)%prime
    raw=[(l.T@((v[:,None]*r)%prime))%prime for v in coefficient]
    return np.stack([((v+v.T)*pow(2,-1,prime))%prime for v in raw])


def controls():
    from fractions import Fraction
    from itertools import permutations
    rng=np.random.default_rng(60915)
    c=rng.integers(-3,4,(2,3)).astype(np.float32)/8
    left=rng.integers(-3,4,(5,4)).astype(np.float32)/4
    right=rng.integers(-3,4,(5,4)).astype(np.float32)/16
    down=rng.integers(-3,4,(3,5)).astype(np.float32)/2
    q=forms_mod_numpy(c,left,right,down)
    def exact(k,i,j):
        return sum(sum(Fraction(float(c[k,a]))*Fraction(float(down[a,h])) for a in range(3))*
                   (Fraction(float(left[h,i]))*Fraction(float(right[h,j]))+Fraction(float(left[h,j]))*Fraction(float(right[h,i])))/2 for h in range(5))
    direct=np.array([[[int(v.numerator*pow(v.denominator,-1,PRIME)%PRIME) for v in [exact(k,i,j)]][0] for j in range(4)] for k in range(2) for i in range(4)]).reshape(2,4,4)
    comm=(q[0]@q[1]-q[1]@q[0])%PRIME
    det=0
    for perm in permutations(range(4)):
        sign=(-1)**sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))
        term=sign
        for i,j in enumerate(perm):term=term*int(comm[i,j])%PRIME
        det=(det+term)%PRIME
    identity=np.eye(4,dtype=np.int64)
    # Full input support alone misses an exact norm-only representation.
    x=rng.normal(size=(17,4));rho=np.sum(x*x,axis=1)
    checks={'rational_contraction_replay':bool(np.array_equal(q,direct)),
            'determinant_permutation_replay':determinant_mod(comm)==det,
            'invertible_obstruction':determinant_mod(np.array([[0,2],[-2,0]]))==4,
            'singular_detected':determinant_mod(np.zeros((3,3),dtype=np.int64))==0,
            'identity_full_rank_norm_only_counterexample':np.linalg.matrix_rank(identity)==4 and np.max(abs(np.einsum('bi,ij,bj->b',x,identity,x)-rho))<1e-12,
            'row_swap_sign':determinant_mod(np.array([[0,1],[1,0]]))==PRIME-1}
    checks={k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks,'toy_commutator_determinant':det}
