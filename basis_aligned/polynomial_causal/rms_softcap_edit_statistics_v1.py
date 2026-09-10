"""Exact finite source-edit decoding with a shared quadratic normalization state.

Inputs are fixed native states and writes, not independent semantic producers.
Preserve the native epsilon explicitly when auditing in a wider precision.
"""
import numpy as np


def compile_statistics(base, sources, readers, *, eps, cap=30.):
    x=np.asarray(base,dtype=float);d=np.asarray(sources,dtype=float);w=np.asarray(readers,dtype=float)
    if x.ndim!=1 or d.ndim!=2 or w.ndim!=2 or d.shape[1]!=len(x) or w.shape[1]!=len(x):
        raise ValueError('base[D], sources[K,D], readers[R,D] required')
    if not all(np.isfinite(a).all() for a in (x,d,w)) or eps<=0 or cap<=0:
        raise ValueError('finite values and positive epsilon/cap required')
    return {'dimension':len(x),'eps':float(eps),'cap':float(cap),
            'base_norm2':float(x@x),'base_source_dots':d@x,'source_gram':d@d.T,
            'base_reader':w@x,'source_readers':d@w.T}


def decode(stats, coefficients):
    a=np.asarray(coefficients,dtype=float)
    if a.shape!=np.asarray(stats['base_source_dots']).shape or not np.isfinite(a).all():
        raise ValueError('one finite coefficient per source required')
    linear=2*float(a@stats['base_source_dots']);quadratic=float(a@stats['source_gram']@a)
    square=stats['base_norm2']+linear+quadratic
    scale=max(1.,abs(stats['base_norm2'])+abs(linear)+abs(quadratic))
    if square < -1e-12*scale:raise ValueError('inconsistent negative squared norm')
    denom=np.sqrt(max(0.,square)/stats['dimension']+stats['eps'])
    numerator=stats['base_reader']+a@stats['source_readers']
    return stats['cap']*np.tanh(numerator/(stats['cap']*denom))


def controls():
    rng=np.random.default_rng(910543);x=rng.normal(size=7);d=rng.normal(size=(4,7))
    w=rng.normal(size=(3,7));w[2]=w[0]-w[1]  # overlapping/dependent raw readers
    eps=float(np.finfo(np.float32).eps)
    s=compile_statistics(x,d,w,eps=eps);errors=[]
    for a in (np.zeros(4),np.array([-1.,0,0,0]),np.array([-1.,-1.,0,0]),rng.normal(size=4)):
        y=x+a@d;expected=30*np.tanh((w@y)/(30*np.sqrt(y@y/7+eps)))
        errors.append(float(np.max(abs(decode(s,a)-expected))))
    # Orthogonal residual-coordinate changes preserve both output and statistics.
    q,_=np.linalg.qr(rng.normal(size=(7,7)))
    rotated=compile_statistics(x@q,d@q,w@q,eps=eps);a=np.array([-.2,.4,.7,-1.])
    gauge=float(np.max(abs(decode(s,a)-decode(rotated,a))))
    zero=compile_statistics(np.array([1.,2.]),np.array([[1.,2.]]),np.eye(2),eps=eps)
    assert np.array_equal(decode(zero,[-1.]),[0.,0.])
    # Dropping source cross-terms gives the wrong norm for overlapping sources.
    overlap=compile_statistics(np.array([1.,0.]),np.array([[1.,1.],[1.,-1.],[1.,1.]]),np.eye(2),eps=eps)
    bad=dict(overlap);bad['source_gram']=np.diag(np.diag(overlap['source_gram']))
    dose=np.array([.5,0.,.5]);missing_cross_error=float(np.max(abs(decode(overlap,dose)-decode(bad,dose))))
    assert max(errors+[gauge])<1e-12 and missing_cross_error>.01
    # Tiny residual direction can dominate a reader despite small vector norm.
    large=np.array([0.,100.]);small=np.array([1.,0.]);target=large+small
    vector_fraction=float(large@target/(target@target))
    reader=np.array([[1.,0.]])
    toy=compile_statistics(target,np.stack([large,small]),reader,eps=eps)
    native=float(decode(toy,[0.,0.])[0]);without_large=float(decode(toy,[-1.,0.])[0])
    without_small=float(decode(toy,[0.,-1.])[0])
    assert vector_fraction>.999 and without_small==0 and without_large>native
    return {'passed':True,'direct_decoder_max_abs':max(errors),'orthogonal_coordinate_error':gauge,
        'zero_state_epsilon_passed':True,'omitted_source_cross_term_error':missing_cross_error,
        'reader_vs_vector_counterexample':{'large_source_vector_projection':vector_fraction,
            'native_reader':native,'without_large':without_large,'without_small':without_small},
        'scope':'Exact decoder algebra on fixed state/write interfaces; no discovered circuit or model compression.'}
