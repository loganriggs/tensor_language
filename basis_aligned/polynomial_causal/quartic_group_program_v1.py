"""Frozen shared-pair, quadratic-intermediate, signed-square executable.
Input is actual normalized MLP16 input. MLP17 input RMS squared is supplied externally.
Native bias/remaining paths/final tail are not included. Internal removals hold denominator fixed.
"""
import torch


def run(program,x,denominator,zero_pairs=(),zero_quadratics=()):
    reads=x@program['bank'];pair=program['pairs'];products=reads[...,pair[0]]*reads[...,pair[1]]
    if zero_pairs:
        products=products.clone();products[...,list(zero_pairs)]=0
    quadratics=products@program['quadratic_readers']
    if zero_quadratics:
        quadratics=quadratics.clone();quadratics[...,list(zero_quadratics)]=0
    terms=quadratics.square()*program['signed_weights'];scalar=terms.reshape(*terms.shape[:-1],2,8).sum(-1)
    numerator=scalar@program['output_writers'].T
    assert bool((denominator>0).all()) and denominator.shape==x.shape[:-1]
    return dict(reads=reads,products=products,quadratics=quadratics,scalar=scalar,numerator=numerator,write=numerator/denominator[...,None])
