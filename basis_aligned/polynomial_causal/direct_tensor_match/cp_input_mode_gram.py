"""Exact first-input unfolding Gram of the symmetric quartic coefficient tensor."""
import itertools
import torch


def gram(factors,C):
    dots={(s,t):factors[s]@factors[t].T for s in range(4) for t in range(4)}
    output=C.T@C;K=C.new_zeros(factors[0].shape[1],factors[0].shape[1])
    for s in range(4):
        remaining_s=[i for i in range(4) if i!=s]
        for t in range(4):
            remaining_t=[i for i in range(4) if i!=t];g=0.
            for perm in itertools.permutations(remaining_t):
                term=1.
                for i,j in zip(remaining_s,perm):term=term*dots[i,j]
                g=g+term
            K=K+factors[s].T@(output*g/96)@factors[t]
    return (K+K.T)/2
