"""Implicit exact loss for a shared bilinear-product dictionary over every token.
No per-token interaction tensor is required. Numerical optimization is separate.
"""
import torch

def product_cross(l,r,a,b):
    return ((l@a.T)*(r@b.T)+(l@b.T)*(r@a.T))/2

def optimal_writers(d,l,r,a,b,rtol=1e-10):
    cross=product_cross(l,r,a,b);gram=product_cross(a,b,a,b)
    return d@cross@torch.linalg.pinv(gram,rtol=rtol),cross,gram

def implicit_squared_error(u_gram,d,native_gram,w,cross,candidate_gram):
    target=((d.T@u_gram@d)*native_gram).sum()
    approximation=((w.T@u_gram@w)*candidate_gram).sum()
    overlap=((d.T@u_gram@w)*cross).sum()
    return target+approximation-2*overlap
