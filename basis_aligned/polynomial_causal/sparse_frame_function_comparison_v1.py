import torch
from joint_quadratic_fit_v1 import product_cross

def compare(q1,e1,w1,q2,e2,w2,metric):
    i,j=e1;k,l=e2
    scale1=torch.where(i==j,torch.ones_like(i,dtype=q1.dtype),torch.full_like(i,2**.5,dtype=q1.dtype))
    scale2=torch.where(k==l,torch.ones_like(k,dtype=q2.dtype),torch.full_like(k,2**.5,dtype=q2.dtype))
    gram=product_cross(q1[:,i].T,q1[:,j].T,q2[:,k].T,q2[:,l].T)*scale1[:,None]*scale2[None,:]
    inner=((w1.T@metric@w2)*gram).sum()
    norm1=((metric@w1)*w1).sum();norm2=((metric@w2)*w2).sum()
    return dict(cosine=float(inner/(norm1*norm2).sqrt()),relative_squared_difference_to_second=float((norm1+norm2-2*inner)/norm2),norm_ratio=float((norm1/norm2).sqrt()))
