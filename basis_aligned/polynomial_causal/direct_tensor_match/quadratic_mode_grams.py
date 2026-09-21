"""Exact mode Gram matrices for symmetric T_vij=sum C_vk sym(A_ki B_kj)."""
import torch

def grams(C,A,B):
 aa=A@A.T;bb=B@B.T;ab=A@B.T;cc=C.T@C
 output=C@(.5*(aa*bb+ab*ab.T))@C.T
 inputs=.25*(A.T@(cc*bb)@A+B.T@(cc*aa)@B+A.T@(cc*ab.T)@B+B.T@(cc*ab)@A)
 return (output+output.T)/2,(inputs+inputs.T)/2

def controls():
 g=torch.Generator().manual_seed(936);rows=[]
 for d,o,k in ((3,2,4),(5,4,7)):
  C=torch.randn(o,k,generator=g,dtype=torch.float64);A=torch.randn(k,d,generator=g,dtype=torch.float64);B=torch.randn(k,d,generator=g,dtype=torch.float64);T=torch.einsum('vk,ki,kj->vij',C,A,B);T=(T+T.transpose(1,2))/2;out,inp=grams(C,A,B);of=T.flatten(1);inf=T.permute(1,0,2).flatten(1)
  err=max(float((out-of@of.T).norm()/(of@of.T).norm()),float((inp-inf@inf.T).norm()/(inf@inf.T).norm()));assert err<1e-12;rows.append(err)
 return rows
