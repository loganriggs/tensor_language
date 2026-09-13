"""Exact 19-function self-product basis for two fixed intervention writers."""
import torch

def monomials(degree):return [(i,degree-i) for i in range(degree+1)]
def multiply(p,q):
 out={}
 for (i,j),x in p.items():
  for (k,l),y in q.items():out[i+k,j+l]=out.get((i+k,j+l),0)+x*y
 return out

def coefficient_system(rho):
 a={(1,0):1};b={(0,1):1}
 numerators=[{k:-v for k,v in multiply(a,rho).items()},{k:-v for k,v in multiply(b,rho).items()}, {(1,0):-1},{(0,1):-1},{(2,0):.5},{(1,1):1},{(0,2):.5}]
 powers=[m for n in range(2,7) for m in monomials(n)]
 basis=[multiply({m:1},multiply(rho,rho)) for m in monomials(2)]+[multiply({m:1},rho) for m in monomials(3)]+[{m:1} for n in range(2,5) for m in monomials(n)]
 pairs=[(i,j) for i in range(7) for j in range(i,7)]
 products=[]
 for i,j in pairs:
  product=multiply(numerators[i],numerators[j]);products.append({k:v*(1 if i==j else 2) for k,v in product.items()})
 def matrix(polys):return torch.tensor([[p.get(m,0) for p in polys] for m in powers],dtype=torch.float64)
 return matrix(basis),matrix(products),pairs

def scalar_functions(a,b,rho):
 mon=lambda n:torch.stack([a**i*b**j for i,j in monomials(n)])
 return torch.cat([mon(2),mon(3)/rho,torch.cat([mon(n) for n in range(2,5)])/rho**2])
