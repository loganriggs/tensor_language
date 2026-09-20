"""Exact symmetric quartic CP inner products and native teacher directional contractions."""
import itertools

def directional(C,L2,R2,D1,L1,R1,vectors):
 def pair(u,v):return .5*((u@L1.T)*(v@R1.T)+(v@L1.T)*(u@R1.T))@D1.T
 def outer(s,t):return .5*((s@L2.T)*(t@R2.T)+(t@L2.T)*(s@R2.T))@C.T
 a,b,c,d=vectors
 return (outer(pair(a,b),pair(c,d))+outer(pair(a,c),pair(b,d))+outer(pair(a,d),pair(b,c)))/3

def cp_gram(factors,other):
 matrices=[[a@b.T for b in other] for a in factors];gram=matrices[0][0].new_zeros(matrices[0][0].shape)
 for p in itertools.permutations(range(4)):
  term=matrices[0][p[0]]
  for i in range(1,4):term=term*matrices[i][p[i]]
  gram=gram+term
 return gram/24

def cp_inner(C,factors,D,other):return ((C.T@D)*cp_gram(factors,other)).sum()

def teacher_cross(teacher,C,factors):return (directional(*teacher,factors)*C.T).sum()

def cp_entries(C,factors,indices):
 values=[a[:,indices].permute(1,0,2) for a in factors]
 features=values[0][...,0].new_zeros(values[0].shape[:2])
 for p in itertools.permutations(range(4)):
  term=values[0][...,p[0]]
  for i in range(1,4):term=term*values[i][...,p[i]]
  features=features+term
 return (features/24)@C.T
