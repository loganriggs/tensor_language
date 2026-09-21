"""Exact Gaussian Hermite energies of exported CP programs (not teacher errors)."""
import json
from pathlib import Path
import torch
from quartic_cp import cp_gram
from gaussian_cp import gaussian_cp_gram
from native_quartic_gaussian_projection import cp_mean
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);rows=[]
 for seed in [1001,1002]:
  p=torch.load(P/f'QUARTIC_CP512_SEED{seed}_V2.pt',weights_only=True,map_location='cpu');f=[a.double() for a in p['factors']];c=p['coefficients'].double()/19054614563.464127
  g4=cp_gram(f,f);g=gaussian_cp_gram(f,f);m=cp_mean(f)
  energy=float(((c.T@c)*g).sum());four=24*float(((c.T@c)*g4).sum());zero=float((c@m).square().sum());two=energy-four-zero
  assert min(zero,two,four)>-1e-10*energy
  rows.append(dict(seed=seed,total_gaussian_energy=energy,degree0_energy=zero,degree2_energy=two,degree4_energy=four,fractions=dict(degree0=zero/energy,degree2=two/energy,degree4=four/energy)))
 result=dict(rows=rows,scope='Exact Gaussian chaos energies of candidate programs only. No native target error inferred. Same16output scalar coordinates and physical coefficients divided by common teacher scale.')
 (P/'CP512_GAUSSIAN_COMPONENTS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
