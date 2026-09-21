"""Coefficient-space geometry of saved quartic CP programs; CPU only."""
import argparse,json
from pathlib import Path
import torch
from quartic_cp import cp_gram
P=Path(__file__).resolve().parent

def main(version):
 torch.set_num_threads(2);models=[];rows=[]
 for seed in [1001,1002]:
  p=torch.load(P/f'QUARTIC_CP512_SEED{seed}_{version}.pt',weights_only=True,map_location='cpu')
  f=[a.double() for a in p['factors']];c=p['coefficients'].double()/19054614563.464127
  # Direction count is architectural; these spectral summaries are descriptive.
  reader=torch.cat(f);e=torch.linalg.eigvalsh(reader.T@reader).clamp_min(0).flip(0)
  weight=c.norm(dim=0);weighted=torch.cat([a*weight.sqrt()[:,None] for a in f]);we=torch.linalg.eigvalsh(weighted.T@weighted).clamp_min(0).flip(0)
  gram=cp_gram(f,f);diagonal=gram.diag().sqrt();correlation=gram/(diagonal[:,None]*diagonal[None,:]);correlation.fill_diagonal_(0)
  coefficient_norm_squared=float(((c.T@c)*gram).sum())
  atom_energy=(c.square().sum(0)*gram.diag()).sort(descending=True).values
  def stats(e):return dict(numerical_rank=int((e>e[0]*1e-10).sum()),top256_fraction=float(e[:256].sum()/e.sum()),directions_for90percent=int(torch.searchsorted(e.cumsum(0),.9*e.sum()))+1)
  rows.append(dict(seed=seed,reader_spectrum=stats(e),coefficient_weighted_reader_spectrum=stats(we),max_absolute_atom_correlation=float(correlation.abs().max()),atom_pairs_above_099=int((correlation.abs()>.99).sum())//2,coefficient_norm_squared=coefficient_norm_squared,top32_atom_diagonal_energy_fraction=float(atom_energy[:32].sum()/atom_energy.sum()),scope='Direction spectra and individual-atom energies are descriptive and gauge-dependent; diagonal atom energy excludes cross-term cancellation. No semantic or target-capacity certificate.'))
  models.append((f,c,coefficient_norm_squared))
 a,b=models;cross=float(((a[1].T@b[1])*cp_gram(a[0],b[0])).sum());distance=(max(0,a[2]+b[2]-2*cross))**.5
 result=dict(version=version,rows=rows,cross_restart_coefficient_cosine=cross/(a[2]*b[2])**.5,cross_restart_distance_over_first_norm=distance/a[2]**.5,scope='Exact symmetric quartic coefficient comparisons between saved programs, not error against teacher. FP32 artifacts evaluated in FP64.')
 (P/f'QUARTIC_CP512_GEOMETRY_{version}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--version',default='V1');main(parser.parse_args().version)
