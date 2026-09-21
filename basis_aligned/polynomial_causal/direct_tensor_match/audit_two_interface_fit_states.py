"""Assess original fitting states at both declared interfaces, without refitting."""
import torch,json
from pathlib import Path
from pairwise_reader_graph import source_reads
from read_error_terms import baseline_reads
from generated_residual_interface import components
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=d['indices'];z,h=d['z'][ids],d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();A=torch.stack([p['a'] for p in d['pairs']],-1);alpha=torch.stack([p['alpha'] for p in d['pairs']]);beta=torch.stack([p['beta'] for p in d['pairs']]);Qs=torch.stack([q for p in d['pairs'] for q in p['Qs']]);reads=torch.einsum('ni,oij,nj->no',z,Qs,z);carry=h@A-reads[...,::2];truth=components(reads,carry,s,alpha,beta);denom=(truth-truth.mean(0)).norm(dim=0);rows=[]
 for name,file in [('graph','FRONTIER_FRESH_GRAPH_V1.pt'),('separate','FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt'),('isotropic','FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt')]:
  p=torch.load(P/file,weights_only=True);q=source_reads(z,p) if name=='graph' else torch.cat([baseline_reads(z,p[str(j)]) for j in range(3)],-1)
  boundary=((h@A-.5*q[...,::2])/s[:,None]-alpha)*(q[...,1::2]/s[:,None]-beta);generated=components(q,carry,s,alpha,beta)
  rows.append(dict(candidate=name,boundary_error=((boundary-truth).norm(dim=0)/denom).tolist(),generated_error=((generated-truth).norm(dim=0)/denom).tolist()))
 out=dict(rows=rows,n=448,scope='No fitting. Uses exact quadratic forms and recomputed RMS17 for internally consistent truth. The generated interface holds supplied carry projections fixed; it is not the total derivative through native attention or RMS producers.')
 (P/'TWO_INTERFACE_FIT_STATES_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
