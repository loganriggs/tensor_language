"""Saved graph replay, generated-interface comparison and strengthened baseline pricing."""
import json,copy,hashlib
from pathlib import Path
import torch
from pairwise_reader_graph import expand,price
from local_shared_reader_graph import decode
from pairwise_component_interface import component_scalars
from audit_pairwise_execution import cast
from read_error_terms import baseline_reads
from generated_residual_interface import components
from quadratic_ball_extrema import extrema
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);path=P/'ROBUST_DIRECTION_DIRECTED_PROGRAM_V1.pt';graph=torch.load(path,weights_only=True);d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=d['indices'];z,h=d['z'][ids],d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();A=torch.stack([p['a'] for p in d['pairs']],-1);alpha=torch.stack([p['alpha'] for p in d['pairs']]);beta=torch.stack([p['beta'] for p in d['pairs']]);Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]);true_reads=torch.einsum('ni,oij,nj->no',z,Q,z);carry=h@A-true_reads[...,::2];truth=components(true_reads,carry,s,alpha,beta);denom=(truth-truth.mean(0)).norm(dim=0)
 from pairwise_reader_graph import source_reads
 values=component_scalars(z,h,graph);drift=(component_scalars(z.float(),h.float(),cast(graph)).double()-values).norm(dim=0)/denom;assert float(drift.max())<1e-4;generated=components(source_reads(z,graph),carry,s,alpha,beta);generated_error=(generated-truth).norm(dim=0)/denom
 old=torch.load(P/'FRONTIER_PAIR_BASELINES_V1.pt',weights_only=True);new=torch.load(P/'ROBUST_WIDTH324_BASELINES_V1.pt',weights_only=True);baselines={};comparisons=[]
 for geometry in ('native_isotropic','calibration_shaped'):
  for low in range(3):
   bundle={str(j):copy.deepcopy((old[geometry+'_323'] if j==low else new[geometry+'_324'])[str(j)]) for j in range(3)};writer=bundle['0']['residual_writer']
   for pair in bundle.values():pair['residual_writer']=writer
   key=geometry+'_low'+str(low);baselines[key]=bundle;unique={v.data_ptr():v for pair in bundle.values() for v in pair.values() if v.is_floating_point()};floats=sum(v.numel() for v in unique.values());mults=sum(pair['shared_reader'].numel()+len(pair['product_weights'])+pair['product_weights'].numel() for pair in bundle.values());assert floats<=price(graph)['stored_floats'] and mults<=price(graph)['source_total_multiplications']
   reads=torch.cat([baseline_reads(z,bundle[str(j)]) for j in range(3)],-1);error=(components(reads,carry,s,alpha,beta)-truth).norm(dim=0)/denom;pair=bundle['2'];ext=extrema(decode(pair)[1]-Q[5],pair['b_linear'],pair['b_bias'],1152**.5)
   comparisons.append(dict(key=key,stored_floats=floats,source_multiplications=mults,generated_errors=error.tolist(),candidate_generated_pass=(generated_error<=1.1*error).tolist(),read3b_worst=ext['worst_absolute']))
 bundle=expand(graph);pair=bundle['2'];worst=extrema(decode(pair)[1]-Q[5],pair['b_linear'],pair['b_bias'],1152**.5)
 for row in comparisons:row['candidate_worst_ratio']=worst['worst_absolute']/row['read3b_worst'];row['candidate_worst_pass']=row['candidate_worst_ratio']<=1.1
 out=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),price=price(graph),fp32_component_drift=drift.tolist(),generated_errors=generated_error.tolist(),worst_read_error=worst['worst_absolute'],baselines=comparisons,predictions=dict(generated_interface=all(all(r['candidate_generated_pass']) for r in comparisons),strengthened_worst_read=all(r['candidate_worst_pass'] for r in comparisons)),scope='Opened448states; six baselines include all placements of widths323/324/324, tied common writer, both coefficient geometries. No text transfer orsemanticidentity claim.')
 torch.save(baselines,P/'ROBUST_MATCHED_BASELINES_V1.pt');(P/'ROBUST_DIRECTION_EXPORT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
