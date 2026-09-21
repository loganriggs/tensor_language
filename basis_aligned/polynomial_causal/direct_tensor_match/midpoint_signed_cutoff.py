from pathlib import Path
import torch,json
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);torch.set_grad_enabled(False);r=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=r['n'].double();m=r['m'].double();K=torch.load(p/'MIDPOINT_FACTOR_PROGRAMS_V1.pt',weights_only=True)['K'].double();parts=json.loads((p/'MIDPOINT_STABILITY_V1.json').read_text())['partitions'];sets={'full':list(range(32))}
for part in parts:
 sets[part['partition']+'_first']=part['first_documents'];sets[part['partition']+'_second']=part['second_documents']
records=[]
for name,ids in sets.items():
 nn=n[ids].flatten(0,1);mm=m[ids].flatten(0,1);a=nn.T@nn/len(nn);b=mm.T@mm/len(mm);M=(a/a.trace()+b/b.trace())/2;e,V=torch.linalg.eigh(M+1e-6*M.trace()/1152*torch.eye(1152));S=(V*e.sqrt()[None,:])@V.T
 for g,k in enumerate(K):
  z=S@k@S;vals=torch.linalg.eigvalsh((z+z.T)/2);v=vals[vals.abs().argsort(descending=True)];records.append(dict(panel=name,feature=g,top6_signed_eigenvalues=v[:6].tolist(),positive_among4=int((v[:4]>0).sum()),negative_among4=int((v[:4]<0).sum()),fourth_fifth_relative_magnitude_gap=float((v[3].abs()-v[4].abs())/v[3].abs()),omitted_energy_fraction=float(v[4:].square().sum()/v.square().sum())))
result=dict(records=records,scope='Post-hoc signed truncation diagnosis; does not change failed individual-product stability criteria. Candidate balanced-sign selection not yet evaluated.')
out=p/'MIDPOINT_SIGNED_CUTOFF_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print([(r['panel'],r['feature'],r['positive_among4'],round(r['fourth_fifth_relative_magnitude_gap'],3)) for r in records if r['feature'] in [1,3]])
