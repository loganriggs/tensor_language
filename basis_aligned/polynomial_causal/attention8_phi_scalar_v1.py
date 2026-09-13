"""Weight-only scalarization of an identified head-input path, with native field check."""
from pathlib import Path
import json,time,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();fold=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True);gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);eig=gen['eigenvalues'][:4];scale=eig.abs().sqrt();C=scale[:,None]*fold['head_output_readers'][2];left,singular,right=torch.linalg.svd(C,full_matrices=False);parent=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);head=torch.load(P/'ATTENTION8_PHI_HEADS_V1_ARTIFACT.pt',weights_only=True);h=head['head_readings'][:,:,2];old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];gain=float(gen['lambdas'][0]);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));records=[]
 target=head['head_donor_fields'][:,:,2]
 for rank in [1,4]:
  approx=((h*scale)@left[:,:rank])@left[:,:rank].T/scale;fields=torch.zeros_like(target)
  for i,row in enumerate(rows):
   j=row['donor_id'];n=len(row['ids']);pos=int(prov['cue_positions'][i]);dh=approx[j,:n]-approx[i,:n];q=parent['parents'][i,:n,1];dv=2*(q*dh*eig).sum(-1)/parent['rho8_squared'][i,:n];dv[:pos+1]=0;fields[i,:n]=prov['gamma'][i,:n,:n]@(dv*gain/prov['rho9'][i,:n])
  records.append(dict(rank=rank,group_errors=[rel(fields[g*24:(g+1)*24],target[g*24:(g+1)*24]) for g in range(4)]))
 l=left[:,0]/scale;vr=singular[0]*right[0];pf=torch.load(P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt',weights_only=True);mu=fold['mixing'];files=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in files if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);V=sd['transformer.h.8.attn.c_v.weight'][256:384].double();V0=sd['transformer.h.0.attn.c_v.weight'][256:384].double();qcoef=2*pf['lambda8'][0]*((eig*l)@pf['product_coefficients']);known=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)['writers'][0];known=scale*(gen['eigenvectors'][:,:4].T@known.double());cos=float(abs(torch.nn.functional.cosine_similarity(known,left[:,0],dim=0)))
 program=dict(head_read_writer=l,preOV_scalar_reader=vr,current_scalar_reader=(1-mu)*vr@V,first_scalar_reader=mu*vr@V0,mlp7_product_coefficients=qcoef)
 result=dict(records=records,rank1_weight_capture=float(singular[0].square()/singular.square().sum()),singular_values=singular.tolist(),known_head8_2_scaled_writer_cosine=cos,rank1_all_group_10percent_pass=all(x<=.1 for x in records[0]['group_errors']),program_scalars=sum(t.numel() for t in program.values()),seconds=time.perf_counter()-tic,scope='Frozen weight SVD in abs-eigenvalue-scaled fourreader metric; tested actualconditional head8.2 donorwrite fields. Current/first scalar readers contractvalueweights; MLP7quadratic usesexternalnativeL7/R7. QK/input/normgeneration noteliminated, no native rank1 physicalconfirmation yet.')
 assert max(records[1]['group_errors'])<1e-10
 torch.save(program,P/'ATTENTION8_PHI_SCALAR_V1_PROGRAM.pt');(P/'ATTENTION8_PHI_SCALAR_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
