from pathlib import Path
import torch,json,itertools,time
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].double();m=rows['m'].double();K=torch.load(p/'MIDPOINT_FACTOR_PROGRAMS_V1.pt',weights_only=True)['K'].double();K=(K+K.transpose(1,2))/2;M=torch.load(p/'MIDPOINT_TIED_READERS_V1.pt',weights_only=True)['common_moment'];e,V=torch.linalg.eigh(M+1e-6*M.trace()/1152*torch.eye(1152));S=(V*e.sqrt()[None,:])@V.T;parts=json.loads((p/'MIDPOINT_STABILITY_V1.json').read_text())['partitions'];records=[]
def directions(ids):
 nn=n[ids].flatten(0,1);mm=m[ids].flatten(0,1);a=nn.T@nn/len(nn);b=mm.T@mm/len(mm);moment=(a/a.trace()+b/b.trace())/2;e,V=torch.linalg.eigh(moment+1e-6*moment.trace()/1152*torch.eye(1152));s=(V*e.sqrt()[None,:])@V.T;inv=(V*e.rsqrt()[None,:])@V.T;result=[]
 for k in K:
  z=s@k@s;vals,U=torch.linalg.eigh((z+z.T)/2);idx=vals.abs().argsort(descending=True)[:4];v=S@inv@U[:,idx];v=v/v.norm(dim=0);result.append((v,vals[idx].sign()))
 return result
for part in parts:
 a=directions(part['first_documents']);b=directions(part['second_documents']);features=[]
 for g,((va,sa),(vb,sb)) in enumerate(zip(a,b)):
  corr=(va.T@vb).square()*sa[:,None]*sb[None,:];perm=max(itertools.permutations(range(4)),key=lambda perm:sum(float(corr[i,perm[i]]) for i in range(4)));matched=[float(corr[i,perm[i]]) for i in range(4)];features.append(dict(feature=g,signed_tensor_cosines=matched,minimum_cosine=min(matched),permutation=list(perm)))
 records.append(dict(partition=part['partition'],features=features));print(part['partition'],[round(z['minimum_cosine'],3) for z in features],flush=True)
result=dict(prediction_all_product_cosines_above_point9=all(f['minimum_cosine']>.9 for r in records for f in r['features']),records=records,seconds=time.perf_counter()-start,scope='Conditional input-product stability with fouroutputdefinitions fixed. Signedrank-one symmetric tensorcosines underfullcalcommonmetric, sign/permutation aligned. Does not repair failedoutputfeaturediscovery stability or establishsemantics.')
out=p/'MIDPOINT_TIED_STABILITY_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result['prediction_all_product_cosines_above_point9'])
