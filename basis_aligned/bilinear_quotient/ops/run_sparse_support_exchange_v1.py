#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_learning pred_c_component
"""Fixed144x4quadratics, exchange up to16of512rootproducts from256newpairs.
Two archivedstarts1101/1102. Replay<1e-4,normal/formula<1e-8;same1088products.
Bothcapture>=1.1initial/query<=initial. Bothroot1response/sensitivity<=10%.
Exactcoefficient objective, fixedcandidatepoolseed11700; no graphglobalopt claim.
"""
import os,sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from sparse_support_exchange import exchange_path,fitted_score
 from sparse_quartic_bank import gram,native_cross,features,entries,support,validate_pairs
 from shared_quadratic_bank import normalize_bank
 from check_sparse_support_exchange import controls
 from quartic_cp import directional
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 def pool(pairs,width):
  old=pairs.cpu();allpairs=torch.triu_indices(width,width,offset=1);encoded=set((old[0]*width+old[1]).tolist())
  keep=torch.tensor([int(a)*width+int(b) not in encoded for a,b in allpairs.T]);rest=allpairs[:,keep]
  take=torch.randperm(rest.shape[1],generator=torch.Generator().manual_seed(11700))[:256]
  result=torch.cat([old,rest[:,take]],1);validate_pairs(result,width);assert result.shape==(2,768);return result
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checked=controls();torch.manual_seed(11701);dtype=torch.float64
  U,V=normalize_bank(torch.randn(144,4,12,dtype=dtype),torch.randn(144,4,12,dtype=dtype));pairs=pool(support(144,512,1100),144);G=gram(U,V,pairs);X=torch.randn(16,768,dtype=dtype)@G
  take,C,info=exchange_path(G,X,torch.arange(512),torch.arange(144),steps=1)
  x=torch.randn(9,12,dtype=dtype);assert (features(x,U,V,pairs[:,take])@C.T).shape==(9,16)
  assert (entries(U,V,pairs[:,take],torch.randint(12,(11,4)))@C.T).shape==(11,16)
  assert len(take)==512 and all(i in take for i in range(144));print(json.dumps(dict(control_cases=len(checked),actual768gram=True,exchange=info,products=1088)));return
 from audit_root_matched_reader import CK
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'SPARSE_SUPPORT_EXCHANGE_NATIVE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].cuda().double()
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
 x=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[z]*4) for z in x.split(128)])
 text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].cuda()/scale;weights=labels['weight'].cuda()
 pairdata=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(pairdata['pairs_flat'],device='cuda').T;reference=torch.tensor([r['reference'] for r in pairdata['pair_rows']],device='cuda',dtype=torch.float64)/scale
 rows=[]
 for seed in [1101,1102]:
  source=torch.load(P/f'SPARSE_QUARTIC_BANK_SEED{seed}_V1.pt',weights_only=True);U,V=[a.cuda().double() for a in source['factors']];pairs=pool(source['pairs'],144).cuda();G=gram(U,V,pairs);X=torch.cat([native_cross(teacher,U,V,p) for p in pairs.split(64,dim=1)],dim=1)
  initial_ids=torch.arange(512,device='cuda');protected=torch.where(pairs[0]==pairs[1])[0];assert len(protected)==144
  _,initial_C=fitted_score(G,X,initial_ids,1e-6)
  def assess(ids,C):
   p=pairs[:,ids];pred=features(text,U,V,p)@C.T;sens=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt()
   return dict(text_error=float((pred-target).norm()/target.norm()),gaussian_error=float((features(x,U,V,p)@C.T-truth).norm()/truth.norm()),sampled_coefficient_error=float((entries(U,V,p,indices)@C.T-query).norm()/query.norm()),root1_sensitivity_error=float(sens[1]),root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()))
  initial=assess(initial_ids,initial_C);ref=features(text[:128],U,V,pairs[:,:512])@initial_C.T;old=features(text[:128],U,V,pairs[:,:512])@(source['coefficients'].cuda().double()/scale).T;replay=float((ref-old).norm()/old.norm());assert replay<1e-4
  ids,C,info=exchange_path(G,X,initial_ids,protected,ridge=1e-6,steps=16);selected=pairs[:,ids];final=assess(ids,C)
  normal=float((C@(G[ids][:,ids]+1e-6*torch.eye(512,device='cuda',dtype=G.dtype))-X[:,ids]).norm()/X[:,ids].norm())
  fp=features(text[:128].float(),U.float(),V.float(),selected)@(C*scale).float().T;ref=features(text[:128],U,V,selected)@C.T;drift=float((fp.double()/scale-ref).norm()/ref.norm())
  assert len(torch.unique(selected))==144 and selected.shape[1]==512
  path=P/f'SPARSE_SUPPORT_EXCHANGE_SEED{seed}_V1.pt';torch.save(dict(factors=source['factors'],pairs=selected.cpu(),coefficients=(C*scale).cpu().float(),writer=writer.cpu().float(),seed=seed,degree=4),path)
  decoded=[dict(**r,removed_pair=pairs[:,r['remove']].tolist(),added_pair=pairs[:,r['add']].tolist()) for r in info['history']];info['history']=decoded
  row=dict(seed=seed,initial=initial,final=final,exchange=info,baseline_replay=replay,normal_residual=normal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest());rows.append(row);print(json.dumps(row),flush=True)
 pred=dict(pred_a_integrity=all(r['baseline_replay']<1e-4 and r['normal_residual']<1e-8 and r['export_error']<1e-4 and all(e['formula_replay']<1e-8 for e in r['exchange']['history']) for r in rows),pred_b_learning=all(r['exchange']['final_score']>=1.1*r['exchange']['initial_score'] and r['final']['sampled_coefficient_error']<=r['initial']['sampled_coefficient_error'] for r in rows),pred_c_component=all(r['final']['root1_same_token_error']<=.1 and r['final']['root1_sensitivity_error']<=.1 for r in rows))
 result=dict(predictions=pred,rows=rows,products=1088,coefficients=1353728,integer_indices=1024,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Up to16greedy exact product exchanges inside fixed768candidate pool, protectall144diagonals. Fixed learnedquadratics, refit16readouts. Coefficient-only graph edits; opened diagnostics, no nativefiniteintervention/OOD/semantic adoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
