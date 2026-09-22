"""All-output finite changes at token/position matched states; no fit."""
import collections,hashlib,json,time
import torch
from audit_conditional_residual_accounting import P,SCALE,load,full_eval,lean_eval

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 tokens=torch.load(P/'RESIDUAL_FRESH_TOKENS_V1.pt',weights_only=True)[:,:64]
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
 assert tokens.shape==(256,64) and x.shape==(16384,1152)
 groups=collections.defaultdict(list)
 for doc in range(256):
  for pos in range(64):groups[(int(tokens[doc,pos]),pos)].append(doc*64+pos)
 rng=torch.Generator().manual_seed(2209220203);pairs=[]
 for key,ids in sorted(groups.items()):
  order=torch.randperm(len(ids),generator=rng).tolist()
  pairs.extend((ids[order[j]],ids[order[j+1]]) for j in range(0,len(ids)-1,2))
 rec,don=torch.tensor(pairs).T
 assert len(set(rec.tolist()+don.tolist()))==2*len(pairs)
 assert torch.equal(rec%64,don%64) and (rec//64!=don//64).all()
 assert torch.equal(tokens.flatten()[rec],tokens.flatten()[don])
 ref=y[don]-y[rec];energy=ref.square().sum(0);rows=[]
 for seed in [1001,1002]:
  for kind,fn in [('MIXED_CP_FEATURES',None),('CONDITIONAL_CP',full_eval),('LEAN_CONDITIONAL_CP',lean_eval)]:
   name=f'{kind}_SEED{seed}'+('_RANK256' if fn else '')+'_V1.pt';p,h=load(name);p['coefficients']/=SCALE
   if fn:p['constant']/=SCALE
   def call(xx):return fn(p,xx) if fn else torch.stack([xx@f.T for f in p['factors']]).prod(0)@p['coefficients'].T
   pred=torch.cat([call(xx) for xx in x.split(2048)]);delta=pred[don]-pred[rec];err=delta-ref
   flip=(delta*ref<0);per=(err.square().sum(0)/energy).sqrt()
   row=dict(kind=kind,seed=seed,sha256=h,pooled_response_error=float(err.norm()/ref.norm()),feature_response_errors=per.tolist(),feature_reference_energy_share=(energy/energy.sum()).tolist(),energy_weighted_wrong_sign=(flip*ref.square()).sum(0).div(energy).tolist(),small_feature_rms=float(per[4:].square().mean().sqrt()),pooled_value_error=float((pred-y).norm()/y.norm()))
   rows.append(row);print(kind,seed,'response',row['pooled_response_error'],'small',row['small_feature_rms'],flush=True)
 result=dict(rows=rows,pairs=pairs,n_pairs=len(pairs),n_documents=len(set((rec//64).tolist()+(don//64).tolist())),n_token_ids=len(set(tokens.flatten()[rec].tolist())),pair_seed=2209220203,seconds=time.monotonic()-start,token_artifact_sha256=hashlib.sha256((P/'RESIDUAL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest(),scope='Opened panel; disjoint token states paired by same current token ID and position across documents without using labels. All16coordinates, purequartic numerator only. Not a controlled semantic counterfactual, native intervention, OOD or new heldout test. Documents recur across pairs; no independent-pair confidence claim.')
 (P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
