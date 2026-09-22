"""Does reader sharing beat simply deleting the same affected quartic atoms?"""
import json
import torch
from cp_linear_reuse import compile_program,evaluate
from audit_conditional_residual_accounting import P,SCALE,load

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);run=json.loads((P/'CP_LINEAR_REUSE_V1.json').read_text());data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();pairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;rows=[]
 for stat in run['proposal_statistics']:
  seed=stat['seed'];p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');C=p['coefficients']/SCALE;fs=p['factors'];base=compile_program(fs,C);ix=sorted({a%512 for a,b,c in stat['selected']});mask=torch.ones(512,dtype=torch.bool);mask[ix]=False;q=compile_program([f[mask] for f in fs],C[:,mask]);old=torch.cat([evaluate(base,xx) for xx in x.split(2048)]);pred=torch.cat([evaluate(q,xx) for xx in x.split(2048)]);ref=old[pairs[1]]-old[pairs[0]];error=pred[pairs[1]]-pred[pairs[0]]-ref
  rows.append(dict(seed=seed,deleted_atoms=len(ix),products=3*int(mask.sum()),stored_floats=q['bank'].numel()+q['coefficients'].numel()+1152*16,parent_value_error=float((pred-old).norm()/old.norm()),parent_response_error=float(error.norm()/ref.norm()),feature_value_errors=((pred-old).square().sum(0)/old.square().sum(0)).sqrt().tolist()))
 result=dict(rows=rows,scope='Posthoc matchedaffectedatom deletioncontrol onopened256documents. No refit/selection: deleteall128atoms touchedbyprimaryreadermerges. Cheaper alternative comparedwithfidelity, not presumedmatchedcost.');(P/'CP_LINEAR_REUSE_DELETION_CONTROL_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
