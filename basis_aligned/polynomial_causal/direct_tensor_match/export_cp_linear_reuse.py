"""Export preregistered128-reader merges and redteam per-output fidelity."""
import hashlib,json
import torch
from audit_conditional_residual_accounting import P,SCALE,load
from cp_linear_reuse import compile_program,evaluate

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);run=json.loads((P/'CP_LINEAR_REUSE_V1.json').read_text());assert run['pred_primary'];data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();pairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;rows=[]
 for stat in run['proposal_statistics']:
  seed=stat['seed'];p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');fs=p['factors'];flat=torch.cat(fs);C=p['coefficients']/SCALE;base=compile_program(fs,C);nf=[a.clone() for a in fs];nc=C.clone()
  for i,j,beta in stat['selected']:nf[i//512][i%512]=flat[j];nc[:,i%512]*=beta
  q=compile_program(nf,nc);arc=dict(bank=q['bank'].float(),indices=q['indices'].int(),coefficients=(q['coefficients']*SCALE).float(),writer=p['writer'].float(),parent_sha256=h,seed=seed)
  old=torch.cat([evaluate(base,xx) for xx in x.split(2048)]);new=torch.cat([evaluate(q,xx) for xx in x.split(2048)]);fp=torch.cat([evaluate(arc,xx.float()).double()/SCALE for xx in x.split(2048)]);replay=float((new-fp).norm()/new.norm());assert replay<1e-4
  delta=old[pairs[1]]-old[pairs[0]];editdelta=(new[pairs[1]]-new[pairs[0]])-delta;fe=((new-old).square().sum(0)/old.square().sum(0)).sqrt();fr=(editdelta.square().sum(0)/delta.square().sum(0)).sqrt();path=P/f'CP_LINEAR_REUSE_SEED{seed}_V1.pt';torch.save(arc,path)
  floats=sum(arc[k].numel() for k in ['bank','coefficients','writer']);row=dict(seed=seed,export_error=replay,parent_feature_value_errors=fe.tolist(),parent_feature_response_errors=fr.tolist(),stored_floats=floats,stored_int32=arc['indices'].numel(),array_bytes=4*(floats+arc['indices'].numel()),artifact_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest());rows.append(row);print(seed,max(fe.tolist()),max(fr.tolist()),replay,flush=True)
 (P/'CP_LINEAR_REUSE_EXPORT_V1.json').write_text(json.dumps(dict(rows=rows,scope='Both preregistered128merge candidates exported, neither selectedbytest. Opened256document per-coordinate editrelativeparents, not nativecomponentrecovery. Packedint32readerindices; no compiledproductreuse beyond linearreaders; metadata/fileoverhead separate.'),indent=2)+'\n')
if __name__=='__main__':main()
