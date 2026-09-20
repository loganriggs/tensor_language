"""Separate metric mismatch and residual alignment after frozen graph edits."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);key=('second_floor01','muon',1)
 source=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True);s=source['students'][key]
 chosen=torch.load(P/'DAG_OUTPUT_SHARING_V1.pt',weights_only=True)['students'][key]
 U,V,C=[s[n].double() for n in ['U','V','C']];W=chosen['W'].double();Z=chosen['Z'].double()
 targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];i,j=torch.triu_indices(4,4);rows=[]
 for panel,y in zip(panels,targets):
  x=panel['rows'].double();q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);phi=q[:,i]*q[:,j];before=phi@C.T;after=phi@Z.T@W.T;r=before-y;e=after-before;den=y.square().sum();base=r.square().sum()/den;added=e.square().sum()/den;cross=2*(r*e).sum()/den;total=(after-y).square().sum()/den
  row=dict(before_relative_error=float(base.sqrt()),after_relative_error=float(total.sqrt()),edit_error_relative_to_student=float(e.norm()/before.norm()),base_squared_error=float(base),edit_squared_error=float(added),residual_cross_term=float(cross),identity_error=float(abs(total-base-added-cross)));assert row['identity_error']<1e-10;rows.append(row)
 out=dict(source_key=list(key),rank=4,weighted_coefficient_edit_error=.013162652755272047,panels=rows,scope='Fixed prior choice; descriptive reused-panel residual decomposition, no new training or selection.')
 (P/'OUTPUT_SHARING_RESIDUAL_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
