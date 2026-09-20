import itertools,json,time
from pathlib import Path
import torch
from exact_cp_fit import fit
from quartic_cp import cp_inner,cp_gram
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1658);pairs=[torch.linalg.qr(torch.randn(1152,2))[0].T for _ in range(4)];out=P/'EXACT_CP_SEPARATION_V1.json';assert not out.exists();rows=[];start=time.perf_counter()
 for ratio,rho in itertools.product([1.,.5,.1],[0.,.5,.95]):
  teacher=[torch.stack([q[0],rho*q[0]+(1-rho*rho)**.5*q[1]]) for q in pairs];tc=torch.tensor([[1.,0.],[0.,ratio],[0.,0.]]);tc/=cp_inner(tc,teacher,tc,teacher).sqrt();gram=cp_gram(teacher,teacher);overlap=float(gram[0,1]/(gram[0,0]*gram[1,1]).sqrt())
  torch.manual_seed(0);initial=[torch.randn_like(v) for v in teacher];initial=[v/v.norm(dim=1,keepdim=True) for v in initial];joint,_,_=fit(tc,teacher,2,'adam',0,1500,initial=initial);joint.update(method='joint_unit_raw1500',ratio=ratio,linear_overlap=rho,quartic_overlap=overlap);rows.append(joint)
  first,c1,f1=fit(tc,teacher,1,'adam',0,600);rc=torch.cat([tc,-c1],1);rf=[torch.cat([a,b],0) for a,b in zip(teacher,f1)];second,c2,f2=fit(rc,rf,1,'adam',100,600);initial=[torch.cat([a,b],0) for a,b in zip(f1,f2)];row,_,_=fit(tc,teacher,2,'adam',0,300,initial=initial);row.update(method='residual600_600_joint300',ratio=ratio,linear_overlap=rho,quartic_overlap=overlap,total_seconds=first['seconds']+second['seconds']+row['seconds']);rows.append(row);print(ratio,rho,joint['relative_error'],row['relative_error'],flush=True);out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='Controlled ratio/separation grid, rank2,d1152,one teacher orientation and one initialization per cell, same1500steps. Not multi-seed generalization.'),indent=2)+'\n')
if __name__=='__main__':main()
