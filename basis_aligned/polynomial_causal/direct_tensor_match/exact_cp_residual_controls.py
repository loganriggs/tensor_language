import json,itertools,time
from pathlib import Path
import torch
from quartic_cp import cp_inner
from exact_cp_fit import fit
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=P/'EXACT_CP_RESIDUAL_CONTROLS_V1.json';assert not out.exists();d=1152;torch.manual_seed(1644);teacher=[torch.randn(2,d) for _ in range(4)];teacher=[x/x.norm(dim=1,keepdim=True) for x in teacher];tc=torch.randn(3,2);tc/=cp_inner(tc,teacher,tc,teacher).sqrt();records=[];start=time.perf_counter()
 for opt,seed in itertools.product(['adam','muon'],[0,1]):
  row,C,F=fit(tc,teacher,2,opt,seed,1500);row.update(method='joint1500',optimizer=opt,seed=seed);records.append(row)
  r1,c1,f1=fit(tc,teacher,1,opt,seed,600);resC=torch.cat([tc,-c1],1);resF=[torch.cat([t,a],0) for t,a in zip(teacher,f1)];r2,c2,f2=fit(resC,resF,1,opt,seed+100,600);initial=[torch.cat([a,b],0) for a,b in zip(f1,f2)];row,C,F=fit(tc,teacher,2,opt,seed,300,initial=initial);row.update(method='residual600_600_joint300',optimizer=opt,seed=seed,first=r1,second_residual=r2,total_seconds=r1['seconds']+r2['seconds']+row['seconds']);records.append(row)
  print(opt,seed,records[-2]['relative_error'],row['relative_error'],row['teacher_feature_correlations'],flush=True)
  out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start,teacher= dict(C=tc.tolist(),factors=[v.tolist() for v in teacher]),scope='Rank-two1152input planted CP, same finalwidth2 and total1500optimizersteps. Joint vs two residual rank-one stages then joint refinement; walltime differs and is recorded.'),indent=2)+'\n')
if __name__=='__main__':main()
