"""Extract fixed canonical scalar features without retaining a 1152-output predictor."""
import json,hashlib
from pathlib import Path
import torch
from arithmetic_dag import DAG
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def evaluate(s,x):
 p=(x@s['A'].T)*(x@s['B'].T)
 h=(p@s['root_left'].T)*(p@s['root_right'].T)
 return h@s['quartic_readout'].T+p@s['quadratic_readout'].T+s['constant']

def build(s):
 d=DAG(degree_limit=4);inputs=[d.input(i) for i in range(s['A'].shape[1])];left=[d.linear(zip(inputs,row.tolist())) for row in s['A']];right=[d.linear(zip(inputs,row.tolist())) for row in s['B']];p=[d.product(a,b) for a,b in zip(left,right)];l=[d.linear(zip(p,row.tolist())) for row in s['root_left']];r=[d.linear(zip(p,row.tolist())) for row in s['root_right']];h=[d.product(a,b) for a,b in zip(l,r)];one=d.constant();out=[d.linear(list(zip(h,a.tolist()))+list(zip(p,b.tolist()))+[(one,float(c))]) for a,b,c in zip(s['quartic_readout'],s['quadratic_readout'],s['constant'])]
 return d,out

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 path=P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt';s={k:v.double() for k,v in torch.load(path,weights_only=True)['programs'][.5].items()};view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].double();mu=view['output_mean'].double()
 exact={k:s[k].clone() for k in ['A','B','root_left','root_right']};exact.update(quartic_readout=U.T@s['output_writer'],quadratic_readout=(U.T@s['skip_writer'])@s['skip_reader'],constant=U.T@(s['constant']-mu));archive={k:t.float() for k,t in exact.items()};loaded={k:t.double() for k,t in archive.items()};panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];checks=[]
 for panel in panels:
  x=panel['rows'].double();ref=(quartic(s,x)-mu)@U;y=evaluate(loaded,x);checks.append(dict(context=panel['context'],exact_composition_error=float((evaluate(exact,x)-ref).norm()/ref.norm()),archive_per_mode_error=((y-ref).norm(dim=0)/ref.norm(dim=0)).tolist()))
 d,out=build(loaded);cost=d.cost(out);x=panels[0]['rows'][:16].double();replay=float((d.evaluate(out,x)-evaluate(loaded,x)).norm()/evaluate(loaded,x).norm());separate=[d.cost([node]) for node in out]
 assert replay<1e-12 and max(r['exact_composition_error'] for r in checks)<1e-12 and max(v for r in checks for v in r['archive_per_mode_error'])<1e-5
 assert cost['products']==10 and cost['stored_coefficients']==13916 and all(c['stored_coefficients']==13883 for c in separate)
 torch.save(dict(program=archive,output_directions=U.float(),source_program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_alpha=.5,scope='Four centered canonical scalar projections only. Does not reproduce the entire primary vector output.'),P/'EXTRACTED_SCALAR_MODES_V1.pt')
 result=dict(shared_cost=cost,separate_costs=separate,graph_replay=replay,checks=checks,scope='Exact folded scalar readouts plus FP32 rounding audit. Shared four-feature dictionary costs13916 coefficients/10products; independent copies duplicate common input work. Fixed downstream residual writers and normalization are separate and must be priced.')
 (P/'EXTRACTED_SCALAR_MODES_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
