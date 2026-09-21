from pathlib import Path
import torch,json
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);new=torch.load(p/'MIDPOINT_JOINT_CORE_REFIT_V1.pt',weights_only=True)['program']
def linear_cost(m):
 counts=(m!=0).sum(0);return dict(coefficients=int(counts.sum()),coefficient_multiplications=int(counts.sum()),additions=int((counts-1).clamp_min(0).sum()),nodes=m.shape[1])
def cost(program,names):
 items=[linear_cost(program[n]) for n in names];return dict(input_coefficients=sum(x['coefficients'] for x in items),linear_coefficient_multiplications=sum(x['coefficient_multiplications'] for x in items),linear_additions=sum(x['additions'] for x in items),input_linear_nodes=sum(x['nodes'] for x in items),variable_products=16,scalar_sum_additions=12,scalar_mean_subtractions=4,scalar_means=4,writer_coefficients=4608,scope='Scalar graph on provided normalized midpoint/source inputs. Dense residual writer extra; upstream model, source projection and normalization excluded and still required.')
a=cost(old,['A','B']);b=cost(new,['Pn','Pm','Tn','Tm']);result=dict(confirmed=a,shared_refit=b,input_coefficient_ratio=b['input_coefficients']/a['input_coefficients'],input_addition_ratio=b['linear_additions']/a['linear_additions'],extra_input_linear_nodes=b['input_linear_nodes']-a['input_linear_nodes'])
# Exact representation compilation check on the stored graph.
torch.set_num_threads(2);gen=torch.Generator().manual_seed(261103);n=torch.randn(17,1152,generator=gen,dtype=torch.float64);m=torch.randn(17,1152,generator=gen,dtype=torch.float64);g={k:v.double() for k,v in new.items()};explicit=(((n@g['Pn'])@g['Tn'])*((m@g['Pm'])@g['Tm']))@g['readout']-g['offset'];collapsed=((n@(g['Pn']@g['Tn']))*(m@(g['Pm']@g['Tm'])))@g['readout']-g['offset'];error=float((explicit-collapsed).norm()/explicit.norm());assert error<1e-12;result['explicit_graph_replay']=error
out=p/'MIDPOINT_GRAPH_PRICE_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
