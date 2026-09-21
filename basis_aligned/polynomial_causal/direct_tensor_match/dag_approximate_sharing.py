"""Approximate functional sharing proposals, followed by global refit and scoring.
Sample similarity proposes edits only; it never certifies polynomial equality.
"""
import torch
from dag_edit_refit import fit_fixed,complexity

def proposals(dag,outputs,x,weights,max_candidates=4,max_relative_error=.1,nodes=None):
    eligible=nodes if nodes is not None else [n for n in dag.reachable(outputs) if dag.degrees[n]>=2 and n not in outputs]
    if len(eligible)<2:return []
    with torch.no_grad():values=dag.evaluate(eligible,x);gram=values.T@(weights[:,None]*values)
    candidates=[]
    for i,target in enumerate(eligible):
        for j,source in enumerate(eligible):
            if i==j or target in dag.reachable([source]):continue
            if dag.degree_limit is not None and dag.degrees[source]>dag.degrees[target]:continue
            if float(gram[i,i])<=1e-30 or float(gram[j,j])<=1e-30:continue
            scale=float(gram[i,j]/gram[j,j]);error=float((gram[i,i]-gram[i,j]**2/gram[j,j]).clamp_min(0)/gram[i,i])**.5
            if error<=max_relative_error:candidates.append(dict(target=target,source=source,scale=scale,proposal_relative_error=error))
    candidates.sort(key=lambda r:(r['proposal_relative_error'],r['target'],r['source']))
    return candidates[:max_candidates]

def one_round(dag,outputs,x,y,weights,steps=200,penalties=(.005,1e-5,1e-5),max_output_error=.01,max_candidates=4):
    baseline,error=fit_fixed(dag,outputs,x,y,weights,steps=steps);current,out=baseline.export();baseline_price=baseline.price();best=error+complexity(baseline_price,penalties);rows=[];winner=None
    for candidate in proposals(current,out,x,weights,max_candidates=max_candidates):
        replacement=current.linear([(candidate['source'],candidate['scale'])]);new=current.replace(out,candidate['target'],replacement)
        model,loss=fit_fixed(current,new,x,y,weights,steps=steps);price=model.price();value=loss+complexity(price,penalties)
        accepted=loss<=max_output_error**2 and value<best
        rows.append(dict(**candidate,squared_relative_error=loss,price=price,objective=value,accepted=accepted))
        if accepted:winner=model;best=value
    return winner,dict(baseline_squared_error=error,baseline_price=baseline_price,baseline_objective=error+complexity(baseline_price,penalties),candidates=rows,accepted=winner is not None,selected_objective=best)
