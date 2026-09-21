"""Global dictionary-node substitution with peer combinations and output refitting."""
import torch

def proposals(dag,outputs,nodes,x,weights):
    values=dag.evaluate(nodes,x);rows=[];sw=weights.sqrt()[:,None]
    for i,target in enumerate(nodes):
        peers=[(j,n) for j,n in enumerate(nodes) if n!=target and target not in dag.reachable([n]) and (dag.degree_limit is None or dag.degrees[n]<=dag.degrees[target])]
        if not peers:continue
        indices,refs=zip(*peers);design=values[:,indices]
        coefficients=torch.linalg.lstsq(sw*design,sw*values[:,i,None],driver='gelsd').solution[:,0]
        replacement=dag.linear(zip(refs,coefficients.tolist()));new=dag.replace(outputs,target,replacement)
        error=float((sw[:,0]*(design@coefficients-values[:,i])).norm()/(sw[:,0]*values[:,i]).norm())
        rows.append((new,dict(target=target,peers=list(refs),coefficients=coefficients.tolist(),local_error=error,cost=dag.cost(new))))
    return rows

def refit_outputs(dag,outputs,x,y,weights):
    terms=[]
    for n in outputs:
        key=dag.nodes[n]
        terms.extend(r for r,c in key[1]) if key[0]=='linear' else terms.append(n)
    nodes=sorted(set(terms));features=dag.evaluate(nodes,x);sw=weights.sqrt()[:,None]
    fit=torch.linalg.lstsq(sw*features,sw*y,driver='gelsd');coef=fit.solution
    new=[dag.linear(zip(nodes,row.tolist())) for row in coef.T]
    return new,dict(design_rank=int(fit.rank),design_width=len(nodes),cost=dag.cost(new))

def contract_readout(dag,outputs,boundary):
    """Exact quadratic polynomial compilation above a small declared cut.
    Boundary nodes are treated as independent symbols; no expansion below them.
    Caller compares resulting global cost before accepting the rewrite.
    """
    from arithmetic_dag import DAG
    if len(boundary)>8:raise ValueError('Bounded readout compilation supports at most eight symbols')
    abstract=DAG(degree_limit=2);mapping={node:abstract.input(i) for i,node in enumerate(boundary)}
    def visit(node):
        if node in mapping:return mapping[node]
        key=dag.nodes[node]
        if key[0]=='input':raise ValueError('Boundary does not cover all paths to outputs')
        if key[0]=='constant':new=abstract.constant()
        elif key[0]=='product':new=abstract.product(visit(key[1]),visit(key[2]))
        else:new=abstract.linear([(visit(n),c) for n,c in key[1]])
        mapping[node]=new;return new
    polynomials=abstract.polynomial([visit(n) for n in outputs],len(boundary));monomials={}
    for poly in polynomials:
        for powers in poly:
            node=dag.constant()
            for ref,power in zip(boundary,powers):
                for _ in range(power):node=dag.product(node,ref)
            monomials[powers]=node
    return [dag.linear([(monomials[powers],value) for powers,value in poly.items()]) for poly in polynomials]
