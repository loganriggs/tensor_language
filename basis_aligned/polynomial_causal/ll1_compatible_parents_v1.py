"""Select shared-parent incidences by joint, not individual, input membership."""
from itertools import combinations
import torch


def select(a, readers, nodes, minimum=.90):
    selections=[];candidates=[]
    for g in range(len(a)):
        available=[i for i,node in enumerate(nodes) if g in node['consumers']]
        basis=torch.linalg.qr(a[g].T,mode='reduced').Q
        choices=[]
        for k in range(len(available)+1):
            for ids in combinations(available,k):
                if k:
                    singular=torch.linalg.svdvals(readers[list(ids)].T)
                    if float(singular[-1]/singular[0])<1e-6:continue
                    span=torch.linalg.qr(readers[list(ids)].T,mode='reduced').Q
                    membership=float(torch.linalg.svdvals(basis.T@span).min().square())
                else:membership=1.
                score=sum(nodes[i]['mixed_energy_over_native'] for i in ids)
                choices.append(dict(ids=list(ids),membership=membership,score=score))
        eligible=[item for item in choices if item['membership']>=minimum]
        chosen=max(eligible,key=lambda item:(len(item['ids']),item['score']))
        selections.append(chosen['ids']);candidates.append(dict(group=g,choices=choices,chosen=chosen))
    consumers=[[g for g,ids in enumerate(selections) if i in ids] for i in range(len(readers))]
    retained=[i for i,ids in enumerate(consumers) if len(ids)>=2]
    new_nodes=[dict(nodes[i],consumers=consumers[i],source_parent=i) for i in retained]
    return readers[retained],new_nodes,dict(retained_source_parents=retained,groups=candidates,
        removed_single_consumer_parents=[i for i,ids in enumerate(consumers) if len(ids)==1])
