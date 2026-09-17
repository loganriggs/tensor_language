"""Group endpoint-only duplicate rows, preserving paired interventions and outputs."""
import torch


def group_rows(rows):
    groups=[];keys={};mapping=[]
    for row in rows:
        key=(tuple(row['ids']),row['city_position'],tuple(row['destination_positions']),row['context_id'],row['cue'])
        if key not in keys:
            keys[key]=len(groups)
            groups.append(dict(row,endpoint_pairs=[]))
        group=keys[key]
        pair=(row['uk_id'],row['us_id'])
        pairs=groups[group]['endpoint_pairs']
        if pair not in pairs:pairs.append(pair)
        mapping.append((group,pairs.index(pair)))
    if len(groups)%2:raise ValueError('Paired input groups required')
    for i in range(0,len(groups),2):
        a,b=groups[i:i+2]
        if a['context_id']!=b['context_id'] or a['cue']!='British' or b['cue']!='American' or a['endpoint_pairs']!=b['endpoint_pairs']:
            raise ValueError('Grouping changed paired intervention ordering')
    sizes={len(g['endpoint_pairs']) for g in groups}
    if len(sizes)!=1:raise ValueError('Uniform endpoint sets required')
    return groups,mapping


def expand(values,mapping,endpoints):
    """Restore original [arm,row,target+four controls] without new model calls."""
    return torch.stack([torch.cat((values[:,g,e:e+1],values[:,g,endpoints:]),-1) for g,e in mapping],1)
