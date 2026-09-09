"""Two shared-destination joins and their shared-normalizer-only composition."""
import torch
import join_write_read_degree_reference as D
import query_hop_fork_reference as H
import suffix_join_middle_match_reference as R


def populations():
    out = {}
    for pop, (tokens, _, metadata) in R.populations().items():
        worlds = []
        for i in range(0, len(tokens), 4):
            row = tokens[i]; m = metadata[i]
            fmap = torch.empty(24, dtype=torch.long).scatter_(0, row[:48:2], row[1:48:2])
            pos = torch.empty(24, dtype=torch.long).scatter_(0, row[:48:2], torch.arange(0,48,2))
            a = int(row[49]); b = int(fmap[a]); c = int(fmap[b])
            if not (pos[b] > pos[a] and pos[b] > pos[c]): continue
            predecessor = int(torch.where(fmap == a)[0][0])
            origins = row[None].expand(2,-1).clone(); origins[1,49] = predecessor
            tok, meta = H.fork(origins, [{**m,'query':0,'entity':a}, {**m,'query':1,'entity':predecessor}])
            masks = {}
            for j, source in enumerate((c,a)):
                mask = torch.zeros(51,51,dtype=torch.bool)
                mask[pos[b]:pos[b]+2,pos[source]:pos[source]+2] = True
                masks[j] = mask[None].expand(8,-1,-1)
            worlds.append({'tokens':tok,'metadata':meta,'masks':masks,'heads':{0:1,1:2}})
        assert len(worlds) == 32; out[pop] = worlds
    return out


def numerator(curve, alpha):
    c = curve['coefficients']
    return ((c[:,:,3]*alpha+c[:,:,2])*alpha+c[:,:,1])*alpha+c[:,:,0]


def compile_pair(program, context, u, v):
    first, second = D.compile_curve(program,context,u), D.compile_curve(program,context,v)
    positions = first['source_positions']
    assert torch.equal(positions,second['source_positions'])
    return {'first':first,'second':second,'x':context['x'][:,positions],
            'u':u[:,positions],'v':v[:,positions]}


def evaluate(pair, a, b):
    first, second = pair['first'],pair['second']
    x = pair['x']+(a-1)*pair['u']+(b-1)*pair['v']
    gain3 = (x.square().mean(-1)+first['epsilon']).pow(-1.5)
    p = numerator(first,a)+numerator(second,b)-numerator(first,1.)
    return first['background']+(gain3[...,None]*p).sum(1)


def controls():
    # A separable two-output numerator, with a genuinely shared nonlinear norm.
    first = {'coefficients':torch.zeros(1,1,4,2,dtype=torch.float64),
             'epsilon':1e-8,'background':torch.zeros(1,2,dtype=torch.float64)}
    second = {**first,'coefficients':first['coefficients'].clone()}
    first['coefficients'][0,0,:2] = torch.tensor([[2.,3.],[1.,0.]])
    second['coefficients'][0,0,:2] = torch.tensor([[3.,2.],[0.,1.]])
    pair = {'first':first,'second':second,'x':torch.tensor([[[2.,1.]]]),
            'u':torch.tensor([[[.1,.2]]]),'v':torch.tensor([[[.2,-.1]]])}
    pair = {k:v.double() if isinstance(v,torch.Tensor) else v for k,v in pair.items()}
    checks = {}; a,b = .25,.5
    x = pair['x']+(a-1)*pair['u']+(b-1)*pair['v']; gain = (x.square().mean(-1)+1e-8).pow(-1.5)
    expected = torch.tensor([[2+a,2+b]],dtype=torch.float64)*gain
    checks['planted_separable_numerator'] = bool(torch.allclose(evaluate(pair,a,b),expected,atol=1e-12,rtol=1e-12))
    with_mixed = expected+(a-1)*(b-1)*gain
    checks['live_mixed_numerator_negative'] = float((with_mixed-evaluate(pair,a,b)).abs().max())>.01
    return {'passed':all(checks.values()),'checks':checks}
