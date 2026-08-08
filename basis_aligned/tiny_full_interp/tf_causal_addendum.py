"""Reviewer-round addendum: the CAUSAL confirmation the standing sign rule
demands, run on the already-published DEPTH-1 cells so FINDING 6 can be
checked rather than assumed."""
import json, sys, torch
import tf_interp2 as I2

stems = sys.argv[1:] or [
    'tf_vanilla_d1_w32_b8192_s0', 'tf_vanilla_d1_w32_b8192_s1',
    'tf_vanilla_d1_w64_b8192_s0', 'tf_vanilla_d1_w64_b8192_s1',
    'tf_vanilla_d1_w128_b8192_s0', 'tf_vanilla_d1_w128_b8192_s1',
    'tf_vanilla_d1_w256_b8192_s0']
out = {}
for s in stems:
    D = I2.DeepFold(s)
    out[s] = {'composed_vs_causal': I2.composed_vs_causal(D, 32, 256),
              'mlp_composed_causal': I2.mlp_composed_causal(D),
              'causal_copy_test': I2.causal_copy_test(D)}
    cc = out[s]['mlp_composed_causal']
    print(s, 'direct r', [round(v['direct_route_pearson'], 3)
                          for k, v in cc.items() if k != 'note'],
          '| thru r', [round(v['through_mlp_pearson'], 3)
                       for k, v in cc.items() if k != 'note'], flush=True)
    ct = out[s]['causal_copy_test']
    print('   causal median rank of attended token',
          [int(v['median_rank_of_attended_token']) for k, v in ct.items()
           if k != 'note'], 'of', D.V, flush=True)
    del D
    torch.cuda.empty_cache()
json.dump(out, open('tf_causal_addendum_d1.json', 'w'), indent=2)
print('written')
