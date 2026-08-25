# Protocol (practice build, v0)
- Substrate: bilin18 (Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd), D=1152, T=256.
- Dev split: FineWeb rows skip=80 (fit) — public iteration data.
- Eval split: FineWeb rows skip=7000, 960 rows, CE on positions >= 64 (184,320
  positions). Private + OOD splits: M2.
- Anchors: per-component mean constant + optimal constant (Adam 3e-3, mean-init,
  150 steps MLPs/attn layers, 100 heads, batch 8) — see bench/anchors/_meta.json.
- Fidelity: (delta_opt - delta_repl) / delta_opt. Optimal constant = 0, identity = 1.
- Head convention: a head's "output" is its 128-dim c_proj input slice.
