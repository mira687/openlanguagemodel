# OLM reference-parity results

| torch: `2.2.2` | transformers: `4.57.1` | python: `3.12.14` | platform: `Darwin 24.6.0 (x86_64)` | device: `cpu` | threads: `1` |

| architecture | reference | dtype | max abs logit diff | max abs loss diff | grad cosine sim | worst per-param cosine |
|---|---|---|---:|---:|---:|---:|
| `gpt2` | GPT2LMHeadModel | float32 | 8.941e-08 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `gpt2` | GPT2LMHeadModel | float64 | 3.925e-08 | 1.403e-09 | 1.000000000000 | 1.000000000 |
| `gpt2-stock` | GPT2LMHeadModel (stock config) | float32 | 3.144e-06 | 4.768e-07 | 0.999999999922 | 0.999999998 |
| `gpt2-stock` | GPT2LMHeadModel (stock config) | float64 | 3.133e-06 | 7.860e-09 | 0.999999999922 | 0.999999998 |
| `llama2-mha` | LlamaForCausalLM (num_kv_heads == num_heads) | float32 | 1.490e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `llama2-mha` | LlamaForCausalLM (num_kv_heads == num_heads) | float64 | 5.284e-08 | 2.426e-09 | 1.000000000000 | 1.000000000 |
| `llama2-gqa` | LlamaForCausalLM (grouped-query) | float32 | 1.490e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `llama2-gqa` | LlamaForCausalLM (grouped-query) | float64 | 6.079e-08 | 1.293e-09 | 1.000000000000 | 1.000000000 |
| `llama2-mha-tied` | LlamaForCausalLM (tie_word_embeddings=True) | float32 | 2.384e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `llama2-mha-tied` | LlamaForCausalLM (tie_word_embeddings=True) | float64 | 1.118e-07 | 1.988e-10 | 1.000000000000 | 1.000000000 |
| `llama3` | LlamaForCausalLM (rope_theta=500000) | float32 | 1.267e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `llama3` | LlamaForCausalLM (rope_theta=500000) | float64 | 6.248e-08 | 1.317e-09 | 1.000000000000 | 1.000000000 |
| `qwen2.5` | Qwen2ForCausalLM | float32 | 1.788e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `qwen2.5` | Qwen2ForCausalLM | float64 | 9.055e-08 | 3.302e-09 | 1.000000000000 | 1.000000000 |
| `mistral` | MistralForCausalLM | float32 | 1.192e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `mistral` | MistralForCausalLM | float64 | 7.392e-08 | 5.981e-10 | 1.000000000000 | 1.000000000 |
| `phi3` | Phi3ForCausalLM | float32 | 2.384e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `phi3` | Phi3ForCausalLM | float64 | 9.481e-08 | 4.241e-10 | 1.000000000000 | 1.000000000 |
| `gemma2` | Gemma2ForCausalLM | float32 | 2.980e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `gemma2` | Gemma2ForCausalLM | float64 | 1.128e-07 | 3.186e-09 | 1.000000000000 | 1.000000000 |
| `gemma2-stock` | Gemma2ForCausalLM (stock config) | float32 | 1.824e-05 | 4.768e-07 | 0.999999999443 | 0.999999999 |
| `gemma2-stock` | Gemma2ForCausalLM (stock config) | float64 | 1.847e-05 | 2.133e-07 | 0.999999999440 | 0.999999999 |
| `olmo` | OlmoForCausalLM | float32 | 1.788e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `olmo` | OlmoForCausalLM | float64 | 6.466e-08 | 2.579e-09 | 1.000000000000 | 1.000000000 |
| `qwen3-moe` | Qwen3MoeForCausalLM | float32 | 1.639e-07 | 0.000e+00 | 1.000000000000 | 1.000000000 |
| `qwen3-moe` | Qwen3MoeForCausalLM | float64 | 7.627e-08 | 4.651e-10 | 1.000000000000 | 1.000000000 |

## Reference configuration deviations

- `gpt2`: activation_function='gelu' (exact erf) instead of the GPT-2 default 'gelu_new'; OLM's ClassicFFN hardcodes nn.GELU(approximate='none')
- `llama2-mha-tied`: tie_word_embeddings=True, whereas released Llama 2/3 checkpoints leave the output head untied.
- `mistral`: head_dim is derived as embed_dim / num_heads. The released Mistral-Small-3.1-24B config sets an explicit head_dim=128 with hidden_size=5120 and 32 query heads, so q_proj is 4096 wide rather than 5120. MistralSmall3_1_Model takes no head_dim argument, so its 24B preset derives 160-dimensional heads and cannot reproduce the released projection shapes. This case therefore validates the block graph, not the advertised preset's geometry. Tracked upstream; compare Qwen3Model, which does accept head_dim.
- `gemma2`: hidden_activation='gelu' (exact erf) instead of the Gemma 2 default 'gelu_pytorch_tanh'; OLM's GeGLU calls F.gelu with no approximation
- `qwen3-moe`: The router load-balancing auxiliary loss is NOT compared. Both sides are scored with plain next-token cross entropy computed from logits, which excludes the auxiliary objective and its router gradients. Qwen3MoeForCausalLM can return router logits and add its configured aux loss, but OLM's Qwen3Model routes through SwiGLUMoEFFN -> MoEFeedForwardBase, whose forward() returns only the output tensor and discards the router logits, so there is nothing to compare against. Read this case as forward and LM-loss parity, not as evidence that real MoE training is equivalent: with no aux loss the two can agree here and still diverge in practice, since load balancing is what stops the router collapsing onto a few experts. OLM's newer olm.nn.moe.MoEFFN does return router logits.
