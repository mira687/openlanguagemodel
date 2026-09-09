"""Llama 2 and Llama 3 vs ``transformers.LlamaForCausalLM``.

Llama 2 is registered twice so that both OLM attention paths are exercised:
``Llama2Block`` picks ``FlashAttentionwithRoPE`` when ``num_kv_heads ==
num_heads`` and ``GroupedQueryAttention`` otherwise.
"""

from __future__ import annotations

from olm.models.meta.llama2 import Llama2Model
from olm.models.meta.llama3 import Llama3Model
from transformers import LlamaConfig, LlamaForCausalLM

from .._spec import ParityCase
from ._llama_family import llama_family_map

VOCAB = 61
EMBED = 64
INTERMEDIATE = 48
LAYERS = 2
HEADS = 4
MAX_SEQ = 32
HEAD_DIM = EMBED // HEADS


def _config(*, num_kv_heads: int, rope_theta: float, rms_eps: float) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=EMBED,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=MAX_SEQ,
        rope_theta=rope_theta,
        rms_norm_eps=rms_eps,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def _map(num_kv_heads: int):
    def build(olm, hf):
        return llama_family_map(
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
        )

    return build


# OLM hardcodes rms_norm_eps=1e-5 for both Llama 2 and Llama 3 blocks, which
# matches the LlamaConfig default.
RMS_EPS = 1e-5

CASES = [
    ParityCase(
        name="llama2-mha",
        reference="LlamaForCausalLM (num_kv_heads == num_heads)",
        build_hf=lambda: LlamaForCausalLM(
            _config(num_kv_heads=HEADS, rope_theta=10000.0, rms_eps=RMS_EPS)
        ),
        build_olm=lambda: Llama2Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=HEADS,
            max_seq_len=MAX_SEQ,
            rope_theta=10000.0,
        ),
        build_map=_map(HEADS),
        notes="Exercises OLM's FlashAttentionwithRoPE path (multi-head).",
    ),
    ParityCase(
        name="llama2-gqa",
        reference="LlamaForCausalLM (grouped-query)",
        build_hf=lambda: LlamaForCausalLM(
            _config(num_kv_heads=2, rope_theta=10000.0, rms_eps=RMS_EPS)
        ),
        build_olm=lambda: Llama2Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=2,
            max_seq_len=MAX_SEQ,
            rope_theta=10000.0,
        ),
        build_map=_map(2),
        notes="Exercises OLM's GroupedQueryAttention path.",
    ),
    ParityCase(
        name="llama3",
        reference="LlamaForCausalLM (rope_theta=500000)",
        build_hf=lambda: LlamaForCausalLM(
            _config(num_kv_heads=2, rope_theta=500000.0, rms_eps=RMS_EPS)
        ),
        build_olm=lambda: Llama3Model(
            vocab_size=VOCAB,
            embed_dim=EMBED,
            intermediate_size=INTERMEDIATE,
            num_layers=LAYERS,
            num_heads=HEADS,
            num_kv_heads=2,
            max_seq_len=MAX_SEQ,
            rope_theta=500000.0,
        ),
        build_map=_map(2),
        notes=(
            "Llama 3.1/3.2 checkpoints additionally use scaled RoPE for long "
            "context; OLM implements plain RoPE, so only the base rope_theta "
            "behaviour is compared here."
        ),
    ),
]
