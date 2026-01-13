"""
Test Block (decoder layer) against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeRotaryEmbedding,
)

from nanomoe.layers.block import Block
from tests.utils import (
    ATOL,
    RTOL,
    copy_weights,
    copy_expert_weights,
    make_causal_mask,
    make_causal_mask_bool,
)


def build_block_mapping(config) -> dict[str, str]:
    """
    Build weight mapping from HF Qwen3MoeDecoderLayer to Block.

    HuggingFace -> Block naming:
    - input_layernorm -> norm1
    - self_attn -> attention
    - post_attention_layernorm -> norm2
    - mlp -> moe
    """
    return {
        # LayerNorms
        "input_layernorm.weight": "norm1.weight",
        "post_attention_layernorm.weight": "norm2.weight",
        # Attention
        "self_attn.q_proj.weight": "attention.q_proj.weight",
        "self_attn.k_proj.weight": "attention.k_proj.weight",
        "self_attn.v_proj.weight": "attention.v_proj.weight",
        "self_attn.o_proj.weight": "attention.o_proj.weight",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        # Router
        "mlp.gate.weight": "moe.router.gate.weight",
    }


def test_block_forward_smoke(test_config, random_hidden_states):
    """Smoke test: Block can run without errors."""
    block = Block(test_config)
    block.eval()

    seq_len = random_hidden_states.shape[1]
    mask = make_causal_mask_bool(seq_len)

    with torch.no_grad():
        out = block(random_hidden_states, mask)

    assert out.shape == random_hidden_states.shape


def test_block_gradient_smoke(test_config, random_hidden_states):
    """Smoke test: gradients flow through Block."""
    block = Block(test_config)

    seq_len = random_hidden_states.shape[1]
    mask = make_causal_mask_bool(seq_len)

    x = random_hidden_states.clone().requires_grad_(True)
    out = block(x, mask)
    out.sum().backward()

    assert x.grad is not None, "Gradient should flow through Block"
    assert block.attention.q_proj.weight.grad is not None
    assert block.moe.experts.gate_proj.grad is not None


def test_block_vs_hf(test_config, random_hidden_states, position_ids):
    """Test Block output matches HuggingFace Qwen3MoeDecoderLayer."""
    hf_decoder = Qwen3MoeDecoderLayer(test_config, layer_idx=0)
    hf_rope = Qwen3MoeRotaryEmbedding(test_config)
    hf_decoder.eval()

    block = Block(test_config)
    block.eval()

    # Copy weights
    mapping = build_block_mapping(test_config)
    copy_weights(hf_decoder, block, mapping)
    copy_expert_weights(hf_decoder.mlp, block.moe.experts, test_config.num_experts)

    batch_size, seq_len, _ = random_hidden_states.shape
    hf_mask = make_causal_mask(seq_len, batch_size)
    block_mask = make_causal_mask_bool(seq_len)

    with torch.no_grad():
        cos, sin = hf_rope(random_hidden_states, position_ids)
        hf_out = hf_decoder(
            hidden_states=random_hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=hf_mask,
        )
        block_out = block(random_hidden_states, block_mask)

    max_diff = (hf_out - block_out).abs().max().item()
    assert torch.allclose(hf_out, block_out, rtol=RTOL, atol=ATOL), \
        f"Block output mismatch. Max diff: {max_diff}"
