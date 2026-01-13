"""
Test MoE components against HuggingFace implementation.
"""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from nanomoe.layers.moe import Experts, Moe, Router
from tests.utils import ATOL, RTOL, copy_expert_weights


def copy_router_weights(hf_moe: Qwen3MoeSparseMoeBlock, user_router: Router):
    """Copy router weights from HF MoE block to user's Router."""
    user_router.gate.weight.data = hf_moe.gate.weight.data.clone()


def copy_moe_weights(hf_moe: Qwen3MoeSparseMoeBlock, user_moe: Moe, num_experts: int):
    """Copy all weights from HF MoE to user's MOE."""
    copy_router_weights(hf_moe, user_moe.router)
    copy_expert_weights(hf_moe, user_moe.experts, num_experts)


# =============================================================================
# Router tests
# =============================================================================


def test_router_shape(test_config, random_hidden_states):
    """Test Router output shapes."""
    router = Router(test_config)
    router.eval()

    batch, seq, _ = random_hidden_states.shape

    with torch.no_grad():
        routing_probs, routing_map = router(random_hidden_states)

    assert routing_probs.shape == (batch, seq, test_config.num_experts)
    assert routing_map.shape == routing_probs.shape
    assert routing_map.dtype == torch.bool


def test_router_topk_selection(test_config, random_hidden_states):
    """Test Router selects exactly top-k experts per token."""
    router = Router(test_config)
    router.eval()

    with torch.no_grad():
        _, routing_map = router(random_hidden_states)

    num_selected = routing_map.sum(dim=-1)
    assert (num_selected == test_config.num_experts_per_tok).all()


def test_router_vs_hf(test_config, random_hidden_states):
    """Test Router produces same top-k selection as HF."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    router = Router(test_config)
    router.eval()
    copy_router_weights(hf_moe, router)

    batch, seq, hidden = random_hidden_states.shape
    flat_input = random_hidden_states.view(-1, hidden)

    with torch.no_grad():
        # HF router
        hf_logits = hf_moe.gate(flat_input)
        hf_weights = torch.softmax(hf_logits, dim=-1)
        _, hf_topk_indices = torch.topk(hf_weights, test_config.num_experts_per_tok)
        hf_selected = torch.zeros_like(hf_weights).scatter_(-1, hf_topk_indices, True)

        # User router
        _, user_map = router(random_hidden_states)
        user_map_flat = user_map.view(-1, test_config.num_experts)

    assert torch.equal(user_map_flat, hf_selected.bool())


def test_router_gradient(test_config, random_hidden_states):
    """Test gradients flow through Router."""
    router = Router(test_config)

    x = random_hidden_states.clone().requires_grad_(True)
    routing_probs, _ = router(x)
    routing_probs.sum().backward()

    assert x.grad is not None
    assert router.gate.weight.grad is not None


# =============================================================================
# Experts tests
# =============================================================================


def test_experts_structure(test_config):
    """Test Experts parameter shapes."""
    experts = Experts(test_config)

    expected_gate = (test_config.num_experts, test_config.hidden_size, test_config.moe_intermediate_size)
    expected_down = (test_config.num_experts, test_config.moe_intermediate_size, test_config.hidden_size)

    assert experts.gate_proj.shape == expected_gate
    assert experts.up_proj.shape == expected_gate
    assert experts.down_proj.shape == expected_down


def test_single_expert_vs_hf(test_config, random_hidden_states):
    """Test each expert matches HF individually."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    experts = Experts(test_config)
    experts.eval()
    copy_expert_weights(hf_moe, experts, test_config.num_experts)

    flat_input = random_hidden_states.view(-1, test_config.hidden_size)

    for i in range(test_config.num_experts):
        with torch.no_grad():
            hf_out = hf_moe.experts[i](flat_input)

            # Route all tokens to expert i
            counts = torch.zeros(test_config.num_experts, dtype=torch.long)
            counts[i:] = flat_input.shape[0]
            user_out = experts(flat_input, counts)

        assert torch.allclose(hf_out, user_out, rtol=RTOL, atol=ATOL), \
            f"Expert {i} mismatch. Max diff: {(hf_out - user_out).abs().max()}"


# =============================================================================
# Full MoE tests
# =============================================================================


def test_moe_vs_hf(test_config, random_hidden_states):
    """Test full MoE output matches HuggingFace."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    moe = Moe(test_config)
    moe.eval()
    copy_moe_weights(hf_moe, moe, test_config.num_experts)

    with torch.no_grad():
        hf_out, _ = hf_moe(random_hidden_states)
        user_out = moe(random_hidden_states)

    assert torch.allclose(hf_out, user_out, rtol=RTOL, atol=ATOL), \
        f"MoE mismatch. Max diff: {(hf_out - user_out).abs().max()}"


def test_moe_gradient(test_config, random_hidden_states):
    """Test gradients flow through full MoE."""
    moe = Moe(test_config)

    x = random_hidden_states.clone().requires_grad_(True)
    out = moe(x)
    out.sum().backward()

    assert x.grad is not None
    assert moe.router.gate.weight.grad is not None
    assert moe.experts.gate_proj.grad is not None
    assert moe.experts.up_proj.grad is not None
    assert moe.experts.down_proj.grad is not None
