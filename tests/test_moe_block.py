"""
Test SparseMoeBlock against HuggingFace implementation.
"""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from tests.utils import ATOL, RTOL, copy_weights
from nanoqwen.layers.moe import Router, Experts, MOE


def build_moe_mapping(num_experts: int) -> dict[str, str]:
    """
    Build weight mapping for MoE block (HF ModuleList style).

    This creates mappings for:
    - gate (router)
    - each expert's gate_proj, up_proj, down_proj

    Note: Qwen3 MoE does NOT have shared_expert (unlike Qwen2 MoE)
    """
    mapping = {
        # Router
        "gate.weight": "gate.weight",
    }

    # Add mappings for each expert
    for i in range(num_experts):
        mapping[f"experts.{i}.gate_proj.weight"] = f"experts.{i}.gate_proj.weight"
        mapping[f"experts.{i}.up_proj.weight"] = f"experts.{i}.up_proj.weight"
        mapping[f"experts.{i}.down_proj.weight"] = f"experts.{i}.down_proj.weight"

    return mapping


def build_router_mapping() -> dict[str, str]:
    """Build weight mapping for Router class."""
    return {"gate.weight": "gate.weight"}


def copy_router_weights(hf_moe: Qwen3MoeSparseMoeBlock, user_router: Router):
    """Copy router weights from HF MoE block to user's Router."""
    user_router.gate.weight.data = hf_moe.gate.weight.data.clone()


def copy_experts_to_fused(hf_moe: Qwen3MoeSparseMoeBlock, user_experts: Experts):
    """
    Copy HF's ModuleList expert weights to user's fused 3D parameter tensors.

    HF structure: experts[i].gate_proj.weight is (intermediate, hidden)
    User structure: gate_proj is (num_experts, hidden, intermediate)

    Note: HF weights are (out_features, in_features), user wants (num_experts, in, out)
    """
    num_experts = len(hf_moe.experts)

    for i in range(num_experts):
        # HF: (intermediate, hidden) -> User: (hidden, intermediate) via transpose
        user_experts.gate_proj.data[i] = hf_moe.experts[i].gate_proj.weight.data.T.clone()
        user_experts.up_proj.data[i] = hf_moe.experts[i].up_proj.weight.data.T.clone()
        # down_proj: HF is (hidden, intermediate), user is (intermediate, hidden)
        user_experts.down_proj.data[i] = hf_moe.experts[i].down_proj.weight.data.T.clone()


def copy_moe_weights(hf_moe: Qwen3MoeSparseMoeBlock, user_moe: MOE):
    """Copy all weights from HF MoE to user's MOE implementation."""
    copy_router_weights(hf_moe, user_moe.router)
    copy_experts_to_fused(hf_moe, user_moe.experts)


# =============================================================================
# Tests for user's Router implementation
# =============================================================================


def test_router_shape(test_config, random_hidden_states):
    """Test that Router produces correct output shapes."""
    router = Router(test_config)
    router.eval()

    batch, seq, hidden = random_hidden_states.shape

    with torch.no_grad():
        routing_probs, routing_map = router(random_hidden_states)

    # routing_probs should have shape (batch, seq, num_experts)
    assert routing_probs.shape == (batch, seq, test_config.num_experts), \
        f"Expected routing_probs shape {(batch, seq, test_config.num_experts)}, got {routing_probs.shape}"

    # routing_map should be boolean with same shape
    assert routing_map.shape == routing_probs.shape, \
        f"routing_map shape {routing_map.shape} should match routing_probs shape {routing_probs.shape}"
    assert routing_map.dtype == torch.bool, "routing_map should be boolean"


def test_router_forward(test_config, random_hidden_states):
    """Test that Router gate output matches HF's gate."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    router = Router(test_config)
    router.eval()

    # Copy weights from HF gate to user's router
    copy_router_weights(hf_moe, router)

    batch, seq, hidden = random_hidden_states.shape
    flat_hidden = random_hidden_states.view(-1, hidden)

    with torch.no_grad():
        # HF computes: router_logits = gate(hidden_states.view(-1, hidden))
        hf_router_logits = hf_moe.gate(flat_hidden)

        # User's router internally does: gate(x) then softmax
        # We test the gate output directly
        user_gate_output = router.gate(random_hidden_states)

    # User's gate output should match HF when flattened
    user_gate_flat = user_gate_output.view(-1, test_config.num_experts)

    assert torch.allclose(hf_router_logits, user_gate_flat, rtol=RTOL, atol=ATOL), \
        f"Router gate output mismatch. Max diff: {(hf_router_logits - user_gate_flat).abs().max()}"


def test_router_topk_selection(test_config, random_hidden_states):
    """Test that Router selects exactly top-k experts per token."""
    router = Router(test_config)
    router.eval()

    with torch.no_grad():
        routing_probs, routing_map = router(random_hidden_states)

    # Each token should have exactly num_experts_per_tok experts selected
    num_selected = routing_map.sum(dim=-1)
    expected = test_config.num_experts_per_tok

    assert (num_selected == expected).all(), \
        f"Expected {expected} experts per token, got {num_selected}"


def test_router_gradient(test_config, random_hidden_states):
    """Test that gradients flow through Router."""
    router = Router(test_config)

    x = random_hidden_states.clone().requires_grad_(True)
    routing_probs, routing_map = router(x)
    routing_probs.sum().backward()

    assert x.grad is not None, "Gradient should flow through Router input"
    assert router.gate.weight.grad is not None, "Router gate should have gradient"


# =============================================================================
# Tests for user's Experts implementation
# =============================================================================


def test_experts_structure(test_config):
    """Test that Experts has correct parameter shapes."""
    experts = Experts(test_config)

    # gate_proj: (num_experts, hidden_size, moe_intermediate_size)
    assert experts.gate_proj.shape == (
        test_config.num_experts,
        test_config.hidden_size,
        test_config.moe_intermediate_size,
    ), f"gate_proj shape mismatch: {experts.gate_proj.shape}"

    # up_proj: (num_experts, hidden_size, moe_intermediate_size)
    assert experts.up_proj.shape == (
        test_config.num_experts,
        test_config.hidden_size,
        test_config.moe_intermediate_size,
    ), f"up_proj shape mismatch: {experts.up_proj.shape}"

    # down_proj: (num_experts, moe_intermediate_size, hidden_size)
    assert experts.down_proj.shape == (
        test_config.num_experts,
        test_config.moe_intermediate_size,
        test_config.hidden_size,
    ), f"down_proj shape mismatch: {experts.down_proj.shape}"


def test_experts_forward(test_config, random_hidden_states):
    """Test that Experts forward produces correct output shape."""
    experts = Experts(test_config)
    experts.eval()

    batch, seq, hidden = random_hidden_states.shape
    num_tokens = batch * seq

    # Simulate expert tokens (all tokens assigned to experts)
    expert_tokens = random_hidden_states.view(-1, hidden)

    # expert_tokens_count is cumulative count per expert
    # For simplicity, distribute tokens evenly across experts
    tokens_per_expert = num_tokens // test_config.num_experts
    expert_tokens_count = torch.tensor([
        tokens_per_expert * (i + 1) for i in range(test_config.num_experts)
    ])

    with torch.no_grad():
        output = experts(expert_tokens[:expert_tokens_count[-1]], expert_tokens_count)

    # Output should have shape (total_selected_tokens, hidden_size)
    # Note: exact shape depends on implementation details
    assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"


# =============================================================================
# Tests for user's MOE implementation
# =============================================================================


def test_single_expert_vs_hf(test_config, random_hidden_states):
    """Test that a single expert produces the same output as HF."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    user_experts = Experts(test_config)
    user_experts.eval()

    # Copy weights from HF to user's experts
    copy_experts_to_fused(hf_moe, user_experts)

    batch, seq, hidden = random_hidden_states.shape
    flat_input = random_hidden_states.view(-1, hidden)

    # Test each expert individually
    for expert_idx in range(test_config.num_experts):
        with torch.no_grad():
            # HF expert forward
            hf_expert_out = hf_moe.experts[expert_idx](flat_input)

            # User expert forward - send all tokens to one expert
            expert_tokens_count = torch.tensor([0] * expert_idx + [flat_input.shape[0]])
            # Pad counts for experts before this one
            expert_tokens_count = torch.tensor(
                [0] * expert_idx + [flat_input.shape[0]] + [flat_input.shape[0]] * (test_config.num_experts - expert_idx - 1)
            )
            # Actually we need cumulative counts, so:
            expert_tokens_count = torch.zeros(test_config.num_experts, dtype=torch.long)
            expert_tokens_count[expert_idx:] = flat_input.shape[0]

            user_expert_out = user_experts(flat_input, expert_tokens_count)

        assert torch.allclose(hf_expert_out, user_expert_out, rtol=RTOL, atol=ATOL), \
            f"Expert {expert_idx} output mismatch. Max diff: {(hf_expert_out - user_expert_out).abs().max()}"


def test_router_weights_vs_hf(test_config, random_hidden_states):
    """Test that router produces same top-k weights as HF."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    user_router = Router(test_config)
    user_router.eval()

    copy_router_weights(hf_moe, user_router)

    batch, seq, hidden = random_hidden_states.shape
    flat_input = random_hidden_states.view(-1, hidden)

    with torch.no_grad():
        # HF router computation (from their forward)
        hf_logits = hf_moe.gate(flat_input)
        hf_weights = torch.softmax(hf_logits, dim=-1)
        hf_topk_weights, hf_topk_indices = torch.topk(
            hf_weights, test_config.num_experts_per_tok, dim=-1
        )
        # HF normalizes top-k weights
        hf_topk_weights = hf_topk_weights / hf_topk_weights.sum(dim=-1, keepdim=True)

        # User router
        user_probs, user_map = user_router(random_hidden_states)
        user_probs_flat = user_probs.view(-1, test_config.num_experts)
        user_map_flat = user_map.view(-1, test_config.num_experts)

    # Check that same experts are selected (using routing_map, not re-running topk)
    hf_selected = torch.zeros_like(user_map_flat).scatter_(-1, hf_topk_indices, True)
    assert torch.equal(user_map_flat, hf_selected), \
        "Expert selection mismatch between HF and user router"

    # Extract selected weights and compare
    # User's probs are already normalized so selected ones should sum to 1
    user_selected_weights = user_probs_flat[user_map_flat]  # flat list of selected weights
    hf_selected_weights = hf_topk_weights.view(-1)  # flat list of HF top-k weights

    assert torch.allclose(user_selected_weights.sort().values, hf_selected_weights.sort().values, rtol=RTOL, atol=ATOL), \
        f"Top-k weights mismatch. Max diff: {(user_selected_weights.sort().values - hf_selected_weights.sort().values).abs().max()}"


def test_moe_output_vs_hf(test_config, random_hidden_states):
    """Test that full MOE output matches HuggingFace numerically."""
    hf_moe = Qwen3MoeSparseMoeBlock(test_config)
    hf_moe.eval()

    user_moe = MOE(test_config)
    user_moe.eval()

    copy_moe_weights(hf_moe, user_moe)

    batch, seq, hidden = random_hidden_states.shape

    with torch.no_grad():
        hf_out, hf_router_logits = hf_moe(random_hidden_states)
        user_out = user_moe(random_hidden_states)

    # Shape check
    assert user_out.shape == (batch * seq, hidden), \
        f"User MOE output shape mismatch: {user_out.shape} vs expected {(batch * seq, hidden)}"

    # Numerical comparison
    hf_out_flat = hf_out.view(-1, hidden)
    assert torch.allclose(hf_out_flat, user_out, rtol=RTOL, atol=ATOL), \
        f"MOE output mismatch. Max diff: {(hf_out_flat - user_out).abs().max():.6f}, " \
        f"Mean diff: {(hf_out_flat - user_out).abs().mean():.6f}"


def test_moe_output_shape(test_config, random_hidden_states):
    """Test that user's MOE output has correct shape."""
    user_moe = MOE(test_config)
    user_moe.eval()

    batch, seq, hidden = random_hidden_states.shape

    with torch.no_grad():
        user_out = user_moe(random_hidden_states)

    assert user_out.shape == (batch * seq, hidden), \
        f"User MOE output shape mismatch: {user_out.shape} vs expected {(batch * seq, hidden)}"


def test_moe_router_output_user(test_config, random_hidden_states):
    """Test that user's MOE router produces valid outputs."""
    user_moe = MOE(test_config)
    user_moe.eval()

    batch, seq, hidden = random_hidden_states.shape

    with torch.no_grad():
        routing_probs, routing_map = user_moe.router(random_hidden_states)

    # Router should produce finite outputs
    assert torch.isfinite(routing_probs).all(), "Routing probs should be finite"

    # Shape checks
    assert routing_probs.shape == (batch, seq, test_config.num_experts)
    assert routing_map.shape == (batch, seq, test_config.num_experts)


def test_moe_full_gradient(test_config, random_hidden_states):
    """Test that gradients flow through the full MOE forward pass."""
    user_moe = MOE(test_config)

    x = random_hidden_states.clone().requires_grad_(True)
    out = user_moe(x)
    out.sum().backward()

    assert x.grad is not None, "Gradient should flow through MOE input"
    assert user_moe.router.gate.weight.grad is not None, "Router should have gradient"
    assert user_moe.experts.gate_proj.grad is not None, "Expert gate_proj should have gradient"
    assert user_moe.experts.up_proj.grad is not None, "Expert up_proj should have gradient"
    assert user_moe.experts.down_proj.grad is not None, "Expert down_proj should have gradient"
