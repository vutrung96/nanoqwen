"""
Shared utilities for nanomoe testing harness.
"""

import torch


def copy_weights(hf_module, your_module, mapping: dict[str, str]):
    """
    Copy weights from HF module to your module using explicit mapping.

    Args:
        hf_module: HuggingFace module
        your_module: Your from-scratch module
        mapping: Dict mapping HF param names -> your param names
                 e.g. {"gate_proj.weight": "w_gate.weight", ...}
    """
    hf_state = hf_module.state_dict()
    your_state = your_module.state_dict()

    for hf_name, your_name in mapping.items():
        if hf_name in hf_state:
            your_state[your_name] = hf_state[hf_name]

    your_module.load_state_dict(your_state)


# Tolerances for comparisons
RTOL = 1e-4
ATOL = 1e-5


def make_causal_mask(seq_len: int, batch_size: int) -> torch.Tensor:
    """Create 4D causal mask for HF attention (float, -inf for masked)."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)


def make_causal_mask_bool(seq_len: int) -> torch.Tensor:
    """Create 2D boolean causal mask for Block (True = masked out)."""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


def copy_expert_weights(hf_moe, user_experts, num_experts: int):
    """
    Copy HF expert weights (ModuleList) to user's fused 3D tensors.

    HF: experts[i].{gate,up,down}_proj.weight is (out_features, in_features)
    User: {gate,up,down}_proj is (num_experts, in_features, out_features)
    """
    with torch.no_grad():
        for i in range(num_experts):
            hf_expert = hf_moe.experts[i]
            user_experts.gate_proj.data[i] = hf_expert.gate_proj.weight.T
            user_experts.up_proj.data[i] = hf_expert.up_proj.weight.T
            user_experts.down_proj.data[i] = hf_expert.down_proj.weight.T
