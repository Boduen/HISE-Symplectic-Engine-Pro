import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from ..config import HISEConfig

# ==========================================
# AICA Constraint Definition: Semantic Constants
# ==========================================
class MoEConstants:
    MLP_EXPANSION_FACTOR: int = 4
    MIN_MASS_EPSILON: float = 1e-6


class PhysicsRouter(nn.Module):
    """
    Implements 'Thermodynamic Routing' with Load Balancing.
    
    AICA Asserts:
        - Strict adherence to config contracts (num_experts_per_tok).
        - Explicit dimensional asserts to prevent implicit broadcasting.
    """
    def __init__(self, config: HISEConfig) -> None:
        super().__init__()
        # AICA Precondition: Prevent invalid topologies
        assert config.num_experts > 0, "Contract Violation: num_experts must be positive."
        assert config.num_experts_per_tok > 0, "Contract Violation: num_experts_per_tok must be positive."
        assert config.num_experts_per_tok <= config.num_experts, "Contract Violation: top_k cannot exceed num_experts."

        self.d_model: int = config.d_model
        self.num_experts: int = config.num_experts
        self.top_k: int = config.num_experts_per_tok

        # Router Gate: Projects input to expert logits
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)

        # Learnable Physics Bias: Sensitivity to Semantic Mass
        self.mass_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mass: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [Batch, Seq, Dim]
            mass: [Batch, Seq, 1]
        """
        # AICA Guard: Strict Topology Contracts
        assert hidden_states.dim() == 3, f"Contract Violation: hidden_states must be 3D, got {hidden_states.shape}"
        assert mass.dim() == 3 and mass.size(-1) == 1, f"Contract Violation: mass must be [B, S, 1], got {mass.shape}"

        B, T, C = hidden_states.shape
        flat_hidden = hidden_states.view(-1, C)
        flat_mass = mass.view(-1, 1)

        # 1. Base Routing Logits
        logits = self.gate(flat_hidden)

        # 2. Apply Differentiable Physics Bias
        physics_impact = flat_mass * self.mass_bias.unsqueeze(0)
        logits = logits + physics_impact

        # 3. Load Balancing & Top-K Selection
        probs = F.softmax(logits, dim=-1)
        
        # AICA Fix: Utilize config-injected top_k instead of magic number
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Aux Loss: Mean Squared Importance (Encourages Uniformity)
        expert_importance = probs.sum(dim=0)
        target_load = float(probs.size(0)) / self.num_experts
        aux_loss = ((expert_importance - target_load) ** 2).mean()

        return logits, top_k_indices, aux_loss, top_k_weights


class MoPEBlock(nn.Module):
    """
    A Mixture-of-Experts block where experts are specialized Physics Engines.
    
    AICA Asserts:
        - Deterministic Gradient Accumulation (Avoids raw in-place indexing where possible).
        - Semantic constants extracted to MoEConstants.
    """
    def __init__(self, config: HISEConfig) -> None:
        super().__init__()
        self.config: HISEConfig = config
        self.num_experts: int = config.num_experts
        
        self.router = PhysicsRouter(config)

        # AICA Fix: Replaced magic number '4' with explicit ML_EXPANSION_FACTOR
        expanded_dim = MoEConstants.MLP_EXPANSION_FACTOR * config.d_model
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, expanded_dim),
                nn.GELU(),
                nn.Linear(expanded_dim, config.d_model)
            ) for _ in range(self.num_experts)
        ])

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mass: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, T, C = hidden_states.shape
        flat_hidden = hidden_states.view(-1, C)

        # 1. Route
        logits, indices, aux_loss, weights = self.router(hidden_states, mass)

        # 2. Dispatch & Execute
        # AICA Fix: Use explicit cloning/zeroing to ensure a clean deterministic computational graph
        results = torch.zeros_like(flat_hidden)

        # Loop over the allowed Top-K choices defined in config
        for k in range(self.config.num_experts_per_tok):
            expert_idx = indices[:, k] 
            w = weights[:, k].unsqueeze(-1) 

            for e_idx, expert in enumerate(self.experts):
                token_mask = (expert_idx == e_idx)

                if token_mask.any():
                    # Extract inputs for this expert
                    inp = flat_hidden[token_mask]
                    out = expert(inp)
                    
                    # Weight the output
                    weighted_out = out * w[token_mask]
                    
                    # AICA Safety Constraint: To strictly prevent non-deterministic in-place accumulation 
                    # during backward pass (scatter reduce), we use the masked accumulation strictly.
                    # In production Triton kernels, this is handled via atomic adds.
                    results[token_mask] = results[token_mask] + weighted_out

        return results.view(B, T, C), aux_loss
