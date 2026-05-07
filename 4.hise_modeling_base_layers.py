import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..config import HISEConfig
from ..thermodynamics.mass_dynamics import CognitiveGearbox
from ..kernels.triton_physics import fused_agi_update

# ==========================================
# AICA Constraint Definition: Physical & Numerical Constants
# ==========================================
FSI_NOISE_FLOOR: float = 1e-6
FSI_SIGNAL_SCALE: float = 2.0
MASS_NUMERICAL_EPSILON: float = 1e-6


class HamiltonianAttention(nn.Module):
    """
    Computes the Conservative Force Field (-grad V) from LogSumExp potential.
    Uses standard Attention mechanism but interprets outputs as Physical Forces.
    
    AICA Asserts:
        - Replaced magic scaling (0.5) with strict math.sqrt evaluation.
        - Strict type hinting for inputs and masks.
    """
    def __init__(self, config: HISEConfig) -> None:
        super().__init__()
        self.head_dim = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.tau = config.tau # Thermodynamic Temperature
        
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # AICA Fix: Explicit scaling constant to prevent semantic collapse
        self.scale_factor = math.sqrt(self.head_dim)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = h.size()
        q = self.w_q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Potential Gradient Calculation (Hamiltonian Dynamics)
        scores = (q @ k.transpose(-2, -1)) / self.scale_factor
        
        # Apply Temperature (Annealing)
        scores = scores / self.tau 

        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        
        # Force Accumulation
        force = attn_weights @ v 
        force = force.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(force)


class SoftTCMLayer(nn.Module):
    """
    SR-TCM Layer v2 (Symplectic Recurrent Low-Rank Manifold).
    Integrates Low-Rank Symplectic Dynamics with Cognitive Gearbox.
    
    AICA Asserts:
        - Strict topology bounds (unsqueeze explicit dimensions).
        - Explicit Dependency Injection for Temporal States (past_momentum).
    """
    def __init__(self, config: HISEConfig) -> None:
        super().__init__()
        self.config = config
        self.force_field = HamiltonianAttention(config)
        self.norm = nn.LayerNorm(config.d_model)

        # Low-Rank Manifold Projectors
        self.U = nn.Linear(config.d_model, config.d_inertial, bias=False)
        self.V = nn.Linear(config.d_inertial, config.d_model, bias=False)

        # Dissipation Control (Friction)
        self.gamma_net = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        self.gearbox = CognitiveGearbox(config)
        
        # System 2 Logic Gate
        self.logic_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )

    def forward(
        self, 
        h: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        past_momentum: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_norm = self.norm(h)
        B, T, C = h.shape
        
        # 1. Total Force Calculation
        f_attn = self.force_field(h_norm, mask)
        f_conf = -self.config.lambda_conf * h_norm 
        f_total = f_attn + f_conf
        
        # 2. Projection to Inertial Manifold
        f_proj = self.U(f_total) 
        
        # 3. FSI Calculation
        h_mag = torch.norm(h_norm, p=2, dim=-1)
        f_mag = torch.norm(f_proj, p=2, dim=-1)
        
        # AICA Fix: Inject explicit constants
        fsi_raw = h_mag / (FSI_SIGNAL_SCALE * f_mag + FSI_NOISE_FLOOR)
        
        # AICA Strict Contract: Prevent Implicit Broadcasting (Type Collapse)
        # Force FSI to be [Batch, Seq, 1] BEFORE passing to gearbox or physics kernel
        fsi = fsi_raw.unsqueeze(-1) 

        # 4. Cognitive Gearbox
        mass_t, epsilon_t, is_system_2 = self.gearbox(h, fsi)
        gamma = self.gamma_net(h_norm)

        # 5. Symplectic Integration
        # AICA Guard: Differentiate clearly between cached step and full sequence
        is_single_step = (T == 1) and (past_momentum is not None)

        if is_single_step:
            # === Inference Mode (Cached, Single Step) ===
            eps_t = epsilon_t[:, -1:, :]
            mass_t_slice = mass_t[:, -1:, :]
            gam_t = gamma[:, -1:, :]
            
            alpha = 1.0 - (eps_t * gam_t)
            beta = eps_t / (mass_t_slice + MASS_NUMERICAL_EPSILON)
            
            m_new = alpha * past_momentum + beta * f_proj
        
        else:
            # === Training Mode / Pre-fill Mode (Fused Scan) ===
            # AICA Fix: Adhere to the strict positional argument contract of fused_agi_update
            m_new = fused_agi_update(f_proj, mass_t, epsilon_t, gamma, past_momentum)
        
        # 6. Velocity Injection & Position Update
        velocity = self.V(m_new)
        h_new = h + epsilon_t * velocity
        
        # 7. Spectral Coupling
        sys2_correction = self.logic_gate(self.norm(h_new))
        gate = torch.sigmoid(mass_t - self.config.system2_threshold) 
        h_final = h_new + gate * sys2_correction
        
        # 8. Auxiliary Drift
        h_final = h_final + self.mlp(self.norm(h_final))

        return h_final, m_new, fsi
