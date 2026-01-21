import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import HISEConfig
from ..thermodynamics.mass_dynamics import CognitiveGearbox
from ..kernels.triton_physics import fused_agi_update


class HamiltonianAttention(nn.Module):
    """
    Computes the Conservative Force Field (-grad V) from LogSumExp potential.
    Uses standard Attention mechanism but interprets outputs as Physical Forces.
    """
    def __init__(self, config: HISEConfig):
        super().__init__()
        self.head_dim = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.tau = config.tau # Thermodynamic Temperature
        
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model)


    def forward(self, h, mask=None):
        B, T, C = h.size()
        q = self.w_q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)


        # Potential Gradient Calculation (Hamiltonian Dynamics)
        # Scaled Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        
        # Apply Temperature (Annealing)
        scores = scores / self.tau 


        if mask is not None:
            scores = scores + mask


        attn_weights = torch.softmax(scores, dim=-1)
        
        # Force Accumulation
        # In HISE, the attention output is interpreted as the "Force" exerted 
        # by past tokens on the current token.
        force = attn_weights @ v 
        force = force.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(force)


class SoftTCMLayer(nn.Module):
    """
    SR-TCM Layer v2 (Symplectic Recurrent Low-Rank Manifold).
    Integrates Low-Rank Symplectic Dynamics with Cognitive Gearbox (System 1/2).
    """
    def __init__(self, config: HISEConfig):
        super().__init__()
        self.config = config
        self.force_field = HamiltonianAttention(config)
        self.norm = nn.LayerNorm(config.d_model)


        # Low-Rank Manifold Projectors (U: Down to Manifold, V: Up to State Space)
        # Reduces dimensionality for efficient symplectic integration
        self.U = nn.Linear(config.d_model, config.d_inertial, bias=False)
        self.V = nn.Linear(config.d_inertial, config.d_model, bias=False)


        # Dissipation Control (Friction)
        self.gamma_net = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        # AGI Core: Cognitive Gearbox
        # Dynamically calculates Mass and Time-step (epsilon)
        self.gearbox = CognitiveGearbox(config)
        
        # System 2 Logic Gate (Spectral-Riemannian Coupling)
        # Only activated when Mass > Threshold (Heavy Thought Mode)
        self.logic_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        
        # Standard Drift / Mixing Layer
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )


    def forward(self, h, mask=None, past_momentum=None):
        """
        Symplectic Forward Pass with Differentiable Physics Metrics.
        """
        h_norm = self.norm(h)
        
        # 1. Total Force Calculation
        f_attn = self.force_field(h_norm, mask)
        f_conf = -self.config.lambda_conf * h_norm # Harmonic Confinement Force
        f_total = f_attn + f_conf
        
        # 2. Projection to Inertial Manifold (Phase Space)
        f_proj = self.U(f_total) 
        
        # 3. FSI Calculation (Fisher Semantic Information)
        # [CRITICAL FIX] Removed 'with torch.no_grad():'
        # Now FSI is part of the computational graph, allowing the Physics Loss
        # to backpropagate gradients to 'h' and 'f_proj'.
        h_mag = torch.norm(h_norm, p=2, dim=-1)
        f_mag = torch.norm(f_proj, p=2, dim=-1)
        
        # FSI = Signal / (Noise + epsilon)
        # Represents the ratio of State Magnitude to Force Magnitude
        fsi = h_mag / (2 * f_mag + 1e-6)


        # 4. Cognitive Gearbox (Determine Mass & Epsilon)
        # Mass is now dependent on differentiable FSI
        mass_t, epsilon_t, is_system_2 = self.gearbox(h, fsi)
        gamma = self.gamma_net(h_norm)


        if past_momentum is None:
            # Initialize momentum as zero (Cold Start)
            past_momentum = torch.zeros_like(f_proj)


        # 5. Fused Symplectic Update (via Triton)
        # Updates momentum considering variable Mass
        # This calls the Autograd-supported kernel we fixed in Step 1
        m_new = fused_agi_update(past_momentum, f_proj, mass_t, epsilon_t, gamma)
        
        # 6. Velocity Injection & Position Update
        # q_{t+1} = q_t + epsilon * (V @ m_{t+1})
        velocity = self.V(m_new)
        h_new = h + epsilon_t * velocity
        
        # 7. Spectral Coupling (System 2 Logic Injection)
        # Soft Gating: Applies logic correction based on Mass excess
        sys2_correction = self.logic_gate(self.norm(h_new))
        
        # Differentiable Gate: Allows learning the threshold behavior
        gate = torch.sigmoid(mass_t - self.config.system2_threshold) 
        h_final = h_new + gate * sys2_correction
        
        # 8. Auxiliary Drift (Standard ResNet connection)
        h_final = h_final + self.mlp(self.norm(h_final))


        return h_final, m_new, fsi
