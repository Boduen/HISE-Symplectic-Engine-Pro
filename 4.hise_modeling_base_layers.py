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
        Handles both Inference (Cached) and Training (Sequential Scan).
        """
        h_norm = self.norm(h)
        B, T, C = h.shape
        
        # 1. Total Force Calculation
        f_attn = self.force_field(h_norm, mask)
        f_conf = -self.config.lambda_conf * h_norm # Harmonic Confinement Force
        f_total = f_attn + f_conf
        
        # 2. Projection to Inertial Manifold (Phase Space)
        f_proj = self.U(f_total) 
        
        # 3. FSI Calculation (Differentiable!)
        # Now part of the computational graph, enabling Physics Loss backprop.
        h_mag = torch.norm(h_norm, p=2, dim=-1)
        f_mag = torch.norm(f_proj, p=2, dim=-1)
        
        # FSI = Signal / (Noise + epsilon)
        fsi = h_mag / (2 * f_mag + 1e-6)

        # 4. Cognitive Gearbox (Determine Mass & Epsilon)
        mass_t, epsilon_t, is_system_2 = self.gearbox(h, fsi)
        gamma = self.gamma_net(h_norm)

        # 5. Symplectic Integration (Momentum Update)
        if past_momentum is not None:
            # === Inference Mode (Cached) ===
            # Simply update the state from the previous step
            m_new = fused_agi_update(past_momentum, f_proj, mass_t, epsilon_t, gamma)
        
        else:
            # === Training Mode (Sequential Scan) ===
            # We must simulate the time evolution step-by-step to build memory.
            # Initialize momentum (Cold Start)
            m_curr = torch.zeros(B, 1, self.config.d_inertial, device=h.device, dtype=h.dtype)
            m_outputs = []
            
            # Recurrent Loop (Python-level scan)
            # Note: For production optimization, this should be replaced by a 
            # Parallel Scan (Prefix Sum) kernel in Triton.
            for t in range(T):
                # Slice current time step inputs [Batch, 1, Dim]
                f_t = f_proj[:, t:t+1, :]
                mass_t_step = mass_t[:, t:t+1, :]
                eps_t_step = epsilon_t[:, t:t+1, :]
                gamma_t = gamma[:, t:t+1, :]
                
                # Update Momentum: m_t = Step(m_{t-1}, Inputs_t)
                m_curr = fused_agi_update(m_curr, f_t, mass_t_step, eps_t_step, gamma_t)
                m_outputs.append(m_curr)
            
            # Concatenate all time steps back to [Batch, Seq, Dim]
            m_new = torch.cat(m_outputs, dim=1)
        
        # 6. Velocity Injection & Position Update
        # q_{t+1} = q_t + epsilon * (V @ m_{t+1})
        velocity = self.V(m_new)
        h_new = h + epsilon_t * velocity
        
        # 7. Spectral Coupling (System 2 Logic Injection)
        # Soft Gating: Applies logic correction based on Mass excess
        sys2_correction = self.logic_gate(self.norm(h_new))
        
        # Differentiable Gate
        gate = torch.sigmoid(mass_t - self.config.system2_threshold) 
        h_final = h_new + gate * sys2_correction
        
        # 8. Auxiliary Drift (ResNet connection)
        h_final = h_final + self.mlp(self.norm(h_final))

        return h_final, m_new, fsi
