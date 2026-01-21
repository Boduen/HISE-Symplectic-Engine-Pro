import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import HISEConfig


class CognitiveGearbox(nn.Module):
    """
    Implements the 'Mass-Entropy' equivalence from Projective Spectral Dynamics (PSD).
    Dynamically calculates semantic mass M(t) and adjusts time-step epsilon(t).
    
    [FIXED VERSION]: Replaced hard clamps with Softplus/Tanh for gradient stability.
    """
    def __init__(self, config: HISEConfig):
        super().__init__()
        self.base_epsilon = config.epsilon
        self.min_epsilon_scale = config.min_epsilon_scale
        self.threshold = config.system2_threshold
        
        # Learnable Boltzmann Constant equivalent for information
        # Corresponds to k_b in PSD theory
        self.k_b = nn.Parameter(torch.tensor(1.0)) 


    def compute_local_entropy(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Dim] -> Entropy: [Batch, Seq, 1]
        # Calculates Shannon entropy of the local semantic distribution
        # Added small epsilon to log to prevent NaN
        prob = F.softmax(x, dim=-1)
        log_prob = torch.log(prob + 1e-9) 
        entropy = -(prob * log_prob).sum(dim=-1, keepdim=True)
        return entropy


    def derive_mass(self, entropy: torch.Tensor, fsi_score: torch.Tensor) -> torch.Tensor:
        """
        Derives Mass M(t) based on Complexity Cost.
        Formula: M ~ sqrt(k_b * H * ln(2))
        """
        # Ensure k_b stays positive using Softplus
        kb_safe = F.softplus(self.k_b)
        
        # Complexity Cost
        complexity_cost = kb_safe * entropy * 0.693 # ln(2) approx
        
        # Base Mass from Complexity
        # [FIX] Replaced sqrt(clamp) with Softplus. 
        # Softplus(x) approx log(1+e^x), behaves like ReLU but smooth.
        # Added +1e-4 to strictly prevent Zero Mass (which causes division by zero in physics kernel).
        mass_base = F.softplus(complexity_cost) + 1e-4
        
        # FSI Modulation: High risk (low FSI) drastically increases Mass (Inertia)
        # to prevent hallucination (Axiom Smuggling).
        
        # [FIX] Smooth Safety Factor
        # Instead of hard clamping 1/FSI, we use a smooth Tanh saturation.
        # Logic: If FSI is close to 0, factor approaches max_scale.
        # 1.0 / (fsi + 1e-2) prevents infinity.
        inv_fsi = 1.0 / (fsi_score + 1e-2)
        
        # Soft cap at 10.0: 10 * tanh(x/10) + 1.0
        # This provides a smooth gradient towards the safety limit.
        safety_factor = 10.0 * torch.tanh(inv_fsi / 10.0) + 1.0
        
        mass_dynamic = mass_base * safety_factor
        return mass_dynamic


    def forward(self, h: torch.Tensor, fsi_score: torch.Tensor):
        entropy = self.compute_local_entropy(h)
        mass_t = self.derive_mass(entropy, fsi_score)
        
        # Determine System Mode (Soft Logic for Training)
        # Instead of a boolean hard switch, we calculate a probability gate
        # This allows gradients to flow through the threshold boundary.
        # Sigmoid steepness (temperature) = 5.0
        system2_prob = torch.sigmoid((mass_t - self.threshold) * 5.0)
        
        # Boolean flag for inference logic (logic_gate activation)
        is_system_2 = mass_t > self.threshold
        
        # Adaptive Epsilon (Step Size)
        # System 1 -> Base Epsilon (Fast)
        # System 2 -> Min Epsilon (Slow, fine-grained)
        
        eps_fast = torch.tensor(self.base_epsilon, device=h.device)
        eps_slow = torch.tensor(self.base_epsilon * self.min_epsilon_scale, device=h.device)
        
        # [FIX] Soft Interpolation instead of torch.where
        # epsilon = (1 - p) * Fast + p * Slow
        # This allows the model to learn "how fast" to move based on Mass.
        epsilon_t = (1.0 - system2_prob) * eps_fast + system2_prob * eps_slow
        
        return mass_t, epsilon_t, is_system_2
