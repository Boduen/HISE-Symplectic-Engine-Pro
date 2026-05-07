import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..config import HISEConfig


class CognitiveGearbox(nn.Module):
    """
    Implements the 'Mass-Entropy' equivalence from Projective Spectral Dynamics (PSD).
    Dynamically calculates semantic mass M(t) and adjusts time-step epsilon(t).
    
    AICA Asserts:
        - Pure Function Forward Pass: No dynamic tensor allocation inside forward().
        - Strict Typing: Explicit dtypes and device handling to prevent EISV.
        - Semantic Constants: Magic numbers are documented or calculated explicitly.
    """
    # 物理與數值穩定性常數 (明確宣告，避免語意坍縮)
    LN_2: float = math.log(2.0)
    ENTROPY_EPSILON: float = 1e-9
    MASS_MIN_OFFSET: float = 1e-4
    FSI_EPSILON: float = 1e-2

    def __init__(self, config: HISEConfig) -> None:
        super().__init__()
        # 確保參數型別為嚴格的 float，避免與 Tensor 運算時產生不預期的 casting
        self.base_epsilon: float = float(config.epsilon)
        self.min_epsilon_scale: float = float(config.min_epsilon_scale)
        self.threshold: float = float(config.system2_threshold)

        # AICA Fix: 明確指定 dtype，阻斷對 PyTorch 全域預設狀態的隱式依賴
        self.k_b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) 

    def compute_local_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates Shannon entropy of the local semantic distribution.
        
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Seq, Dim]
            
        Returns:
            torch.Tensor: Local entropy of shape [Batch, Seq, 1]
        """
        prob = F.softmax(x, dim=-1)
        # AICA Fix: 使用類別常數取代 hardcoded 魔術數字
        log_prob = torch.log(prob + self.ENTROPY_EPSILON) 
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
        # AICA Fix: 替換 0.693 為精確計算的 LN_2 常數
        complexity_cost = kb_safe * entropy * self.LN_2 

        # Base Mass from Complexity
        mass_base = F.softplus(complexity_cost) + self.MASS_MIN_OFFSET

        # FSI Modulation: High risk (low FSI) drastically increases Mass (Inertia)
        inv_fsi = 1.0 / (fsi_score + self.FSI_EPSILON)

        # Soft cap at 10.0: 10 * tanh(x/10) + 1.0
        safety_factor = 10.0 * torch.tanh(inv_fsi / 10.0) + 1.0

        mass_dynamic = mass_base * safety_factor
        return mass_dynamic

    def forward(
        self, 
        h: torch.Tensor, 
        fsi_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for cognitive gearbox routing.
        
        Args:
            h (torch.Tensor): Hidden states from the Hamiltonian Manifold.
            fsi_score (torch.Tensor): Falsification Score Index for hallucination check.
            
        Returns:
            Tuple containing:
                - mass_t (torch.Tensor): Dynamically calculated mass.
                - epsilon_t (torch.Tensor): Adaptive step size.
                - is_system_2 (torch.Tensor): Boolean tensor mask for System 2 routing.
        """
        entropy = self.compute_local_entropy(h)
        mass_t = self.derive_mass(entropy, fsi_score)

        # Determine System Mode (Soft Logic for Training)
        system2_prob = torch.sigmoid((mass_t - self.threshold) * 5.0)

        # Boolean flag for inference logic (logic_gate activation)
        is_system_2 = mass_t > self.threshold

        # AICA Fix: 解除 Device Allocation Coupling
        # 移除 torch.tensor(..., device=h.device)，改為直接利用 PyTorch 的純量廣播機制 (Scalar Broadcasting)。
        # 這樣不僅提升效能，也排除了對當前硬體狀態 (Device State) 的隱式依賴。
        eps_slow_scalar = self.base_epsilon * self.min_epsilon_scale
        
        epsilon_t = (1.0 - system2_prob) * self.base_epsilon + system2_prob * eps_slow_scalar

        return mass_t, epsilon_t, is_system_2