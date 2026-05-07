import math
import torch
import torch.nn as nn
from typing import Dict, List, Any

# ==========================================
# AICA Constraint Definition: Thermodynamic Constants
# ==========================================
class ThermoPhaseConstants:
    COOLING_EPOCHS: float = 10.0
    TURBULENCE_GRAD_NORM_THRESHOLD: float = 1.0
    OVERDAMPED_FRICTION_SCALE: float = 1.2
    UNDERDAMPED_FRICTION_SCALE: float = 0.9
    MIN_MASS_FLOOR: float = 1.0


class ThermodynamicScheduler:
    """
    Manages the thermodynamic state of the HISE Physics Engine during training.
    
    AICA Asserts:
        - Strict Registry Pattern: Eliminates duck-typing (hasattr) environment sniffing.
        - Dependency Injection: `avg_dynamic_mass` must be explicitly passed into `step()`.
        - Elimination of Magic Numbers via ThermoPhaseConstants.
        - Pure PyTorch/Math standard library (No numpy cross-ecosystem calls).
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        base_tau: float = 1.0, 
        min_tau: float = 0.1
    ) -> None:
        
        assert base_tau >= min_tau > 0.0, "Contract Violation: Temperature parameters must be strictly positive."
        
        self.optimizer: torch.optim.Optimizer = optimizer
        self.base_tau: float = base_tau
        self.min_tau: float = min_tau
        self.current_step: int = 0

        # Track initial LRs for scaling
        self.base_lrs: List[float] = [group['lr'] for group in optimizer.param_groups]
        
        # AICA Strict Contract: explicit registry for Temperature-dependent modules
        # Prevents EISV via runtime attribute sniffing (hasattr).
        self._tau_modules_registry: List[Any] = self._build_tau_registry(model)

    def _build_tau_registry(self, model: nn.Module) -> List[Any]:
        """Statically builds a registry of modules that explicitly support temperature scaling."""
        registry = []
        for module in model.modules():
            # AICA Guard: Only register specific known classes or explicit Protocols
            # Assuming HamiltonianAttention is the known layer from previous context
            if module.__class__.__name__ == 'HamiltonianAttention':
                registry.append(module)
        return registry

    def step(
        self, 
        epoch: float, 
        loss_val: float, 
        gradient_norm: float, 
        avg_dynamic_mass: float  # AICA Contract: Explicit Dependency Injection
    ) -> Dict[str, float]:
        """
        Adjusts physical constants based on energy landscape topology.
        
        Preconditions:
            - avg_dynamic_mass MUST be provided by the forward pass reduction, 
              preventing Temporal State Coupling.
        """
        self.current_step += 1

        # 1. Temperature Annealing (Simulated Cooling)
        decay_progress: float = min(1.0, epoch / ThermoPhaseConstants.COOLING_EPOCHS)
        new_tau: float = self.base_tau - (self.base_tau - self.min_tau) * decay_progress

        # Apply new Tau strictly via the pre-built registry
        for module in self._tau_modules_registry:
            module.tau = new_tau

        # 2. Dynamic Friction (Gamma) Adjustment
        if gradient_norm > ThermoPhaseConstants.TURBULENCE_GRAD_NORM_THRESHOLD:
            friction_scaling: float = ThermoPhaseConstants.OVERDAMPED_FRICTION_SCALE
        else:
            friction_scaling: float = ThermoPhaseConstants.UNDERDAMPED_FRICTION_SCALE

        # 3. Inverse-Mass LR Scaling (SR-TCM Theory)
        # We scale the learning rate proportional to sqrt(mass)
        # Using math.sqrt instead of numpy to prevent cross-ecosystem type coercion.
        safe_mass: float = max(ThermoPhaseConstants.MIN_MASS_FLOOR, avg_dynamic_mass)
        mass_factor: float = math.sqrt(safe_mass)

        # Apply strictly bounded LR scaling
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * mass_factor

        return {
            "tau": new_tau,
            "friction_scale": friction_scaling,
            "global_mass": safe_mass,
            "lr_scale": mass_factor
        }
