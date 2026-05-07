import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# ==========================================
# AICA Constraint Definition: Semantic Constants
# ==========================================
MAX_TRITON_BLOCK_D: int = 1024
MASS_NUMERICAL_EPSILON: float = 1e-6

# ==========================================
# 1. Forward Kernel: Fused Linear Recurrence
# ==========================================
@triton.jit
def _fused_recurrence_fwd_kernel(
    # Explicit Inputs
    f_proj_ptr,     # [Batch, Seq, D]
    mass_ptr,       # [Batch, Seq, 1]
    eps_ptr,        # [Batch, Seq, 1]
    gamma_ptr,      # [Batch, Seq, 1]
    h0_ptr,         # [Batch, D] (Explicit Initial State Contract)
    
    # Explicit Outputs
    m_out_ptr,      # [Batch, Seq, D] 
    
    # Strides
    stride_b, stride_s, stride_d,       
    stride_sb, stride_ss, stride_sd,    
    stride_h0_b, stride_h0_d,           
    
    # Constants (Injected to prevent Magic Numbers)
    MASS_EPS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_H0: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    batch_offset = pid_b * stride_b
    scalar_batch_offset = pid_b * stride_sb
    dim_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offset < stride_d 

    # AICA Fix: Explicit Initial State Handling (Eliminate Temporal Coupling)
    h_curr = tl.zeros([BLOCK_D], dtype=tl.float32)
    if HAS_H0:
        h0_offset = (pid_b * stride_h0_b) + dim_offset
        h_curr = tl.load(h0_ptr + h0_offset, mask=dim_mask, other=0.0)

    for t in range(SEQ_LEN):
        scalar_ptr_offset = scalar_batch_offset + t * stride_ss

        mass_val = tl.load(mass_ptr + scalar_ptr_offset)
        eps_val = tl.load(eps_ptr + scalar_ptr_offset)
        gamma_val = tl.load(gamma_ptr + scalar_ptr_offset)

        alpha = 1.0 - (eps_val * gamma_val)
        # AICA Fix: Use injected constexpr instead of magic number
        beta = eps_val / (mass_val + MASS_EPS)

        f_ptr = f_proj_ptr + batch_offset + (t * stride_s) + dim_offset
        f_val = tl.load(f_ptr, mask=dim_mask, other=0.0)

        h_curr = alpha * h_curr + beta * f_val

        out_ptr = m_out_ptr + batch_offset + (t * stride_s) + dim_offset
        tl.store(out_ptr, h_curr, mask=dim_mask)

# ==========================================
# 2. Backward Kernel: Fused Adjoint Recurrence
# ==========================================
@triton.jit
def _fused_recurrence_bwd_kernel(
    grad_out_ptr,   
    f_proj_ptr, mass_ptr, eps_ptr, gamma_ptr,
    m_out_ptr,      
    
    grad_f_ptr,     
    grad_mass_ptr,  
    grad_eps_ptr,   
    grad_gamma_ptr, 

    stride_b, stride_s, stride_d,
    stride_sb, stride_ss, stride_sd,

    MASS_EPS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_h_next = tl.zeros([BLOCK_D], dtype=tl.float32)

    batch_offset = pid_b * stride_b
    scalar_batch_offset = pid_b * stride_sb
    dim_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offset < stride_d

    for t in range(SEQ_LEN - 1, -1, -1):
        scalar_ptr_offset = scalar_batch_offset + t * stride_ss
        mass = tl.load(mass_ptr + scalar_ptr_offset)
        eps = tl.load(eps_ptr + scalar_ptr_offset)
        gamma = tl.load(gamma_ptr + scalar_ptr_offset)

        prev_h_val = tl.zeros([BLOCK_D], dtype=tl.float32)
        if t > 0:
            m_prev_ptr = m_out_ptr + batch_offset + ((t-1) * stride_s) + dim_offset
            prev_h_val = tl.load(m_prev_ptr, mask=dim_mask)

        f_ptr = f_proj_ptr + batch_offset + (t * stride_s) + dim_offset
        f_val = tl.load(f_ptr, mask=dim_mask)

        grad_out_ptr_t = grad_out_ptr + batch_offset + (t * stride_s) + dim_offset
        d_m_curr = tl.load(grad_out_ptr_t, mask=dim_mask)

        d_h_total = d_m_curr + d_h_next

        alpha = 1.0 - (eps * gamma)
        beta = eps / (mass + MASS_EPS)

        d_f = d_h_total * beta
        tl.store(grad_f_ptr + batch_offset + (t*stride_s) + dim_offset, d_f, mask=dim_mask)

        d_h_next = d_h_total * alpha

        d_alpha = tl.sum(d_h_total * prev_h_val, axis=0)
        d_beta = tl.sum(d_h_total * f_val, axis=0)

        # AICA Fix: Strict bounds via injected MASS_EPS
        safe_mass = mass + MASS_EPS
        d_eps_val = d_alpha * (-gamma) + d_beta * (1.0 / safe_mass)
        d_gamma_val = d_alpha * (-eps)
        d_mass_val = d_beta * (-eps / (safe_mass * safe_mass))

        # AICA Safety Constraint: We assume 1 block processes the full D dimension.
        # If multiple blocks process D, this tl.store requires tl.atomic_add to avoid data races.
        # (The python layer assertion guarantees grid=(B,1), making tl.store safe here).
        tl.store(grad_eps_ptr + scalar_ptr_offset, d_eps_val)
        tl.store(grad_gamma_ptr + scalar_ptr_offset, d_gamma_val)
        tl.store(grad_mass_ptr + scalar_ptr_offset, d_mass_val)

# ==========================================
# 3. PyTorch Autograd Interface & AICA Contracts
# ==========================================
class FusedRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_proj: torch.Tensor, mass: torch.Tensor, epsilon: torch.Tensor, gamma: torch.Tensor, h0: Optional[torch.Tensor] = None):
        B, S, D = f_proj.shape
        
        # AICA Assert: Prevent Implicit Hardware/Algorithm Divergence
        if D > MAX_TRITON_BLOCK_D:
            raise NotImplementedError(f"AICA Guard: Dimension D ({D}) exceeds supported MAX_TRITON_BLOCK_D ({MAX_TRITON_BLOCK_D}). Block loop expansion is strictly required to prevent silent truncation.")

        m_out = torch.empty_like(f_proj)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, 1)

        has_h0 = h0 is not None
        h0_stride_b = h0.stride(0) if has_h0 else 0
        h0_stride_d = h0.stride(1) if has_h0 else 0

        _fused_recurrence_fwd_kernel[grid](
            f_proj, mass, epsilon, gamma, h0,
            m_out,
            f_proj.stride(0), f_proj.stride(1), f_proj.stride(2),
            mass.stride(0), mass.stride(1), mass.stride(2),
            h0_stride_b, h0_stride_d,
            MASS_EPS=MASS_NUMERICAL_EPSILON,
            SEQ_LEN=S, BLOCK_D=BLOCK_D,
            HAS_H0=has_h0,
            num_warps=4
        )

        ctx.save_for_backward(f_proj, mass, epsilon, gamma, m_out)
        return m_out

    @staticmethod
    def backward(ctx, grad_out):
        f_proj, mass, epsilon, gamma, m_out = ctx.saved_tensors
        B, S, D = f_proj.shape

        grad_f = torch.empty_like(f_proj)
        grad_mass = torch.empty_like(mass)
        grad_eps = torch.empty_like(epsilon)
        grad_gamma = torch.empty_like(gamma)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, 1)

        _fused_recurrence_bwd_kernel[grid](
            grad_out,
            f_proj, mass, epsilon, gamma, m_out,
            grad_f, grad_mass, grad_eps, grad_gamma,
            f_proj.stride(0), f_proj.stride(1), f_proj.stride(2),
            mass.stride(0), mass.stride(1), mass.stride(2),
            MASS_EPS=MASS_NUMERICAL_EPSILON,
            SEQ_LEN=S, BLOCK_D=BLOCK_D
        )

        # Note: grad for h0 is currently None in this simplified kernel.
        return grad_f, grad_mass, grad_eps, grad_gamma, None


def fused_agi_update(
    f_proj: torch.Tensor, 
    mass: torch.Tensor, 
    epsilon: torch.Tensor, 
    gamma: torch.Tensor,
    past_momentum: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Standard Interface with Strict AICA Contract Enforcement.
    
    Preconditions:
      - f_proj: Tensor[B, S, D]
      - mass, epsilon, gamma: Tensor[B, S, 1]
    """
    # AICA Fix: Type Collapse Prevention. 
    # Reject implicit `.unsqueeze(-1)` coercions. Callers must supply explicit [B, S, 1] shapes.
    assert f_proj.dim() == 3, f"Contract Violation: f_proj must be 3D, got {f_proj.shape}"
    assert mass.dim() == 3 and mass.size(-1) == 1, f"Contract Violation: mass must be [B, S, 1], got {mass.shape}"
    assert epsilon.dim() == 3 and epsilon.size(-1) == 1, f"Contract Violation: epsilon must be [B, S, 1], got {epsilon.shape}"
    assert gamma.dim() == 3 and gamma.size(-1) == 1, f"Contract Violation: gamma must be [B, S, 1], got {gamma.shape}"
    
    if past_momentum is not None:
        assert past_momentum.dim() == 2 and past_momentum.size(-1) == f_proj.size(-1), f"Contract Violation: past_momentum must be [B, D]."

    return FusedRecurrenceFunction.apply(f_proj, mass, epsilon, gamma, past_momentum)