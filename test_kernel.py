import torch
from hise.kernels.triton_physics import fused_agi_update

def test_kernel():
    print("[Testing] Triton Fused Recurrent Kernel...")
    if not torch.cuda.is_available():
        print("[SKIP] Skipped: CUDA not available.")
        return

    device = "cuda"
    B, S, D = 2, 128, 64
    
    # Mock Inputs
    f_proj = torch.randn(B, S, D, device=device)
    mass = torch.rand(B, S, 1, device=device) + 0.1
    epsilon = torch.ones(B, S, 1, device=device) * 0.1
    gamma = torch.rand(B, S, 1, device=device)
    
    # 1. Forward Pass Test
    try:
        m_out = fused_agi_update(None, f_proj, mass, epsilon, gamma)
        print(f"[PASS] Forward Pass Successful. Output shape: {m_out.shape}")
    except Exception as e:
        print(f"[FAIL] Forward Pass Failed: {e}")
        return

    # 2. Backward Pass Test (Autograd)
    try:
        loss = m_out.sum()
        loss.backward()
        print("[PASS] Backward Pass Successful (Gradients computed).")
    except Exception as e:
        print(f"[FAIL] Backward Pass Failed: {e}")

if __name__ == "__main__":
    test_kernel()
