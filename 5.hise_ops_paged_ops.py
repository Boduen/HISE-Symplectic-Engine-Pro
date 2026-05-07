import torch
from typing import List, Dict

# ==========================================
# AICA Constraint Definition: State Clearance Constant
# ==========================================
ZERO_STATE_VALUE: float = 0.0


class PagedMomentumManager:
    """
    Manages physical memory for Hamiltonian Momentum states (m).
    Implements a PagedAttention-style memory layout to eliminate fragmentation
    during long-context System 2 reasoning.
    
    AICA Asserts:
        - Explicit Device & Dtype Injection (No implicit "cuda" defaults).
        - State Clearance on Free (Prevents HDIV: Cache Pollution).
        - Strict Type Contracts and Precondition Checks.
    """
    def __init__(
        self, 
        num_blocks: int, 
        block_size: int, 
        d_inertial: int, 
        dtype: torch.dtype,   # AICA Fix: Removed implicit default torch.float16
        device: torch.device  # AICA Fix: Removed implicit default "cuda"
    ) -> None:
        # AICA Precondition Checks: Prevent semantic collapse at initialization
        assert num_blocks > 0, "Contract Violation: num_blocks must be strictly positive."
        assert block_size > 0, "Contract Violation: block_size must be strictly positive."
        assert d_inertial > 0, "Contract Violation: d_inertial must be strictly positive."

        self.block_size: int = block_size
        self.d_inertial: int = d_inertial
        
        # Pre-allocate contiguous memory block
        # Shape: [Num_Blocks, Block_Size, D_Inertial]
        self.momentum_block_tables: torch.Tensor = torch.zeros(
            (num_blocks, block_size, d_inertial), 
            dtype=dtype, 
            device=device
        )
        self.free_blocks: List[int] = list(range(num_blocks))
        self.seq_to_block_table: Dict[int, List[int]] = {} 

    def allocate(self, seq_id: int, seq_len: int) -> List[int]:
        """Allocates physical blocks for a new sequence."""
        assert seq_len > 0, "Contract Violation: seq_len must be strictly positive."
        assert seq_id not in self.seq_to_block_table, f"Contract Violation: seq_id {seq_id} is already allocated."

        needed_blocks: int = (seq_len + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < needed_blocks:
            # AICA Guard: Prevent silent failure or undefined divergence behavior
            raise RuntimeError(
                f"OOM Guard: Insufficient PagedMomentum blocks. "
                f"Requested {needed_blocks}, Available {len(self.free_blocks)}."
            )
            
        blocks: List[int] = [self.free_blocks.pop() for _ in range(needed_blocks)]
        self.seq_to_block_table[seq_id] = blocks
        return blocks

    def get_physical_pointer(self, seq_id: int, token_pos: int) -> torch.Tensor:
        """
        Resolves logical position to physical memory address.
        Used by Triton kernels to write m_new.
        """
        # AICA Guard: Prevent dictionary lookup failures causing divergence
        if seq_id not in self.seq_to_block_table:
            raise KeyError(f"AICA Guard: Sequence ID {seq_id} not found in allocation table.")
            
        blocks: List[int] = self.seq_to_block_table[seq_id]
        block_idx: int = token_pos // self.block_size
        block_offset: int = token_pos % self.block_size
        
        # AICA Guard: Prevent Out-of-Bounds memory access
        if block_idx >= len(blocks):
            raise IndexError(f"AICA Guard: Token position {token_pos} exceeds allocated blocks for seq_id {seq_id}.")
        
        physical_block: int = blocks[block_idx]
        
        return self.momentum_block_tables[physical_block, block_offset]

    def free(self, seq_id: int) -> None:
        """
        Reclaims blocks when a sequence finishes.
        """
        if seq_id in self.seq_to_block_table:
            blocks: List[int] = self.seq_to_block_table.pop(seq_id)
            
            # AICA Fix: Prevent Temporal State Coupling & Cache Pollution.
            # We must explicitly zero out the memory before returning it to the pool,
            # ensuring rigorous deterministic initialization for future allocations.
            for block_idx in blocks:
                self.momentum_block_tables[block_idx].fill_(ZERO_STATE_VALUE)
                
            self.free_blocks.extend(blocks)
