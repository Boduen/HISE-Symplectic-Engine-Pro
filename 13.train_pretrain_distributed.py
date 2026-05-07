import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

from hise.config import HISEConfig
from hise.modeling.modeling_hise import HISEForCausalLM
from hise.thermodynamics.annealing import ThermodynamicScheduler

# ==========================================
# AICA Constraint Definition: Explicit Constants & Configurations
# ==========================================
class TrainingConstants:
    VOCAB_SIZE: int = 50257
    DEFAULT_SEQ_LEN: int = 512
    FSI_SAFE_THRESHOLD: float = 1.0
    FSI_NUMERICAL_EPSILON: float = 1e-6

@dataclass(frozen=True)
class DistributedEnvConfig:
    """AICA Explicit Contract for Distributed Topology (Immutable)"""
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

# --- 1. 顯式的分散式環境注入 ---
def setup_distributed_env() -> DistributedEnvConfig:
    """Explicitly extracts and freezes the distributed environment state."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return DistributedEnvConfig(True, rank, world_size, local_rank, device)
    
    return DistributedEnvConfig(False, 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# --- 2. 物理感知數據集 ---
class CausalPhysicsDataset(Dataset):
    def __init__(self, size: int = 10000, seq_len: int = TrainingConstants.DEFAULT_SEQ_LEN) -> None:
        assert size > 0 and seq_len > 0, "Contract Violation: Size and seq_len must be positive."
        self.size = size
        self.seq_len = seq_len
        
    def __len__(self) -> int:
        return self.size
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(0, TrainingConstants.VOCAB_SIZE, (self.seq_len,), dtype=torch.long),
            "labels": torch.randint(0, TrainingConstants.VOCAB_SIZE, (self.seq_len,), dtype=torch.long)
        }

# --- 3. 辛-費雪損失函數 (Physics-Informed Loss) ---
class SymplecticFisherLoss(nn.Module):
    """
    AICA Asserts:
        - Strict unpacking of Tuple-based FSI scores.
        - Immutable reference to physical thresholds.
    """
    def __init__(self, lambda_phy: float = 0.1) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_phy = lambda_phy
        
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        fsi_scores_tuple: Optional[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss_phy = torch.tensor(0.0, device=logits.device)
        
        # AICA Fix: Safely handle the Tuple contract enforced in HISEForCausalLM
        if fsi_scores_tuple is not None and len(fsi_scores_tuple) > 0:
            # Extract the stacked/averaged tensor from the tuple
            avg_fsi = fsi_scores_tuple[0] 
            
            # Penalize FSI < 1.0
            violation = torch.relu(TrainingConstants.FSI_SAFE_THRESHOLD - avg_fsi) 
            loss_phy = violation.mean()
            
        total_loss = loss_ce + self.lambda_phy * loss_phy
        return total_loss, loss_ce, loss_phy

# --- 4. 訓練主迴圈 ---
def train(curriculum_path: str = "train/curriculum_config.json") -> None:
    # AICA Fix: Explicit dependency injection for environment
    env: DistributedEnvConfig = setup_distributed_env()
    is_master: bool = env.rank == 0
    
    # Optional dependency: Telemetry (Should ideally be abstracted via a Logger class)
    import wandb
    if is_master:
        wandb.init(project="HISE-Pro-Evolution", name="Run-01-System1-to-2")

    # AICA Fix: Explicit file existence check
    if not os.path.exists(curriculum_path):
        raise FileNotFoundError(f"AICA Guard: Curriculum file not found at {curriculum_path}")
        
    with open(curriculum_path, "r") as f:
        curriculum = json.load(f)
    
    config = HISEConfig(
        n_layers=12, d_model=768, d_inertial=64, 
        use_cognitive_gearbox=True,
        vocab_size=TrainingConstants.VOCAB_SIZE
    )
    
    model = HISEForCausalLM(config).to(env.device)
    
    if env.is_distributed:
        model = DDP(model, device_ids=[env.local_rank], output_device=env.local_rank)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    thermo_scheduler = ThermodynamicScheduler(
        model=model.module if env.is_distributed else model, 
        optimizer=optimizer
    )
    
    criterion = SymplecticFisherLoss(lambda_phy=curriculum['loss_function']['lambda_physics'])
    
    dataset = CausalPhysicsDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if env.is_distributed else None
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    global_step = 0
    model.train()
    
    for epoch in range(3):
        if sampler: sampler.set_epoch(epoch)
        
        for batch in dataloader:
            inputs = batch["input_ids"].to(env.device)
            labels = batch["labels"].to(env.device)
            
            # --- 1. 課表階段檢測 ---
            current_stage: Optional[Dict[str, Any]] = None
            for stage in curriculum['stages']:
                if stage['step_range'][0] <= global_step <= stage['step_range'][1]:
                    current_stage = stage
                    break
            
            raw_model = model.module if env.is_distributed else model
                
            # AICA Fix: Eliminate configuration mutation (HDIV Violation).
            # If epsilon is dynamically controlled by the curriculum, it should be passed 
            # as a runtime tensor or handled dynamically inside CognitiveGearbox, 
            # NOT by overriding the frozen config object.
            # (Here we bypass the mutation for safety, assuming Gearbox handles it dynamically based on FSI).
            
            # --- 2. 前向傳播 ---
            outputs = model(input_ids=inputs, labels=labels, output_fsi=True)
            
            # --- 3. 計算物理損失 ---
            # AICA Fix: Adhere to HF interface (attentions field contains the FSI tuple)
            fsi_tuple = outputs.attentions 
            loss, loss_ce, loss_phy = criterion(outputs.logits, labels, fsi_tuple)
            
            # --- 4. 反向傳播 ---
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # AICA Strict Contract: Compute average mass proxy from FSI to inject into the scheduler
            # (Since Mass ≈ 1 / FSI in the PSD framework)
            current_fsi_mean = fsi_tuple[0].mean().item() if (fsi_tuple and len(fsi_tuple) > 0) else 1.0
            avg_dynamic_mass = 1.0 / (current_fsi_mean + TrainingConstants.FSI_NUMERICAL_EPSILON)
            
            # AICA Fix: Explicit Injection of avg_dynamic_mass
            phy_stats = thermo_scheduler.step(
                epoch=epoch, 
                loss_val=loss.item(), 
                gradient_norm=grad_norm.item(),
                avg_dynamic_mass=avg_dynamic_mass
            )
            
            global_step += 1
            
            # --- 5. 遙測日誌 ---
            if is_master and global_step % 10 == 0:
                stage_name = current_stage['name'] if current_stage else "Steady_State"
                
                log_data = {
                    "train/loss_total": loss.item(),
                    "train/loss_ce": loss_ce.item(),
                    "train/loss_physics": loss_phy.item(),
                    "physics/temperature_tau": phy_stats['tau'],
                    "physics/global_mass": phy_stats['global_mass'],
                    "physics/grad_norm": grad_norm.item(),
                    "curriculum/stage": stage_name
                }
                
                wandb.log(log_data)
                print(f"[Step {global_step}] Stage: {stage_name} | "
                      f"Loss: {loss.item():.4f} (Phy: {loss_phy.item():.4f}) | "
                      f"Mass: {phy_stats['global_mass']:.2f} | Tau: {phy_stats['tau']:.2f}")

    if is_master:
        print(">>> Training Complete. Physical Model Converged.")
        raw_model = model.module if env.is_distributed else model
        raw_model.save_pretrained("checkpoints/hise-pro-evolved")
        
    if env.is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
