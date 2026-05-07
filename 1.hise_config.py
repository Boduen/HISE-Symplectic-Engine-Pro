from transformers import PretrainedConfig
from typing import Any

class HISEConfig(PretrainedConfig):
    """
    Configuration class for the HISE (Symplectic Engine Pro) model.

    This class inherits from Hugging Face's `PretrainedConfig` and is used to store
    the configuration of a HISE model, including standard transformer parameters,
    physics-informed inertial dynamics, cognitive gearbox settings, and Mixture
    of Physics Experts (MoPE) routing parameters.
    """
    model_type = "hise"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        max_position_embeddings: int = 8192,

        # --- Physics Parameters ---
        d_inertial: int = 64,          
        epsilon: float = 0.1,            
        tau: float = 1.0,                
        lambda_conf: float = 0.01,       

        # --- Cognitive Gearbox (System 1/2) ---
        use_cognitive_gearbox: bool = True,
        min_epsilon_scale: float = 0.1,  
        system2_threshold: float = 1.0,  

        # --- S-Tier Engineering (MOE & Memory) ---
        use_paged_momentum: bool = False,
        fsi_threshold: float = 1.0,

        # --- MoE Configuration ---
        use_moe: bool = False,          
        num_experts: int = 4,          
        num_experts_per_tok: int = 2,  
        moe_loss_weight: float = 0.01,   

        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        # 1. 執行參數的嚴格驗證 (Validation)
        self._validate_config(use_moe, num_experts, num_experts_per_tok, min_epsilon_scale)

        # 2. 基礎模型參數
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings

        # 3. 物理與流形動力學參數
        self.d_inertial = d_inertial
        self.epsilon = epsilon
        self.tau = tau
        self.lambda_conf = lambda_conf

        # 4. 認知齒輪參數
        self.use_cognitive_gearbox = use_cognitive_gearbox
        self.min_epsilon_scale = min_epsilon_scale
        self.system2_threshold = system2_threshold

        # 5. 進階工程與快取參數
        self.use_paged_momentum = use_paged_momentum
        self.fsi_threshold = fsi_threshold

        # 6. MoE (Mixture of Experts) 參數
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_loss_weight = moe_loss_weight

        self.initializer_range = initializer_range

        # 初始化父類別 (處理 name_or_path 等 kwargs)
        super().__init__(**kwargs)

    def _validate_config(
        self, 
        use_moe: bool, 
        num_experts: int, 
        num_experts_per_tok: int, 
        min_epsilon_scale: float
    ) -> None:
        """Validates critical constraints among configuration parameters."""
        if use_moe and num_experts_per_tok > num_experts:
            raise ValueError(
                f"`num_experts_per_tok` ({num_experts_per_tok}) cannot exceed "
                f"`num_experts` ({num_experts})."
            )
        
        if not (0.0 <= min_epsilon_scale <= 1.0):
            raise ValueError(
                f"`min_epsilon_scale` must be between 0.0 and 1.0, got {min_epsilon_scale}."
            )
