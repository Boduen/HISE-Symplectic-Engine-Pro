import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithCrossAttentions

from ..config import HISEConfig
from .base_layers import SoftTCMLayer
from .moe_router import MoPEBlock 
from ..thermodynamics.mass_dynamics import CognitiveGearbox

# ==========================================
# AICA Constraint Definition: Semantic Constants
# ==========================================
CAUSAL_MASK_VALUE: float = float('-inf')
ROUTING_DUMMY_FSI: float = 1.0


class HISEPreTrainedModel(PreTrainedModel):
    config_class = HISEConfig
    base_model_prefix = "hise"
    _no_split_modules = ["SoftTCMLayer", "MoPEBlock"] 

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights adhering to deterministic statistical bounds."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HISEModel(HISEPreTrainedModel):
    """
    Core HISE Model combining System 1 (Dense TCM) and System 2 (MoPE).
    
    AICA Asserts:
        - Prevents Type Collapse by strictly adhering to HF Output structures.
        - Employs explicit tensor shapes and explicitly declared dummy states.
    """
    def __init__(self, config: HISEConfig) -> None:
        super().__init__(config)
        self.embed_dim: int = config.d_model
        self.config: HISEConfig = config
        
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
        if config.use_moe:
            self.router_gearbox = CognitiveGearbox(config)
        
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            if config.use_moe:
                self.layers.append(MoPEBlock(config, num_experts=config.num_experts))
            else:
                self.layers.append(SoftTCMLayer(config))
        
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_momentums: Optional[List[torch.FloatTensor]] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_fsi: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        
        # AICA Guard: Strict dimensionality enforcement
        assert input_ids.dim() == 2, f"Contract Violation: input_ids must be 2D, got {input_ids.shape}"

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_fsi = output_fsi if output_fsi is not None else False
        
        B, T = input_ids.size()
        device = input_ids.device
        
        past_length: int = 0
        if past_momentums is not None and len(past_momentums) > 0 and past_momentums[0] is not None:
            past_length = past_momentums[0].shape[1]
            
        pos = torch.arange(past_length, past_length + T, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand(B, T)
        
        hidden_states = self.wte(input_ids) + self.wpe(pos)
        
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((T, T), CAUSAL_MASK_VALUE, device=device), 
                diagonal=1
            )

        next_momentums: List[torch.Tensor] = []
        all_fsi_scores: List[torch.Tensor] = [] 
        total_aux_loss: torch.Tensor = torch.tensor(0.0, device=device)
        
        # AICA Fix: Explicitly named and shaped semantic spoofing constant. 
        # (Ideal fix would be to add `derive_routing_mass` to Gearbox, avoiding FSI calculation entirely)
        dummy_fsi_for_routing = torch.full((B, T, 1), ROUTING_DUMMY_FSI, device=device, dtype=torch.float32)

        for i, layer in enumerate(self.layers):
            layer_past = past_momentums[i] if past_momentums is not None else None
            
            if self.config.use_moe:
                routing_mass, _, _ = self.router_gearbox(hidden_states, dummy_fsi_for_routing)
                hidden_states, layer_aux_loss = layer(hidden_states, routing_mass)
                
                total_aux_loss += layer_aux_loss
                
                m_new, fsi = None, None
            else:
                hidden_states, m_new, fsi = layer(
                    hidden_states, 
                    mask=attention_mask, 
                    past_momentum=layer_past
                )
            
            if use_cache and m_new is not None:
                next_momentums.append(m_new)
                
            if output_fsi and fsi is not None:
                all_fsi_scores.append(fsi)

        hidden_states = self.ln_f(hidden_states)

        # AICA Structural Contract Enforcement: 
        # We MUST NOT pass a scalar float into `cross_attentions` (Type Collapse). 
        # We pack it into a Tuple[torch.Tensor] to maintain Hugging Face API structural integrity.
        aux_loss_tuple = (total_aux_loss,) if self.config.use_moe else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_momentums if use_cache else None,
            # FSI scores packaged safely; HF expects Tuple of Tensors here.
            hidden_states=tuple(all_fsi_scores) if (output_fsi and all_fsi_scores) else None,
            cross_attentions=aux_loss_tuple, 
        )


class HISEForCausalLM(HISEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: HISEConfig) -> None:
        super().__init__(config)
        self.model = HISEModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_momentums: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_fsi: Optional[bool] = False,
    ) -> CausalLMOutputWithCrossAttentions:
        
        outputs = self.model(
            input_ids=input_ids,
            past_momentums=past_momentums,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_fsi=output_fsi
        )
        
        hidden_states = outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # AICA Fix: Avoid dynamic object allocation in forward pass. Use functional API.
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # AICA Structural Unpacking: Retrieve the safely packed aux_loss
            if self.config.use_moe and outputs.cross_attentions is not None:
                aux_loss = outputs.cross_attentions[0] # Unpack from tuple
                loss += self.config.moe_loss_weight * aux_loss

        fsi_metric = None
        
        if output_fsi and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
             fsi_stack = torch.stack(outputs.hidden_states)
             fsi_metric = fsi_stack.mean(dim=0)
             
        # Package FSI into attentions safely (Tuple expected by HF API)
        fsi_tuple = (fsi_metric,) if fsi_metric is not None else None

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, 
            attentions=fsi_tuple, 
            cross_attentions=outputs.cross_attentions
        )
