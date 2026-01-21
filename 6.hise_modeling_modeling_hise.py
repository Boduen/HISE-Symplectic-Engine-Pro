import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..config import HISEConfig
from .base_layers import SoftTCMLayer
# [NEW] Import MoPE components
from .moe_router import MoPEBlock 
from ..thermodynamics.mass_dynamics import CognitiveGearbox


class HISEPreTrainedModel(PreTrainedModel):
    config_class = HISEConfig
    base_model_prefix = "hise"
    # [NEW] Added MoPEBlock to prevent split errors during distributed training
    _no_split_modules = ["SoftTCMLayer", "MoPEBlock"] 

    def _init_weights(self, module):
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
    def __init__(self, config: HISEConfig):
        super().__init__(config)
        self.embed_dim = config.d_model
        self.config = config
        
        # 1. Embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
        # [NEW] Shared Gearbox for MoE Routing
        # If using MoE, we need a way to calculate 'Mass' to guide the router
        # regardless of whether the expert itself calculates physics.
        if config.use_moe:
            self.router_gearbox = CognitiveGearbox(config)
        
        # 2. Physics Layers (Stack of Soft-TCM or MoPE Blocks)
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            if config.use_moe:
                # [System 2] Mixture of Physics Experts
                self.layers.append(MoPEBlock(config, num_experts=config.num_experts))
            else:
                # [System 1] Standard Dense Physics Layer
                self.layers.append(SoftTCMLayer(config))
        
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_momentums: Optional[List[torch.FloatTensor]] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_fsi: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, Tuple]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_fsi = output_fsi if output_fsi is not None else False
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        device = input_ids.device
        
        # Position Embeddings
        if past_momentums is not None:
            # Handle list of past states safely
            if len(past_momentums) > 0 and past_momentums[0] is not None:
                past_length = past_momentums[0].shape[1]
            else:
                past_length = 0
        else:
            past_length = 0
            
        pos = torch.arange(past_length, past_length + input_shape[-1], dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).view(-1, input_shape[-1])
        
        hidden_states = self.wte(input_ids) + self.wpe(pos)
        
        # Causal Mask
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(input_shape[-1], input_shape[-1], device=device) * float('-inf'), diagonal=1)

        next_momentums = []
        all_fsi_scores = [] 
        total_aux_loss = 0.0 # [NEW] Track MoE Load Balancing Loss
        
        # Helper for MoE Routing Mass
        dummy_fsi_for_routing = torch.ones(input_ids.shape[0], input_ids.shape[1], 1, device=device)

        for i, layer in enumerate(self.layers):
            layer_past = past_momentums[i] if past_momentums is not None else None
            
            if self.config.use_moe:
                # === MoE Path ===
                # 1. Calculate Mass for Routing (using shared gearbox)
                # We use a dummy FSI=1.0 because we just need entropy-based mass for routing
                routing_mass, _, _ = self.router_gearbox(hidden_states, dummy_fsi_for_routing)
                
                # 2. Forward through MoPE Block
                # MoPE returns: hidden_states, aux_loss
                hidden_states, layer_aux_loss = layer(hidden_states, routing_mass)
                
                total_aux_loss += layer_aux_loss
                
                # MoE blocks currently abstract away the low-level momentum/fsi
                # So we append None or maintain structure
                m_new = None 
                fsi = None
                
            else:
                # === Dense Physics Path ===
                # Returns: hidden_states, m_new (Momentum), fsi (Risk Metric)
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

        # [NEW] Return Strategy
        # We overload 'cross_attentions' to carry total_aux_loss
        # We overload 'hidden_states' (tuple) to carry all_fsi_scores
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_momentums if use_cache else None,
            hidden_states=tuple(all_fsi_scores) if output_fsi and len(all_fsi_scores) > 0 else None,
            cross_attentions=total_aux_loss if self.config.use_moe else None,
        )


class HISEForCausalLM(HISEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HISEModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_momentums: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_fsi: Optional[bool] = False,
    ) -> CausalLMOutputWithCrossAttentions:
        
        outputs = self.model(
            input_ids,
            past_momentums=past_momentums,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_fsi=output_fsi
        )
        
        hidden_states = outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 1. Standard CE Loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 2. Add MoE Auxiliary Loss (Load Balancing)
            # Retrieved from 'cross_attentions' field
            if self.config.use_moe and outputs.cross_attentions is not None:
                aux_loss = outputs.cross_attentions
                # Weighted sum (weight defined in config, default 0.01)
                loss += self.config.moe_loss_weight * aux_loss

        # FSI Handling for Safety Valve
        fsi_metric = None
        if output_fsi and outputs.hidden_states is not None:
             # Stack layers: [Layers, Batch, Seq]
             fsi_stack = torch.stack(outputs.hidden_states)
             # Average across layers to get a global System 2 risk score
             fsi_metric = fsi_stack.mean(dim=0)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, 
            attentions=fsi_metric, 
            cross_attentions=outputs.cross_attentions # Pass Aux Loss for monitoring
        )
