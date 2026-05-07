import torch
from enum import Enum
from typing import List, Dict, Optional, Tuple, Callable

# ==========================================
# AICA Constraint Definition: Semantic Enums & Constants
# ==========================================
class SafetyStatus(Enum):
    SAFE = "safe"
    HALLUCINATION_RISK = "hallucination_risk"

class SafetyAction(Enum):
    CONTINUE = "continue"
    TRIGGER_RAG = "trigger_rag"

# ==========================================
# 1. Physics-Informed Safety Valve
# ==========================================
class FSISafetyValve:
    """
    Implements the Semantic Nyquist Limit check via Fisher Semantic Information (FSI) monitoring.
    
    AICA Asserts:
        - Replaced magic strings with strict Enums.
        - Eliminated implicit print() I/O side-effects. Logging must be injected.
    """
    def __init__(self, threshold: float = 1.0) -> None:
        assert threshold > 0.0, "Contract Violation: FSI threshold must be positive."
        self.threshold: float = threshold 

    def check_safety(
        self, 
        fsi_score: float, 
        current_token_id: int, 
        tokenizer,
        event_logger: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Enum]:
        
        status = SafetyStatus.SAFE
        action = SafetyAction.CONTINUE

        # FSI < threshold implies insufficient semantic mass (Axiom Smuggling)
        if fsi_score < self.threshold:
            status = SafetyStatus.HALLUCINATION_RISK
            action = SafetyAction.TRIGGER_RAG

            # AICA Fix: Dependency Injection for I/O operations
            if event_logger is not None:
                token_str = tokenizer.decode([current_token_id])
                event_logger(
                    f"[HISE-Guard] ALERT: FSI {fsi_score:.4f} < {self.threshold}. "
                    f"Token '{token_str}' triggered Axiom Smuggling warning."
                )

        return {
            "status": status,
            "action": action
        }

# ==========================================
# 2. RAG Controller (Entropy Sink)
# ==========================================
class RAGController:
    """
    Acts as an 'Entropy Sink', injecting low-entropy axioms to restore geodesic stability.
    """
    def __init__(self, event_logger: Optional[Callable[[str], None]] = None) -> None:
        self.event_logger = event_logger

    def retrieve_context(self, query_embedding: Optional[torch.Tensor] = None) -> str:
        if self.event_logger:
            self.event_logger("[RAG] Retrieving external axioms to restore thermodynamic balance...")
        
        # Simulated external retrieval
        return " [System Note: External Axiom Retrieved: The gravitational constant is G = 6.674e-11.] "

# ==========================================
# 3. Generation Loop with AICA Contracts
# ==========================================
def generate_with_safety(
    model, 
    tokenizer, 
    input_ids: torch.Tensor, 
    max_new_tokens: int = 50, 
    rag_controller: Optional[RAGController] = None,
    event_logger: Optional[Callable[[str], None]] = None
) -> torch.Tensor:
    """
    Generation loop with Physics-Informed Safety checks.
    
    AICA Asserts:
        - Strict handling of Hugging Face Tupled outputs (Type Collapse prevention).
        - State Reset constraint: cache MUST be cleared upon RAG intervention.
    """
    assert input_ids.dim() == 2, f"Contract Violation: input_ids must be 2D, got {input_ids.shape}"
    
    if rag_controller is None:
        rag_controller = RAGController(event_logger=event_logger)

    # Note: getattr used defensively in case config doesn't expose fsi_threshold directly
    threshold = getattr(model.config, 'fsi_threshold', 1.0)
    safety_valve = FSISafetyValve(threshold=threshold)

    generated = input_ids
    past_momentums = None

    for _ in range(max_new_tokens):
        # AICA State Consistency: Route correctly based on cache existence
        model_inputs = generated if past_momentums is None else generated[:, -1:]

        outputs = model(
            input_ids=model_inputs, 
            past_momentums=past_momentums, 
            use_cache=True,
            output_fsi=True 
        )

        next_token_logits = outputs.logits[:, -1, :]
        past_momentums = outputs.past_key_values

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # AICA Structural Unpacking: Safely extract FSI from HF Tuple contract
        fsi_tuple = outputs.attentions 
        
        if fsi_tuple is not None and len(fsi_tuple) > 0:
            fsi_metric_batch = fsi_tuple[0] 
            current_fsi = fsi_metric_batch[0].item()
            
            safety_res = safety_valve.check_safety(
                current_fsi, 
                next_token.item(), 
                tokenizer,
                event_logger=event_logger
            )

            if safety_res["action"] == SafetyAction.TRIGGER_RAG:
                # === System 2 Failure: External Intervention ===
                context_str = rag_controller.retrieve_context()
                context_ids = tokenizer.encode(context_str, return_tensors="pt").to(generated.device)

                # Inject Axioms
                generated = torch.cat([generated, context_ids], dim=1)

                # AICA Critical Fix: HDIV Temporal Coupling Prevention!
                # We CANNOT keep the old momentum cache. The sequence has fundamentally changed.
                # We must wipe the cache to force a full re-scan of the physics kernel.
                past_momentums = None 

                if event_logger:
                    event_logger("[HISE-Guard] Context injected. Cleared inertial cache. Resuming generation...")
                
                # Skip appending the hallucinated 'next_token' and restart the loop
                continue 

        # Normal generation step
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == getattr(tokenizer, 'eos_token_id', -1):
            break

    return generated
