from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for transformer model"""
    
    # Architecture
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    
    # Sequence
    max_seq_len: int = 2048
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Initialization
    initializer_range: float = 0.02
    
    @property
    def n_params(self) -> int:
        """Calculate approximate number of parameters"""
        params = (
            self.vocab_size * self.d_model +
            self.max_seq_len * self.d_model +
            self.n_layers * (
                4 * self.d_model * self.d_model +
                2 * self.d_model * self.d_ff +
                5 * self.d_model
            ) +
            self.vocab_size * self.d_model
        )
        return params

# Predefined model sizes
CONFIGS = {
    "tiny": ModelConfig(
        d_model=128, n_heads=2, n_layers=2, d_ff=512, max_seq_len=512
    ),
    "small": ModelConfig(
        d_model=768, n_heads=12, n_layers=12, d_ff=3072
    ),
    "medium": ModelConfig(
        d_model=1024, n_heads=16, n_layers=24, d_ff=4096
    ),
    "large": ModelConfig(
        d_model=1280, n_heads=20, n_layers=36, d_ff=5120
    ),
}
