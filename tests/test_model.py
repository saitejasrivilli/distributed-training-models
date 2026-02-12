import torch
import pytest
from src.model.transformer import GPTModel
from src.model.config import CONFIGS

def test_model_creation():
    """Test model can be created"""
    config = CONFIGS['tiny']
    model = GPTModel(config)
    assert model is not None
    print(f"✅ Model created with {config.n_params:,} parameters")

def test_model_forward():
    """Test forward pass"""
    config = CONFIGS['tiny']
    model = GPTModel(config)
    
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids)
    
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    print("✅ Forward pass successful")

def test_model_backward():
    """Test backward pass"""
    config = CONFIGS['tiny']
    model = GPTModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print("✅ Backward pass successful")

if __name__ == "__main__":
    test_model_creation()
    test_model_forward()
    test_model_backward()
    print("\n✅ All tests passed!")
