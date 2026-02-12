#!/usr/bin/env python3
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    from src.model.transformer import GPTModel
    from src.model.config import CONFIGS
    
    print("Creating model...")
    model = GPTModel(CONFIGS['tiny'])
    print(f"✅ Model created: {CONFIGS['tiny'].n_params:,} parameters")
    
    print("\nTrying to load config...")
    import yaml
    with open('configs/quick_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded: {config['model']['name']}")
    
    print("\n🎉 Everything works! Ready to train.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
