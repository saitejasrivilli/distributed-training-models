#!/bin/bash

echo "================================================"
echo "📊 DISTRIBUTED LLM TRAINING - RESULTS SUMMARY"
echo "================================================"
echo ""

# Single GPU
if [ -f "experiments/quick_test/training_summary.json" ]; then
    echo "✅ Single GPU Results:"
    python3 << 'PYEOF'
import json
with open('experiments/quick_test/training_summary.json') as f:
    data = json.load(f)
print(f"   Loss: {data['final_loss']:.4f}")
print(f"   Steps: {data['final_step']}")
print(f"   GPUs: {data['num_gpus']}")
PYEOF
else
    echo "❌ Single GPU: No results"
fi

echo ""

# Multi GPU
if [ -f "experiments/multi_gpu_quick_test/training_summary.json" ]; then
    echo "✅ Multi-GPU Results:"
    python3 << 'PYEOF'
import json
with open('experiments/multi_gpu_quick_test/training_summary.json') as f:
    data = json.load(f)
print(f"   Loss: {data['final_loss']:.4f}")
print(f"   Steps: {data['final_step']}")
print(f"   GPUs: {data['num_gpus']}")
PYEOF
else
    echo "⏳ Multi-GPU: Running or not started..."
fi

echo ""
echo "================================================"
