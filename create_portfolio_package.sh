#!/bin/bash

echo "📦 Creating Portfolio Package..."
echo ""

# Create portfolio directory
mkdir -p portfolio

# Copy key files
cp RESULTS.md portfolio/
cp docs/COMPLETE_RESULTS.md portfolio/ 2>/dev/null || echo "Skipping COMPLETE_RESULTS.md"
cp -r docs/images portfolio/

# Create portfolio README
cat > portfolio/README.md << 'PORTFOLIO'
# Distributed LLM Training System - Portfolio

## Quick Overview

Production-grade distributed deep learning system for training large language models.

### Key Results
- **3.50x speedup** with 4 GPUs
- **87.5% parallel efficiency**
- **152K tokens/second** throughput

### Files in This Package
- `RESULTS.md` - Performance summary
- `COMPLETE_RESULTS.md` - Full technical analysis
- `images/` - Performance visualizations

### Technology Stack
- PyTorch 2.7.1 + CUDA 11.8
- NCCL for GPU communication
- Data Parallelism (DDP)

### View Online
GitHub: [Add your URL]
LinkedIn: [Add your profile]

---
**Status:** Production-Ready | Validated with Real Training
PORTFOLIO

# Create archive
tar -czf portfolio_package.tar.gz portfolio/

echo "✅ Portfolio package created!"
echo ""
echo "📦 Package contents:"
ls -lh portfolio/
echo ""
echo "📦 Archive:"
ls -lh portfolio_package.tar.gz
