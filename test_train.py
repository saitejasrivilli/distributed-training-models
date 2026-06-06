"""Sanity tests for the training pipeline: config loading and model instantiation."""
import pytest
from pathlib import Path

import torch
import yaml

from src.model.transformer import GPTModel
from src.model.config import CONFIGS


def test_torch_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available for distributed training"


def test_tiny_config_params():
    cfg = CONFIGS["tiny"]
    assert cfg.n_params > 0
    assert cfg.vocab_size > 0
    assert cfg.n_layers > 0


def test_quick_test_yaml_loads():
    config_path = Path(__file__).parent / "configs" / "quick_test.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert "model" in config
    assert "name" in config["model"]


def test_model_matches_config_params():
    cfg = CONFIGS["tiny"]
    model = GPTModel(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == cfg.n_params, f"Expected {cfg.n_params}, got {actual}"
