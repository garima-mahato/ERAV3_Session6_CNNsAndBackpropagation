import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import Net
import pytest

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = Net()
    total_params = count_parameters(model)
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20000"

def test_batch_normalization():
    model = Net()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model does not use Batch Normalization"

def test_dropout():
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model does not use Dropout"

def test_gap():
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_gap, "Model does not use Global Average Pooling" 