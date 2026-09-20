import torch

from drc_mamba.metrics import IRSTDMetrics


def test_perfect_prediction():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1.0
    logits = torch.full_like(target, -10.0)
    logits[:, :, 7:9, 7:9] = 10.0

    metric = IRSTDMetrics()
    metric.update(logits, target)
    result = metric.compute()

    assert abs(result.iou - 1.0) < 1e-6
    assert abs(result.niou - 1.0) < 1e-6
    assert abs(result.pd - 1.0) < 1e-6
    assert abs(result.fa) < 1e-6


def test_unmatched_component_contributes_fa():
    target = torch.zeros(1, 1, 10, 10)
    logits = torch.full_like(target, -10.0)
    logits[:, :, 1:3, 1:3] = 10.0

    metric = IRSTDMetrics()
    metric.update(logits, target)
    result = metric.compute()

    assert result.fa > 0.0
