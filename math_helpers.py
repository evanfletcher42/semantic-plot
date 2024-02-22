"""
A collection of helpful math functions for Pytorch tensors.
"""
import torch


def cubrt(x):
    """
    Returns the cubic root of a number, without being silly about negative numbers.
    """
    return torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0)


def safe_sqrt(x):
    """
    Returns 0 for x <= 0; sqrt(x) otherwise.
    """
    v = torch.where(x > 0, x, torch.zeros_like(x))
    return torch.sqrt(v)


def safe_acos(x):
    """
    Returns acos(x) with x clamped to (-1, 1))
    """
    return torch.arccos(torch.clip(x, -1.0, 1.0))
