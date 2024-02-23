"""
A collection of helpful math functions for Pytorch tensors.
"""
import torch


def cubrt(x):
    """
    Returns the cubic root of a number, without being silly about negative numbers.
    """
    absx = torch.abs(x)
    absxclip = torch.clamp(absx, min=1e-9, max=None)
    return torch.sign(x) * torch.pow(absxclip, 1.0 / 3.0)


def safe_sqrt(x):
    """
    Returns 0 for x <= 0; sqrt(x) otherwise.
    """
    v = torch.where(x > 0, x, torch.zeros_like(x))
    return torch.sqrt(v)


def safe_acos(x):
    """
    Returns acos(x) with x clamped close to (-1, 1)
    Note: Close to, but not at - gradients are undefined at ±1
    """
    return torch.arccos(torch.clip(x, -0.9999, 0.9999))
