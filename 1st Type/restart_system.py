import torch

def should_restart(x_current: torch.Tensor, x_old: torch.Tensor, y_current: torch.Tensor, y_old: torch.Tensor) -> bool:
    """Adaptive restart triggered by inner product condition."""
    dx = x_current - x_old
    dy = y_current - y_old
    return torch.dot(dx, dx) + torch.dot(dy, dy) < 0