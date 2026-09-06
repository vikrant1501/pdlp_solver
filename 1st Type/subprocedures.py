import torch

def project_dual_variables(y: torch.Tensor, m1: int):
    """Clips dual variables associated with inequalities (G) to be non-negative."""
    y[:m1].relu_()
    return y

def project_primal_bounds(x: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
    """Projects primal variables x into bounds."""
    return torch.clamp(x, min=lower_bound, max=upper_bound)