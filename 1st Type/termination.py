import torch

def check_termination(primal_residual: torch.Tensor, dual_residual: torch.Tensor, tol: float = 1e-4) -> bool:
    """Evaluates KKT conditions for convergence."""
    p_norm = torch.linalg.vector_norm(primal_residual, ord=float('inf'))
    d_norm = torch.linalg.vector_norm(dual_residual, ord=float('inf'))
    
    return (p_norm < tol) and (d_norm < tol)