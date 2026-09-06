import torch
from subprocedures import project_dual_variables

class PDHGEngine:
    def __init__(self, tau: float, sigma: float, m1: int):
        self.tau = tau
        self.sigma = sigma
        self.m1 = m1

    def step(self, x: torch.Tensor, y: torch.Tensor, K: torch.Tensor, KT: torch.Tensor):
        x_old = x.clone()
        
        if K.layout == torch.sparse_csr:
            primal_grad = torch.sparse.mm(KT, y.unsqueeze(1)).squeeze(1)
        else:
            primal_grad = KT @ y
            
        x = x - self.tau * primal_grad

        x_extrapolated = 2 * x - x_old

        if K.layout == torch.sparse_csr:
            dual_grad = torch.sparse.mm(K, x_extrapolated.unsqueeze(1)).squeeze(1)
        else:
            dual_grad = K @ x_extrapolated
            
        y = y + self.sigma * dual_grad
        y = project_dual_variables(y, self.m1)

        return x, y