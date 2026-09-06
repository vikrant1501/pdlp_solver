import torch
from setup_and_sparse import LockFreeReductionEngine

class RuizPreconditioner:
    def __init__(self, eps: float = 1e-8, max_iter: int = 10):
        self.eps = eps
        self.max_iter = max_iter

    def equilibrate(self, K: torch.Tensor, KT: torch.Tensor, is_sparse: bool):
        device = K.device
        m, n = K.shape
        
        D = torch.ones(m, device=device)
        E = torch.ones(n, device=device)
        reducer = LockFreeReductionEngine(is_sparse=is_sparse)

        for _ in range(self.max_iter):
            row_max, col_max = reducer(K, KT if is_sparse else None)
            
            r_scale = torch.rsqrt(torch.clamp(row_max, min=self.eps))
            c_scale = torch.rsqrt(torch.clamp(col_max, min=self.eps))
            
            D *= r_scale
            E *= c_scale
            
            if not is_sparse:
                K = K * r_scale.unsqueeze(1) * c_scale.unsqueeze(0)
                KT = K.t().contiguous()
            else:
                row_indices = torch.arange(m, device=device).repeat_interleave(K.crow_indices()[1:] - K.crow_indices()[:-1])
                col_indices = K.col_indices()
                K.values().mul_(r_scale[row_indices] * c_scale[col_indices])
                KT.values().mul_(c_scale[KT.col_indices()] * r_scale[torch.arange(n, device=device).repeat_interleave(KT.crow_indices()[1:] - KT.crow_indices()[:-1])])

        return K, KT, D, E