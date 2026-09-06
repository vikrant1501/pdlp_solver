import torch

def compute_dense_max_reductions_safe(M: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Lock-free, memory-coalesced max reduction for Dense Matrices."""
    M_abs = M.abs().contiguous()
    row_max = torch.amax(M_abs, dim=1)
    col_max = torch.amax(M_abs, dim=0)
    
    row_max = torch.where(row_max > eps, row_max, torch.ones_like(row_max))
    col_max = torch.where(col_max > eps, col_max, torch.ones_like(col_max))
    return row_max, col_max

def compute_csr_row_max_safe(crow_indices: torch.Tensor, values_abs: torch.Tensor, num_rows: int, eps: float = 1e-12) -> torch.Tensor:
    """Lock-Free Segmented Parallel Reduction for CSR matrices."""
    if values_abs.numel() == 0:
        return torch.ones(num_rows, device=values_abs.device, dtype=values_abs.dtype)

    row_max = torch.segment_reduce(values_abs, reduce="amax", offsets=crow_indices)
    row_max = torch.where(
        torch.isinf(row_max) | (row_max <= eps), 
        torch.ones_like(row_max), 
        row_max
    )
    return row_max

def compute_csr_csc_max_reductions_safe(K_csr: torch.Tensor, K_csc: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Lock-Free Row and Column Maximums using dual CSR/CSC representations."""
    assert K_csr.layout == torch.sparse_csr, "K_csr must be in torch.sparse_csr layout"
    assert K_csc.layout == torch.sparse_csr, "K_csc (K^T) must be pre-computed in torch.sparse_csr layout"
    
    row_max = compute_csr_row_max_safe(K_csr.crow_indices(), K_csr.values().abs(), K_csr.shape[0], eps)
    col_max = compute_csr_row_max_safe(K_csc.crow_indices(), K_csc.values().abs(), K_csc.shape[0], eps)
    return row_max, col_max

class LockFreeReductionEngine:
    def __init__(self, is_sparse: bool, eps: float = 1e-12):
        self.is_sparse = is_sparse
        self.eps = eps

    def __call__(self, K: torch.Tensor, K_csc: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_sparse:
            return compute_dense_max_reductions_safe(K, self.eps)
        else:
            assert K_csc is not None, "K_csc must be provided for sparse matrices."
            return compute_csr_csc_max_reductions_safe(K, K_csc, self.eps)

class RobustDataInspector:
    def __init__(self, density_threshold: float = 0.10, vram_budget_ratio: float = 0.5):
        self.density_threshold = density_threshold
        self.vram_budget_ratio = vram_budget_ratio

    def _get_available_vram(self, device: torch.device) -> int:
        if device.type == "cuda":
            free_mem, _ = torch.cuda.mem_get_info(device)
            return free_mem
        return 4 * 1024 * 1024 * 1024

    def _estimate_dense_density(self, M: torch.Tensor, sample_rows: int = 128) -> float:
        m, n = M.shape
        if m <= sample_rows:
            return (torch.count_nonzero(M) / (m * n)).item()
        
        sample_indices = torch.randint(0, m, (sample_rows,), device=M.device)
        sampled_rows = M[sample_indices]
        return (torch.count_nonzero(sampled_rows) / (sample_rows * n)).item()

    def _direct_stitch_csr(self, G_csr: torch.Tensor, A_csr: torch.Tensor, m1: int, m2: int, n: int) -> torch.Tensor:
        crow_G, col_G, val_G = G_csr.crow_indices(), G_csr.col_indices(), G_csr.values()
        crow_A, col_A, val_A = A_csr.crow_indices(), A_csr.col_indices(), A_csr.values()

        nnz_G = val_G.numel()
        crow_K = torch.cat([crow_G, crow_A[1:] + nnz_G], dim=0)
        col_K = torch.cat([col_G, col_A], dim=0)
        val_K = torch.cat([val_G, val_A], dim=0)

        return torch.sparse_csr_tensor(crow_K, col_K, val_K, size=(m1 + m2, n), dtype=val_G.dtype, device=val_G.device)

    def inspect_and_format(self, G: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str, dict]:
        device, dtype = G.device, G.dtype
        m1, n = G.shape
        m2 = A.shape[0]
        m = m1 + m2
        total_elements = m * n

        element_size = torch.tensor([], dtype=dtype).element_size()
        dense_memory_bytes = total_elements * element_size
        free_vram = self._get_available_vram(device)

        if G.is_sparse or G.is_sparse_csr:
            density = (G.values().numel() + A.values().numel()) / max(total_elements, 1)
        else:
            density = (self._estimate_dense_density(G) * m1 + self._estimate_dense_density(A) * m2) / m

        fits_in_vram = dense_memory_bytes <= (free_vram * self.vram_budget_ratio)
        prefer_dense = (density > self.density_threshold) and fits_in_vram

        if prefer_dense:
            format_choice = "dense"
            G_dense = G.to_dense() if G.is_sparse or G.is_sparse_csr else G
            A_dense = A.to_dense() if A.is_sparse or A.is_sparse_csr else A
            K = torch.cat([G_dense, A_dense], dim=0).contiguous()
            KT = K.t().contiguous()
            total_nnz = total_elements
        else:
            format_choice = "csr"
            G_csr = G.to_sparse_csr() if not G.is_sparse_csr else G
            A_csr = A.to_sparse_csr() if not A.is_sparse_csr else A
            K = self._direct_stitch_csr(G_csr, A_csr, m1, m2, n)
            KT = K.to_sparse_coo().t().coalesce().to_sparse_csr()
            total_nnz = K.values().numel()

        metadata = {
            "m1": m1, "m2": m2, "total_rows": m, "cols": n,
            "density": density, "total_nnz": total_nnz, "format": format_choice,
            "device": str(device)
        }
        return K, KT, format_choice, metadata

def run_setup_pipeline(G: torch.Tensor, A: torch.Tensor):
    """Master Setup Pipeline."""
    inspector = RobustDataInspector()
    K, KT, fmt, meta = inspector.inspect_and_format(G, A)
    
    is_sparse = (fmt == "csr")
    reducer = LockFreeReductionEngine(is_sparse=is_sparse)
    row_max, col_max = reducer(K, KT if is_sparse else None)
    
    return K, KT, fmt, meta, row_max, col_max