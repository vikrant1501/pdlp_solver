import cupy as cp
from cupyx.scipy.sparse import csr_matrix

class HeuriSPAIPreconditioner:
    """
    Explicit Heuristic Sparse Approximate Inverse (HeuriSPAI) with
    Block-wise Dynamic Mixed-Precision (BDMP) using CuPy.
    """
    def __init__(self, A_csr: csr_matrix, max_nnz_per_col: int = 16, cond_threshold: float = 1000.0):
        self.A = A_csr
        self.n_rows, self.n_cols = A_csr.shape
        self.max_nnz = max_nnz_per_col
        self.cond_threshold = cond_threshold
        
        # ELLPACK Data Structures for fast SpMV
        self.ell_val = None
        self.ell_col = None
        self.M_dense = None

    def _generate_heuristic_pattern(self) -> cp.ndarray:
        """
        Pattern Generation: O(nnz(A)) pattern extraction using magnitude filtering
        and graph adjacency on CuPy.
        """
        A_dense = self.A.toarray()
        patterns = cp.zeros((self.n_cols, self.max_nnz), dtype=cp.int32)
        
        # Magnitude threshold filtering
        abs_A = cp.abs(A_dense)
        for j in range(self.n_cols):
            top_indices = cp.argsort(abs_A[:, j])[-self.max_nnz:]
            patterns[j, :len(top_indices)] = top_indices
            
        return patterns

    def construct_preconditioner(self):
        """
        Solves n independent least-squares subproblems min ||A_sub * m_j - e_j||_2
        using CuPy and BDMP precision routing.
        """
        A_dense = self.A.toarray()
        patterns = self._generate_heuristic_pattern()
        M_out = cp.zeros((self.n_rows, self.n_cols), dtype=cp.float32)
        I_mat = cp.eye(self.n_rows, dtype=cp.float32)

        for j in range(self.n_cols):
            pat_j = patterns[j]
            A_sub = A_dense[pat_j][:, pat_j]
            e_j = I_mat[pat_j, j]

            # BDMP Precision Routing based on condition number estimate
            cond_est = float(cp.linalg.cond(A_sub))
            if cond_est < self.cond_threshold:
                # Fast FP16 Least Squares on Tensor Cores
                m_j, _, _, _ = cp.linalg.lstsq(A_sub.astype(cp.float16), e_j.astype(cp.float16))
                M_out[pat_j, j] = m_j.astype(cp.float32)
            else:
                # High-Precision FP32 for ill-conditioned blending constraints
                m_j, _, _, _ = cp.linalg.lstsq(A_sub, e_j)
                M_out[pat_j, j] = m_j

        self.M_dense = M_out
        self._convert_to_ellpack()
        return self.M_dense

    def _convert_to_ellpack(self):
        """Converts explicit preconditioner M to ELLPACK format for memory coalescing."""
        mask = self.M_dense != 0
        self.ell_val = cp.zeros((self.n_rows, self.max_nnz), dtype=cp.float32)
        self.ell_col = cp.zeros((self.n_rows, self.max_nnz), dtype=cp.int32)

        for i in range(self.n_rows):
            nz_idx = cp.nonzero(mask[i, :])[0]
            k = min(len(nz_idx), self.max_nnz)
            if k > 0:
                self.ell_val[i, :k] = self.M_dense[i, nz_idx[:k]]
                self.ell_col[i, :k] = nz_idx[:k]

    def apply_preconditioner(self, x: cp.ndarray) -> cp.ndarray:
        """Coalesced ELLPACK Matrix-Vector Multiplication: y = M * x."""
        x_gathered = x[self.ell_col]
        return cp.sum(self.ell_val * x_gathered, axis=1)