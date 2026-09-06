import cupy as cp
from cupyx.scipy.sparse import csr_matrix

class MRPLRefineryLP:
    """MRPL LP Problem definition for CuPy sparse processing."""
    def __init__(self, A: csr_matrix, b: cp.ndarray, c: cp.ndarray, unit_mapping: dict = None):
        assert A.shape[0] == b.shape[0], "Dimension mismatch between A and b"
        assert A.shape[1] == c.shape[0], "Dimension mismatch between A and c"
        self.A = A
        self.b = b.astype(cp.float32)
        self.c = c.astype(cp.float32)
        self.num_rows, self.num_cols = A.shape
        self.unit_mapping = unit_mapping or {}