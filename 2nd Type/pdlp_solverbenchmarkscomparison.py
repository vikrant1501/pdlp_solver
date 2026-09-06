import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from pdlp_solver.core.problem import MRPLRefineryLP
from pdlp_solver.solvers.pdhg_solver import HeuriSPAIPDHGSolver

def run_mrpl_benchmark():
    """Generates synthetic MRPL refinery matrix and benchmark solver."""
    cp.random.seed(42)
    n_dim = 200
    
    # Synthetic ill-conditioned matrix simulating refinery blending
    dense_A = cp.random.randn(n_dim, n_dim, dtype=cp.float32)
    U, S, V = cp.linalg.svd(dense_A)
    S_ill = cp.logspace(0, 4, n_dim, dtype=cp.float32) # Cond ~ 10^4
    A_mat = csr_matrix(U @ cp.diag(S_ill) @ V.T)
    
    b = cp.abs(cp.random.randn(n_dim, dtype=cp.float32) * 100.0)
    c = cp.random.randn(n_dim, dtype=cp.float32) * 10.0

    problem = MRPLRefineryLP(A_mat, b, c)
    solver = HeuriSPAIPDHGSolver(problem)
    x_opt, y_opt = solver.solve(max_iters=500, tol=1e-3)

if __name__ == "__main__":
    run_mrpl_benchmark()