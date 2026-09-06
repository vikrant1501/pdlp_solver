import cupy as cp
from pdlp_solver.core.problem import MRPLRefineryLP
from pdlp_solver.preprocessing.preconditioning import HeuriSPAIPreconditioner
from pdlp_solver.optimization.pdhg_engine import PreconditionedPDHGEngine

class HeuriSPAIPDHGSolver:
    """Main Solver API for MRPL Refinery LPs."""
    def __init__(self, problem: MRPLRefineryLP):
        self.problem = problem
        self.precond = HeuriSPAIPreconditioner(problem.A)
        self.engine = None

    def solve(self, max_iters: int = 1000, tol: float = 1e-3):
        print("=== Constructing HeuriSPAI + BDMP Preconditioner ===")
        self.precond.construct_preconditioner()
        
        self.engine = PreconditionedPDHGEngine(
            self.problem.A, self.problem.b, self.problem.c, self.precond
        )
        
        print("=== Executing Preconditioned PDHG Solve Loop ===")
        for i in range(1, max_iters + 1):
            p_res, d_res = self.engine.step()
            
            if i % 100 == 0 or i == 1:
                print(f"Iter {i:4d} | Primal Residual (Ax - b): {p_res:.6f} | Dual Residual: {d_res:.6f}")
                
            if p_res < tol and d_res < tol:
                print(f"\nConverged successfully in {i} iterations!")
                break

        return self.engine.x, self.engine.y