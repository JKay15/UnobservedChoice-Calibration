import torch
import torch.nn as nn
import torch.optim as optim
import time
from ..config import ExpConfig

class CalibrationSolver:
    """
    Implements the calibration algorithms to recover gamma from biased simulators.
    """
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.device = cfg.device

    def solve_linear(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 1: Linear Regression Calibration.
        ...
        """
        N = y.shape[0]
        
        # 1. Construct Design Matrix A = [1, z, s_hat]
        ones = torch.ones(N, 1, device=self.device)
        s_hat_col = s_hat.view(-1, 1)
        
        A = torch.cat([ones, z, s_hat_col], dim=1)
        
        # 2. Solve Least Squares: min ||A*theta - y||^2
        # 【Fix for Mac MPS】: MPS currently doesn't support lstsq.
        # We strictly perform this specific solve on CPU.
        if self.device == 'mps':
            A_cpu = A.cpu()
            y_cpu = y.view(-1, 1).cpu()
            result = torch.linalg.lstsq(A_cpu, y_cpu)
            theta = result.solution.flatten().to(self.device)
        else:
            # CUDA or CPU works fine
            result = torch.linalg.lstsq(A, y.view(-1, 1))
            theta = result.solution.flatten()
        
        # 3. Extract Coefficients
        theta_s = theta[-1]
        theta_z = theta[1:-1]
        
        # 4. Parameter Recovery
        b_hat = -theta_s
        if torch.abs(b_hat) < 1e-6:
            # Avoid division by zero
            b_hat = 1e-6 * torch.sign(b_hat)
            
        gamma_hat = theta_z / b_hat
        
        return gamma_hat

    # def solve_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     """
    #     Algorithm 2: Maximum Rank Correlation (MRC) Calibration.
        
    #     Objective: Find gamma that maximizes the rank correlation between
    #                eta = (gamma^T z - s_hat)  and  y.
        
    #     Implementation:
    #     Since the rank indicator function is non-differentiable, we use a 
    #     smooth Pairwise Logistic Loss (Proxy Objective).
        
    #     Loss = sum_{i,j} log(1 + exp( -sign(y_i - y_j) * (eta_i - eta_j) ))
        
    #     Optimizer: L-BFGS (Quasi-Newton method) for high precision convergence.
    #     """
    #     N = y.shape[0]
    #     dim_z = z.shape[1]
        
    #     # 1. Initialize Gamma
    #     # Use standard normal initialization. requires_grad=True for optimization.
    #     gamma = torch.zeros(dim_z, device=self.device, requires_grad=True)
        
    #     # Initialize with a small random noise to break symmetry if needed, 
    #     # or just zeros if the loss surface is convex enough.
    #     with torch.no_grad():
    #         gamma.normal_(0, 0.1)

    #     # 2. Define Optimizer (L-BFGS is standard for MRC-like problems)
    #     # It usually converges much faster/accurately than SGD/Adam for this scale.
    #     optimizer = optim.LBFGS([gamma], lr=1.0, max_iter=100, history_size=10)

    #     # 3. Prepare Pairwise Targets (y_i - y_j)
    #     # To avoid O(N^2) memory explosion for large N, we can:
    #     # A. Use full matrix if N <= 5000 (5000^2 * 4 bytes ~ 100MB, totally fine on GPU)
    #     # B. Use mini-batch sampling if N is huge.
    #     # Here we assume N <= 10000, so full matrix is fine.
        
    #     # y_diff_sign: Sign of (y_i - y_j)
    #     # shape: (N, 1) - (1, N) -> (N, N) via broadcasting
    #     y_diff = y.view(-1, 1) - y.view(1, -1)
    #     y_sign = torch.sign(y_diff) # {-1, 0, 1}
        
    #     # We only care about pairs where y_i != y_j (informative pairs)
    #     # Mask out ties or diagonal
    #     mask = (y_sign != 0)
        
    #     # 4. Optimization Loop (L-BFGS requires a closure)
    #     def closure():
    #         optimizer.zero_grad()
            
    #         # Calculate estimated eta
    #         # eta = z @ gamma - s_hat
    #         eta = z @ gamma - s_hat
            
    #         # Calculate pairwise differences of eta
    #         # eta_diff: (N, N)
    #         eta_diff = eta.view(-1, 1) - eta.view(1, -1)
            
    #         # Soft Rank Loss (Logistic Loss on pairs)
    #         # We want sign(eta_diff) to match y_sign.
    #         # Maximize y_sign * eta_diff  => Minimize -y_sign * eta_diff
    #         # Smooth approximation: Softplus(-y_sign * eta_diff)
    #         # This is equivalent to BCE on pairs.
            
    #         # Apply mask to only train on informative pairs
    #         loss = torch.sum(torch.nn.functional.softplus(-y_sign[mask] * eta_diff[mask]))
            
    #         # Normalize by number of pairs to keep gradient scale reasonable
    #         loss = loss / mask.sum()
            
    #         if loss.requires_grad:
    #             loss.backward()
            
    #         return loss

    #     # Run Optimization
    #     optimizer.step(closure)
        
    #     return gamma.detach()
    def solve_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 2: Maximum Rank Correlation (MRC).
        """
        # ==========================================
        # [Stability Fix] MPS Fallback
        # MPS on Mac often crashes (Trace Trap) with L-BFGS or massive indexing.
        # We force CPU execution for the optimization loop if device is MPS.
        # ==========================================
        target_device = self.device
        if target_device == 'mps':
            target_device = 'cpu'
            z = z.cpu()
            s_hat = s_hat.cpu()
            y = y.cpu()
            
        N = y.shape[0]
        dim_z = z.shape[1]
        
        # 1. Initialize Gamma
        gamma = torch.zeros(dim_z, device=target_device, requires_grad=True)
        with torch.no_grad():
            gamma.normal_(0, 0.01)

        optimizer = optim.LBFGS([gamma], lr=1.0, max_iter=50, history_size=10, line_search_fn='strong_wolfe')

        # 2. Pair Sampling Strategy
        # [Optimization] 3M pairs is too much and causes OOM/Crash. 
        # 200k is sufficient for convergence.
        n_pairs = 200_000 
        
        use_sampling = (N * N > n_pairs * 2) # Use sampling if full matrix is much larger than n_pairs
        
        if not use_sampling:
            # Full Matrix Mode
            y_col = y.view(-1, 1)
            y_diff = y_col - y_col.T
            y_sign = torch.sign(y_diff)
            mask = (y_sign != 0)
        else:
            # Sampling Mode
            # Generate indices on the target device (CPU if MPS)
            idx_i = torch.randint(0, N, (n_pairs,), device=target_device)
            idx_j = torch.randint(0, N, (n_pairs,), device=target_device)
            
            # Filter valid pairs
            y_i = y[idx_i]
            y_j = y[idx_j]
            y_sign_sample = torch.sign(y_i - y_j)
            
            # Pre-fetch features (Running this on CPU avoids the MPS Trace Trap)
            z_i = z[idx_i]
            z_j = z[idx_j]
            s_i = s_hat[idx_i]
            s_j = s_hat[idx_j]

        # 3. Optimization Closure
        def closure():
            optimizer.zero_grad()
            
            if not use_sampling:
                eta = z @ gamma - s_hat
                eta_diff = eta.view(-1, 1) - eta.view(1, -1)
                loss = torch.sum(torch.nn.functional.softplus(-y_sign[mask] * eta_diff[mask])) / mask.sum()
            else:
                # Sampled Pairs
                eta_i = (z_i @ gamma) - s_i
                eta_j = (z_j @ gamma) - s_j
                eta_diff = eta_i - eta_j
                loss = torch.mean(torch.nn.functional.softplus(-y_sign_sample * eta_diff))

            # L2 Regularization
            loss = loss + 0.001 * torch.sum(gamma ** 2)
            
            if loss.requires_grad:
                loss.backward()
            return loss

        # 4. Execute
        optimizer.step(closure)
        
        # Move result back to original device
        return gamma.detach().to(self.device)
    
    def solve_multi_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, Y: torch.Tensor, method: str = 'median') -> torch.Tensor:
        """
        Algorithm 3: Multi-Simulator MRC.
        
        Args:
            z: (N, d) Features
            s_hat: (N,) Estimated Inclusive Value
            Y: (N, M) Matrix of M simulators
            method: 
                - 'median': Robust aggregation (Algorithm 3 Choice B)
                - 'logit_mean': Naive averaging of logits (Non-robust baseline)
        """
        # 1. Device Handling
        target_device = self.device
        if target_device == 'mps':
            target_device = 'cpu'
            z = z.cpu()
            s_hat = s_hat.cpu()
            Y = Y.cpu()
            
        N, M = Y.shape
        
        # ==========================================
        # Branch A: Naive Logit Averaging (Non-Robust)
        # ==========================================
        if method == 'logit_mean':
            # Strategy: Simply take the average of the logits y_k = (1/M) * sum(y_k^m)
            # If one simulator has extreme outliers (e.g. -100), this mean is ruined.
            y_aggregated = torch.mean(Y, dim=1) # (N,)
            
            # Solve as a standard Single-Simulator MRC problem
            # We reuse the logic (Sampling or Full) from solve_mrc, but reimplement here for cleanliness 
            # or we can just call self.solve_mrc if we move data back. 
            # Let's just call solve_mrc recursively to save code, handling device transfer.
            
            # Move back to original device to call public API
            y_agg_device = y_aggregated.to(self.device)
            z_device = z.to(self.device)
            s_device = s_hat.to(self.device)
            
            return self.solve_mrc(z_device, s_device, y_agg_device)

        # ==========================================
        # Branch B: Pairwise Robust Aggregation (Median)
        # ==========================================
        # Sampling Strategy
        n_pairs = 200_000
        use_sampling = (N * N > n_pairs * 2)
        
        if use_sampling:
            idx_i = torch.randint(0, N, (n_pairs,), device=target_device)
            idx_j = torch.randint(0, N, (n_pairs,), device=target_device)
            
            # Fetch Y pairs: (K, M)
            y_i = Y[idx_i]
            y_j = Y[idx_j]
            
            # Diffs: (K, M)
            diffs = y_i - y_j
            
            if method == 'median':
                # Median Aggregation: robust to outlier simulators
                median_diff = torch.median(diffs, dim=1).values
                s_kl = torch.sign(median_diff)
            elif method == 'voting_mean':
                # Legacy: Sum of signs (equivalent to median for M=3)
                votes = torch.sign(diffs).sum(dim=1)
                s_kl = torch.sign(votes)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            # Filter ties
            valid_mask = (s_kl != 0)
            idx_i = idx_i[valid_mask]
            idx_j = idx_j[valid_mask]
            s_kl = s_kl[valid_mask]
            
            z_i, z_j = z[idx_i], z[idx_j]
            s_i, s_j = s_hat[idx_i], s_hat[idx_j]
            
        else:
            # Full matrix fallback (omitted for brevity, assume N is large enough)
            raise NotImplementedError("Use N > 500 for Multi-Sim experiments.")

        # Optimization Loop
        dim_z = z.shape[1]
        gamma = torch.zeros(dim_z, device=target_device, requires_grad=True)
        with torch.no_grad():
            gamma.normal_(0, 0.01)
            
        optimizer = optim.LBFGS([gamma], lr=1.0, max_iter=50, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            eta_i = (z_i @ gamma) - s_i
            eta_j = (z_j @ gamma) - s_j
            eta_diff = eta_i - eta_j
            
            loss = torch.mean(torch.nn.functional.softplus(-s_kl * eta_diff))
            loss = loss + 0.001 * torch.sum(gamma ** 2)
            
            if loss.requires_grad:
                loss.backward()
            return loss
            
        optimizer.step(closure)
        
        return gamma.detach().to(self.device)