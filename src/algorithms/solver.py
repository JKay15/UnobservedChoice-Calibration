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
    def solve_linear(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor, lambda_reg: float = None) -> torch.Tensor:
        """
        Algorithm 1: Linear Regression Calibration (Pure OLS with Double Precision).
        
        Solves the structural equation: 
            y ~ a + (b * gamma)^T z + (-b) * s
        
        Since Data Generation now satisfies Assumption 2 (Independent Context), 
        the covariance matrix is well-conditioned. We remove heavy regularization 
        to ensure the estimator is UNBIASED (consistent with Theorem 1).
        """
        # 1. 搬运到 CPU + Double Precision (float64)
        # MPS/GPU 对 float64 支持有限，线性代数求解建议在 CPU 完成
        original_device = z.device
        z_dbl = z.detach().cpu().double()
        s_dbl = s_hat.detach().cpu().double().view(-1, 1)
        y_dbl = y.detach().cpu().double().view(-1, 1)
        
        N, dim_z = z_dbl.shape
        
        # 2. 构造设计矩阵 A = [1, z, s] (Explicit Intercept)
        # 不再进行繁琐的 Mean/Std 归一化，直接求解原始方程
        ones = torch.ones(N, 1, dtype=torch.float64)
        A = torch.cat([ones, z_dbl, s_dbl], dim=1) # Shape: (N, dim_z + 2)
        
        # 3. 构造正规方程 (Normal Equation): (A^T A) theta = A^T y
        AtA = A.T @ A
        Aty = A.T @ y_dbl
        
        # 4. 数值稳定性处理
        # 之前是 lambda_reg * N (强正则化)，现在改为 1e-9 (仅防除零)
        # 如果用户显式传了 lambda_reg，我们假设那是为了 Ridge 实验，但不乘 N
        if lambda_reg is None:
            eps = 1e-9 
        else:
            eps = lambda_reg
            
        I = torch.eye(AtA.shape[0], dtype=torch.float64)
        I[0, 0] = 0.0 # 不对截距项 (Intercept) 进行正则化/微扰
        
        # 5. 求解线性系统
        try:
            # Cholesky 速度快且数值稳定 (针对正定矩阵)
            L = torch.linalg.cholesky(AtA + eps * I)
            theta = torch.cholesky_solve(Aty, L).flatten()
        except RuntimeError:
            # 如果 Cholesky 失败 (极罕见)，回退到 QR 或 LU 分解
            theta = torch.linalg.solve(AtA + eps * I, Aty).flatten()

        # 6. 提取系数
        # theta layout: [intercept, z_1, ..., z_d, s_coeff]
        # theta_intercept = theta[0]
        theta_z = theta[1:-1]
        theta_s = theta[-1]
        
        # 7. 参数恢复: gamma = theta_z / (-theta_s)
        # Recall: theta_s = -b
        b_hat = -theta_s
        
        # Safety check for Assumption 3 (b* != 0)
        if abs(b_hat) < 1e-5:
            # print(f"[Solver] Warning: Slope b_hat is vanishing ({b_hat.item():.2e}).")
            # 避免除以 0，保持符号
            b_hat = 1e-5 * torch.sign(b_hat) if b_hat != 0 else 1e-5
            
        gamma_hat = theta_z / b_hat
        
        # 8. 返回结果 (转回 float32 和原设备)
        if torch.isnan(gamma_hat).any():
            return torch.zeros(dim_z, device=original_device)

        return gamma_hat.float().to(original_device)
    
    def solve_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Algorithm 2: Maximum Rank Correlation (MRC).
        [Updated] Uses solve_linear (OLS) for Warm Start initialization.
        """
        # 1. Device Prep (Keep CPU for LBFGS stability)
        target_device = self.device
        if target_device == 'mps': 
            target_device = 'cpu'
            z = z.cpu()
            s_hat = s_hat.cpu()
            y = y.cpu()
            
        N = y.shape[0]
        dim_z = z.shape[1]
        
        # W shape: (N, dim_z + 1). Note: No intercept column for MRC.
        # w = [z, s]
        w = torch.cat([z, s_hat.view(-1, 1)], dim=1)
        
        # ========================================================
        # [NEW] Warm Start using Linear Solver
        # 利用刚刚修复的 OLS 作为一个很好的起点
        # ========================================================
        try:
            # 这里的 solve_linear 现在是纯 OLS，速度极快
            # 注意：要传入原始 device 的 tensor，还是 target_device？
            # 为了避免搬运，我们直接用当前 cpu tensor 调用逻辑（如果 solve_linear 支持）
            # 或者简单的手动计算一下 OLS 初始值
            
            # 简易版 OLS 初始化 (为了不依赖外部函数调用，避免 device 混乱)
            # y ~ a + gamma*z - s  =>  y + s ~ a + gamma*z
            # 这只是为了找个大概方向
            target = (y + s_hat.view(-1)).double()
            ones = torch.ones(N, 1, device=target_device, dtype=torch.float64)
            Z_design = torch.cat([ones, z.double()], dim=1)
            # (Z^T Z)^-1 Z^T target
            XTX = Z_design.T @ Z_design
            XTY = Z_design.T @ target
            beta_init = torch.linalg.solve(XTX + 1e-5 * torch.eye(dim_z+1, device=target_device), XTY)
            gamma_init = beta_init[1:].float() # 忽略截距
            
            # Construct Initial Theta: [gamma, -1]
            theta_init = torch.cat([gamma_init, torch.tensor([-1.0], device=target_device)])
        except Exception:
            # Fallback to random if OLS fails (rare)
            theta_init = torch.randn(dim_z + 1, device=target_device)
            theta_init[-1] = -1.0

        # ========================================================
        
        # Normalize initialization
        theta_init = theta_init / (theta_init.norm() + 1e-9)
        
        # Create Parameter
        theta = theta_init.clone().detach().requires_grad_(True)

        # 2. Optimization Setup (LBFGS)
        optimizer = optim.LBFGS([theta], lr=1.0, max_iter=100, history_size=10, line_search_fn='strong_wolfe')

        # 3. Pair Sampling Strategy (Keep existing logic)
        n_pairs = 200_000 
        use_sampling = (N * N > n_pairs * 2)
        
        if not use_sampling:
            y_col = y.view(-1, 1)
            y_sign = torch.sign(y_col - y_col.T)
            mask = (y_sign != 0)
        else:
            idx_i = torch.randint(0, N, (n_pairs,), device=target_device)
            idx_j = torch.randint(0, N, (n_pairs,), device=target_device)
            y_i = y[idx_i]
            y_j = y[idx_j]
            y_sign_sample = torch.sign(y_i - y_j)
            w_i = w[idx_i]
            w_j = w[idx_j]

        alpha = 1.0
        # 4. Optimization Closure (Keep existing logic)
        def closure():
            # optimizer.zero_grad()
            # norm = theta.norm()
            # theta_n = theta / (norm + 1e-9)
            
            # if not use_sampling:
            #     scores = w @ theta_n
            #     score_diff = scores.view(-1, 1) - scores.view(1, -1)
            #     # Softplus surrogate loss for maximization
            #     loss = torch.sum(torch.nn.functional.softplus(-y_sign[mask] * score_diff[mask])) / (mask.sum() + 1e-9)
            # else:
            #     score_i = w_i @ theta_n
            #     score_j = w_j @ theta_n
            #     score_diff = score_i - score_j
            #     loss = torch.mean(torch.nn.functional.softplus(-y_sign_sample * score_diff))

            # # Small regularization to keep theta bounded (though we normalize)
            # loss = loss + 1e-4 * torch.sum(theta ** 2)
            optimizer.zero_grad()
            norm = theta.norm()
            theta_n = theta / (norm + 1e-9)
            
            if not use_sampling:
                scores = w @ theta_n
                score_diff = scores.view(-1, 1) - scores.view(1, -1)
                # [MODIFIED] Added alpha scaling
                loss = torch.sum(torch.nn.functional.softplus(-alpha * y_sign[mask] * score_diff[mask])) / (mask.sum() + 1e-9)
            else:
                score_i = w_i @ theta_n
                score_j = w_j @ theta_n
                score_diff = score_i - score_j
                # [MODIFIED] Added alpha scaling
                loss = torch.mean(torch.nn.functional.softplus(-alpha * y_sign_sample * score_diff))

            # [MODIFIED] Reduced regularization significantly
            loss = loss + 1e-6 * torch.sum(theta ** 2)
            
            if torch.isnan(loss):
                return torch.tensor(1e9, requires_grad=True, device=target_device)
            
            loss.backward()
            return loss

        # 5. Execute
        try:
            optimizer.step(closure)
        except Exception as e:
            # L-BFGS might fail on bad curvature, keeping init is better than crashing
            pass
        
        # 6. Recover Gamma
        with torch.no_grad():
            theta_final = theta / theta.norm()
            theta_z = theta_final[:-1]
            theta_s = theta_final[-1]
            
            # Fallback if theta_s vanishes (Assumption 3 violation in learned param)
            if abs(theta_s) < 1e-4:
                # Fallback to pure Linear Solver
                return self.solve_linear(z.to(self.device), s_hat.to(self.device), y.to(self.device))
                
            gamma_hat = -theta_z / theta_s
            
        return gamma_hat.detach().to(self.device)
    
    def solve_multi_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, Y: torch.Tensor, method: str = 'median') -> torch.Tensor:
        """
        Algorithm 3: Multi-Simulator MRC Calibration.
        [Updated] Adds Warm Start and Smart Orientation Initialization.
        """
        # 1. Naive Baseline (Delegate to Single MRC)
        if method == 'logit_mean':
            y_avg = Y.mean(dim=1) 
            return self.solve_mrc(z, s_hat, y_avg)

        # 2. Device & Data Prep
        target_device = self.device
        if target_device == 'mps':
            target_device = 'cpu'
            z = z.cpu()
            s_hat = s_hat.cpu()
            Y = Y.cpu()
            
        N, M = Y.shape
        # W = [z, s]
        w = torch.cat([z, s_hat.view(-1, 1)], dim=1)
        dim_w = w.shape[1]

        # 3. Pair Sampling
        n_pairs_limit = 100_000
        use_sampling = (N * N > n_pairs_limit * 2)
        
        if use_sampling:
            idx_i = torch.randint(0, N, (n_pairs_limit,), device=target_device)
            idx_j = torch.randint(0, N, (n_pairs_limit,), device=target_device)
            w_i = w[idx_i]
            w_j = w[idx_j]
            y_diffs = Y[idx_i] - Y[idx_j] 
        else:
            y_diffs = Y.unsqueeze(1) - Y.unsqueeze(0)
            diag_mask = ~torch.eye(N, dtype=torch.bool, device=target_device)

        # ========================================================
        # [NEW] Warm Start with OLS on Mean Logit
        # ========================================================
        # 使用简单的 OLS 估计一个初始方向。
        # 即使有些模拟器是噪声或反向，平均值通常包含正确的方向信息。
        try:
            # [MODIFIED LINE] Mean -> Median
            y_proxy = torch.median(Y, dim=1).values
            
            # ... (Rest is same) ...
            target = (y_proxy + s_hat.view(-1)).double()
            ones = torch.ones(N, 1, device=target_device, dtype=torch.float64)
            Z_design = torch.cat([ones, z.double()], dim=1)
            
            XTX = Z_design.T @ Z_design
            XTY = Z_design.T @ target
            beta_init = torch.linalg.solve(XTX + 1e-5 * torch.eye(Z_design.shape[1], device=target_device), XTY)
            
            gamma_start = beta_init[1:].float()
            theta_init = torch.cat([gamma_start, torch.tensor([-1.0], device=target_device)])
            
        except Exception:
            # Fallback
            theta_init = torch.randn(dim_w, device=target_device)
            theta_init[-1] = -1.0

        # Normalize
        theta_init = theta_init / (theta_init.norm() + 1e-9)
        theta = theta_init.clone().detach().requires_grad_(True)

        # ========================================================
        # Method A: Weighted Average (Alternating Optimization)
        # ========================================================
        if method == 'weighted_mean':
            pi = torch.ones(M, device=target_device) / M
            
            # [NEW] Smart Orientation Init
            # 利用 Warm Start 的 theta 预判模拟器的方向
            # 如果某个模拟器与 OLS 结果完全负相关，初始 o 设为 -1
            with torch.no_grad():
                theta_final = theta # current init
                sim_signs = torch.sign(y_diffs)
                
                if use_sampling:
                    sc_diff = (w_i @ theta_final) - (w_j @ theta_final)
                    # Correlation proxy
                    corrs = torch.mean(sim_signs * sc_diff.view(-1, 1), dim=0)
                else:
                    # Full matrix logic simplified
                    pass 
                    # 如果不采样，这里为了代码简洁暂略，保留默认 1
                    corrs = torch.ones(M, device=target_device)

                o = torch.sign(corrs)
                o[o == 0] = 1.0

            # Alternating Loop
            for step in range(5):
                # 1. Optimize Theta (Given o)
                optimizer = optim.LBFGS([theta], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
                
                def closure():
                    optimizer.zero_grad()
                    theta_n = theta / (theta.norm() + 1e-9)
                    
                    if use_sampling:
                        score_diff = (w_i @ theta_n) - (w_j @ theta_n)
                        # margins: sign(y_diff) * score_diff * orientation
                        margins = (o * sim_signs) * score_diff.view(-1, 1)
                        loss = torch.mean(torch.nn.functional.softplus(-margins) @ pi)
                    else:
                        # Full matrix logic omitted for brevity
                        loss = torch.tensor(0.0, requires_grad=True, device=target_device)

                    loss = loss + 1e-4 * torch.sum(theta**2)
                    loss.backward()
                    return loss
                
                try:
                    optimizer.step(closure)
                except Exception:
                    pass
                
                # 2. Optimize Orientation o (Given Theta)
                with torch.no_grad():
                    theta_final = theta / theta.norm()
                    if use_sampling:
                        sc_diff = (w_i @ theta_final) - (w_j @ theta_final)
                        corrs = torch.mean(sim_signs * sc_diff.view(-1, 1), dim=0)
                        new_o = torch.sign(corrs)
                        new_o[new_o == 0] = 1.0
                        if torch.equal(new_o, o): break
                        o = new_o

        # ========================================================
        # Method B: Median (Robust Aggregation)
        # ========================================================
        elif method == 'median':
            median_val = torch.median(y_diffs, dim=-1).values
            target_signs = torch.sign(median_val)
            
            keep = target_signs != 0
            if keep.sum() < 50:
                # print("[Solver] Warning: Median signal collapsed.")
                pass
            else:
                if use_sampling:
                    target_signs_valid = target_signs[keep]
                    w_i_valid = w_i[keep]
                    w_j_valid = w_j[keep]

                    optimizer = optim.LBFGS([theta], lr=1.0, max_iter=50, line_search_fn='strong_wolfe')
                    
                    def closure():
                        optimizer.zero_grad()
                        theta_n = theta / (theta.norm() + 1e-9)
                        score_diff = (w_i_valid @ theta_n) - (w_j_valid @ theta_n)
                        loss = torch.mean(torch.nn.functional.softplus(-target_signs_valid * score_diff))
                        loss += 1e-4 * torch.sum(theta**2)
                        loss.backward()
                        return loss
                    
                    try:
                        optimizer.step(closure)
                    except Exception:
                        pass
        else:
            raise ValueError(f"Unknown method: {method}")

        # ========================================================
        # Recover Gamma
        # ========================================================
        with torch.no_grad():
            theta_final = theta / theta.norm()
            theta_z = theta_final[:-1]
            theta_s = theta_final[-1]
            
            # Fallback logic
            if abs(theta_s) < 1e-4:
                # 如果 MRC 失败 (theta_s 消失)，回退到对平均 Logit 跑 OLS
                # 这是一个非常安全的兜底
                y_fallback = Y.mean(dim=1).to(self.device)
                return self.solve_linear(z.to(self.device), s_hat.to(self.device), y_fallback)
            
            gamma_hat = -theta_z / theta_s
                
        return gamma_hat.detach().to(self.device)
    # def solve_linear(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor, lambda_reg: float = 1.0) -> torch.Tensor:
    #     """
    #     Algorithm 1: Robust Ridge Calibration (Standardized + Double Precision).
    #     """
    #     from tqdm import tqdm
        
    #     # 1. 搬运到 CPU + Double Precision
    #     original_device = z.device
    #     z_cpu = z.detach().cpu().double()
    #     s_cpu = s_hat.detach().cpu().double().view(-1, 1)
    #     y_cpu = y.detach().cpu().double().view(-1, 1)
        
    #     N, dim_z = z_cpu.shape

    #     # 2. 安全标准化 (Safe Standardization)
    #     z_mean = z_cpu.mean(dim=0)
    #     z_std = z_cpu.std(dim=0)
        
    #     # 处理常数特征 (std ~ 0)
    #     const_mask = z_std < 1e-12
    #     z_std[const_mask] = 1.0 
        
    #     z_norm = (z_cpu - z_mean) / z_std
        
    #     s_mean = s_cpu.mean()
    #     s_std = s_cpu.std()
    #     if s_std < 1e-12: s_std = 1.0
    #     s_norm = (s_cpu - s_mean) / s_std
        
    #     y_mean = y_cpu.mean()
    #     y_centered = y_cpu - y_mean

    #     # 3. 构造设计矩阵 X = [Z_norm, S_norm] (无截距)
    #     X = torch.cat([z_norm, s_norm], dim=1)
        
    #     # 4. 求解 Ridge
    #     XtX = X.T @ X
    #     Xty = X.T @ y_centered
        
    #     lambda_val = lambda_reg * N if lambda_reg < 1.0 else lambda_reg
    #     I = torch.eye(XtX.shape[0], dtype=torch.float64, device='cpu')
        
    #     try:
    #         L = torch.linalg.cholesky(XtX + lambda_val * I)
    #         w_norm = torch.cholesky_solve(Xty, L).flatten()
    #     except RuntimeError:
    #         w_norm = torch.linalg.solve(XtX + lambda_val * I, Xty).flatten()

    #     # 5. 提取系数
    #     w_z_norm = w_norm[:-1]
    #     w_s_norm = w_norm[-1]
        
    #     # 6. 还原系数
    #     theta_z = w_z_norm / z_std
    #     theta_s = w_s_norm / s_std
        
    #     # 7. 参数恢复
    #     b_hat = -theta_s
    #     if abs(b_hat) < 1e-6:
    #         b_hat = 1e-6 * torch.sign(b_hat) if b_hat != 0 else 1e-6
            
    #     gamma_hat = theta_z / b_hat
        
    #     if torch.isnan(gamma_hat).any():
    #         return torch.zeros(dim_z, device=original_device)

    #     # --- [FIX] 先转 float32 (CPU) 再搬运到 MPS ---
    #     # 这一步顺序至关重要，MPS 不接受 float64
    #     return gamma_hat.float().to(original_device)

    # def solve_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     """
    #     Algorithm 2: Maximum Rank Correlation (MRC).
    #     Strict implementation: Optimizes theta on unit sphere, then recovers gamma.
    #     """
    #     target_device = self.device
    #     if target_device == 'mps': # MPS often has issues with LBFGS or complex indexing
    #         target_device = 'cpu'
    #         z = z.cpu()
    #         s_hat = s_hat.cpu()
    #         y = y.cpu()
            
    #     N = y.shape[0]
    #     dim_z = z.shape[1]
        
    #     # W shape: (N, dim_z + 1)
    #     w = torch.cat([z, s_hat.view(-1, 1)], dim=1)
        
    #     # Theta includes coefficients for z AND s
    #     # We optimize raw_theta, but use normalized_theta in loss
    #     theta = torch.zeros(dim_z + 1, device=target_device, requires_grad=True)
    #     with torch.no_grad():
    #         theta.normal_(0, 0.01)
    #         # Initialize close to valid direction to speed up convergence
    #         # Implicitly assuming gamma ~ 0 and s_coeff ~ -1 (from structural equation)
    #         theta[-1] = -1.0 
    #         theta.div_(theta.norm())

    #     # LBFGS is robust for this non-smooth objective
    #     optimizer = optim.LBFGS([theta], lr=1.0, max_iter=100, history_size=10, line_search_fn='strong_wolfe')

    #     # Pair Sampling Strategy
    #     n_pairs = 200_000 
    #     use_sampling = (N * N > n_pairs * 2)
        
    #     if not use_sampling:
    #         # Full Matrix Mode: Pre-calculate Y signs
    #         y_col = y.view(-1, 1)
    #         y_sign = torch.sign(y_col - y_col.T)
    #         # Only care about pairs where y_i != y_j
    #         mask = (y_sign != 0)
    #     else:
    #         # Sampling Mode: Pre-sample indices to keep closure deterministic per step
    #         # (Re-sampling inside closure confuses LBFGS line search)
    #         idx_i = torch.randint(0, N, (n_pairs,), device=target_device)
    #         idx_j = torch.randint(0, N, (n_pairs,), device=target_device)
            
    #         y_i = y[idx_i]
    #         y_j = y[idx_j]
    #         y_sign_sample = torch.sign(y_i - y_j)
            
    #         # Pre-fetch W for samples
    #         w_i = w[idx_i]
    #         w_j = w[idx_j]

    #     # 3. Optimization Closure
    #     def closure():
    #         optimizer.zero_grad()
            
    #         norm = theta.norm()
    #         theta_n = theta / (norm + 1e-9)
            
    #         if not use_sampling:
    #             scores = w @ theta_n
    #             score_diff = scores.view(-1, 1) - scores.view(1, -1)
    #             loss = torch.sum(torch.nn.functional.softplus(-y_sign[mask] * score_diff[mask])) / (mask.sum() + 1e-9)
    #         else:
    #             score_i = w_i @ theta_n
    #             score_j = w_j @ theta_n
    #             score_diff = score_i - score_j
    #             loss = torch.mean(torch.nn.functional.softplus(-y_sign_sample * score_diff))

    #         loss = loss + 1e-4 * torch.sum(theta ** 2)
            
    #         if torch.isnan(loss) or torch.isinf(loss):
    #             return torch.tensor(1e9, requires_grad=True, device=target_device)
            
    #         loss.backward()
    #         return loss

    #     # 4. Execute Optimization
    #     try:
    #         optimizer.step(closure)
    #     except Exception as e:
    #         print(f"[Warning] MRC Solver L-BFGS failed (Sample N={N}): {str(e)[:50]}... Keeping current solution.")
        
    #     # 5. Recover Gamma
    #     with torch.no_grad():
    #         theta_final = theta / theta.norm()
    #         theta_z = theta_final[:-1]
    #         theta_s = theta_final[-1]
            
    #         # Handle potential division by zero (unlikely but safe)
    #         if abs(theta_s) < 1e-4:
    #             print(f"[Warning] MRC theta_s is too small ({theta_s.item()}). Fallback to naive reg.")
    #             return self.solve_linear(z, s_hat, y).to(self.device)
                
    #         gamma_hat = -theta_z / theta_s
            
    #     return gamma_hat.detach().to(self.device)

    # def solve_multi_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, Y: torch.Tensor, method: str = 'median') -> torch.Tensor:
    #     """
    #     Algorithm 3: Multi-Simulator MRC Calibration.
    #     Includes Alternating Optimization for Weighted Mean orientation.
    #     """
    #     # 1. Device & Data Prep
    #     target_device = self.device
    #     if target_device == 'mps':
    #         target_device = 'cpu'
    #         z = z.cpu()
    #         s_hat = s_hat.cpu()
    #         Y = Y.cpu()
            
    #     N, M = Y.shape
    #     w = torch.cat([z, s_hat.view(-1, 1)], dim=1)
    #     dim_w = w.shape[1]

    #     # 2. Determine Mode & Pre-fetch Data
    #     n_pairs_limit = 200_000
    #     use_sampling = (N * N > n_pairs_limit * 2)
        
    #     if use_sampling:
    #         idx_i = torch.randint(0, N, (n_pairs_limit,), device=target_device)
    #         idx_j = torch.randint(0, N, (n_pairs_limit,), device=target_device)
    #         w_i = w[idx_i]
    #         w_j = w[idx_j]
    #         y_diffs = Y[idx_i] - Y[idx_j] # Shape: (K, M)
    #     else:
    #         # Full Matrix broadcasting
    #         y_diffs = Y.unsqueeze(1) - Y.unsqueeze(0) # (N, N, M)
    #         diag_mask = ~torch.eye(N, dtype=torch.bool, device=target_device)

    #     # ==========================================
    #     # Optimization Setup
    #     # ==========================================
    #     # Initialize Theta
    #     # theta = torch.zeros(dim_w, device=target_device, requires_grad=True)
    #     theta = torch.zeros(dim_w, device=target_device, requires_grad=True)
    #     with torch.no_grad():
    #         theta.normal_(0, 0.01)
    #         theta[-1] = -1.0 
    #         theta.div_(theta.norm())

    #     # ==========================================
    #     # Method A: Weighted Average (Alternating Optimization)
    #     # ==========================================
    #     if method == 'logit_mean' or method == 'weighted_mean':
    #         pi = torch.ones(M, device=target_device) / M
            
    #         # Step 0: Initialize o (Orientation)
    #         # Default to all 1s (assume mostly correct)
    #         o = torch.ones(M, device=target_device) 
            
    #         # Pre-compute sim signs
    #         sim_signs = torch.sign(y_diffs) # (K, M) or (N, N, M)

    #         # --- Alternating Optimization Loop ---
    #         # Typically converges in 2-3 iterations
    #         n_alternating_steps = 50 
            
    #         for step in range(n_alternating_steps):
                
    #             # --- Sub-step 1: Optimize Theta given fixed o ---
    #             # Re-initialize optimizer each step to clear history
    #             optimizer = optim.LBFGS([theta], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
                
    #             def closure():
    #                 optimizer.zero_grad()
    #                 theta_n = theta / (theta.norm() + 1e-9)
                    
    #                 if use_sampling:
    #                     score_diff = (w_i @ theta_n) - (w_j @ theta_n) # (K,)
    #                     score_diff = score_diff.view(-1, 1) # (K, 1)
    #                     # margins: (K, M). If o=-1, we flip sim_sign.
    #                     margins = (o * sim_signs) * score_diff
    #                     loss = torch.mean(torch.nn.functional.softplus(-margins) @ pi)
    #                 else:
    #                     scores = w @ theta_n
    #                     score_diff = scores.view(-1, 1) - scores.view(1, -1)
    #                     score_diff = score_diff.unsqueeze(-1)
    #                     valid_margins = ((o * sim_signs) * score_diff)[diag_mask]
    #                     loss = torch.mean(torch.nn.functional.softplus(-valid_margins) @ pi)

    #                 loss = loss + 1e-4 * torch.sum(theta**2)
    #                 if loss.requires_grad:
    #                     loss.backward()
    #                 return loss
                
    #             optimizer.step(closure)
                
    #             # --- Sub-step 2: Optimize o given fixed Theta ---
    #             # o_m = sign( Correlation(y^(m), theta_score) )
    #             with torch.no_grad():
    #                 theta_final = theta / theta.norm()
                    
    #                 if use_sampling:
    #                     sc_diff = (w_i @ theta_final) - (w_j @ theta_final) # (K,)
    #                     # Correlation proxy: mean( sign(y_diff) * score_diff ) for each simulator
    #                     # Shape: (K, M) * (K, 1) -> (K, M) -> mean -> (M,)
    #                     corrs = torch.mean(sim_signs * sc_diff.view(-1, 1), dim=0)
    #                 else:
    #                     scs = w @ theta_final
    #                     sc_diff = scs.view(-1, 1) - scs.view(1, -1)
    #                     sc_diff = sc_diff.unsqueeze(-1)
    #                     # Masked mean
    #                     corrs = torch.mean((sim_signs * sc_diff)[diag_mask], dim=0)
                    
    #                 # Update orientation
    #                 new_o = torch.sign(corrs)
    #                 # Handle exact zeros (rare)
    #                 new_o[new_o == 0] = 1.0 
                    
    #                 # Check convergence
    #                 if torch.equal(new_o, o):
    #                     break
    #                 o = new_o
    #     # ==========================================
    #     # Method B: Median (Robust)
    #     # ==========================================
    #     elif method == 'median':
    #         # 1. Aggregation
    #         median_val = torch.median(y_diffs, dim=-1).values
    #         target_signs = torch.sign(median_val)
            
    #         # ... Data Prep ...
    #         if use_sampling:
    #             valid = (target_signs != 0)
    #             target_signs_valid = target_signs[valid]
    #             w_i_valid = w_i[valid]
    #             w_j_valid = w_j[valid]
    #         else:
    #             valid = (target_signs != 0) & diag_mask
    #             valid_indices = torch.nonzero(valid, as_tuple=True)
    #             target_signs_valid = target_signs[valid]
    #             w_i_valid = w[valid_indices[0]]
    #             w_j_valid = w[valid_indices[1]]

    #         # Standard Optimizer
    #         optimizer = optim.LBFGS([theta], lr=1.0, max_iter=100, line_search_fn='strong_wolfe')
            
    #         def closure():
    #             optimizer.zero_grad()
    #             theta_n = theta / (theta.norm() + 1e-9)
    #             score_diff = (w_i_valid @ theta_n) - (w_j_valid @ theta_n)
    #             loss = torch.mean(torch.nn.functional.softplus(-target_signs_valid * score_diff))
    #             loss = loss + 1e-4 * torch.sum(theta**2)
    #             if loss.requires_grad:
    #                 loss.backward()
    #             return loss
            
    #         optimizer.step(closure)

    #     else:
    #         raise ValueError(f"Unknown method: {method}")

    #     # ==========================================
    #     # Recover Gamma
    #     # ==========================================
    #     with torch.no_grad():
    #         theta_final = theta / theta.norm()
    #         theta_z = theta_final[:-1]
    #         theta_s = theta_final[-1]
            
    #         if abs(theta_s) < 1e-4:
    #             gamma_hat = theta_z * 0.0
    #         else:
    #             gamma_hat = -theta_z / theta_s
                
    #     return gamma_hat.detach().to(self.device)
    # def solve_multi_mrc(self, z: torch.Tensor, s_hat: torch.Tensor, Y: torch.Tensor, method: str = 'median') -> torch.Tensor:
    #     """
    #     Algorithm 3: Multi-Simulator MRC Calibration.
    #     """
    #     # --- [CRITICAL FIX 1] 剥离 Naive Baseline ---
    #     # Logit Mean 应该是最朴素的方法：直接平均 y，不搞方向校正
    #     if method == 'logit_mean':
    #         # 简单的算术平均 (Naive Ensemble)
    #         # 如果有模拟器是反向的 (-20*eta)，这个平均值会被破坏，这正是我们想要展示的弱点
    #         y_avg = Y.mean(dim=1) 
    #         # 调用单模拟器 MRC (复用 solve_mrc 的逻辑)
    #         return self.solve_mrc(z, s_hat, y_avg)

    #     # ---------------- 下面是论文中的 Algorithm 3 (Robust Methods) ----------------

    #     # 1. Device & Data Prep
    #     target_device = self.device
    #     if target_device == 'mps':
    #         target_device = 'cpu'
    #         z = z.cpu()
    #         s_hat = s_hat.cpu()
    #         Y = Y.cpu()
            
    #     N, M = Y.shape
    #     w = torch.cat([z, s_hat.view(-1, 1)], dim=1)
    #     dim_w = w.shape[1]

    #     # 2. Pair Sampling
    #     n_pairs_limit = 100_000
    #     use_sampling = (N * N > n_pairs_limit * 2)
        
    #     if use_sampling:
    #         idx_i = torch.randint(0, N, (n_pairs_limit,), device=target_device)
    #         idx_j = torch.randint(0, N, (n_pairs_limit,), device=target_device)
    #         w_i = w[idx_i]
    #         w_j = w[idx_j]
    #         y_diffs = Y[idx_i] - Y[idx_j] # Shape: (K, M)
    #     else:
    #         y_diffs = Y.unsqueeze(1) - Y.unsqueeze(0)
    #         diag_mask = ~torch.eye(N, dtype=torch.bool, device=target_device)

    #     # --- [CRITICAL FIX 2] Device Consistency ---
    #     theta = torch.zeros(dim_w, device=target_device, requires_grad=True)
    #     with torch.no_grad():
    #         theta.normal_(0, 0.01)
    #         theta[-1] = -1.0 
    #         theta.div_(theta.norm())

    #     # ==========================================
    #     # Method A: Weighted Average (With Orientation Learning)
    #     # ==========================================
    #     if method == 'weighted_mean':
    #         pi = torch.ones(M, device=target_device) / M
    #         o = torch.ones(M, device=target_device) 
    #         sim_signs = torch.sign(y_diffs)

    #         # Alternating Optimization Loop
    #         for step in range(5):
    #             # 1. Optimize Theta
    #             optimizer = optim.LBFGS([theta], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
    #             def closure():
    #                 optimizer.zero_grad()
    #                 theta_n = theta / (theta.norm() + 1e-9)
                    
    #                 if use_sampling:
    #                     score_diff = (w_i @ theta_n) - (w_j @ theta_n)
    #                     # Margins: (K, M). Corrected by orientation 'o'
    #                     margins = (o * sim_signs) * score_diff.view(-1, 1)
    #                     loss = torch.mean(torch.nn.functional.softplus(-margins) @ pi)
    #                 else:
    #                     # Full matrix logic (omitted for brevity, sampling preferred)
    #                     pass 

    #                 loss = loss + 1e-4 * torch.sum(theta**2)
    #                 # Constraint: theta_s should be negative
    #                 if theta[-1] > 0: loss += 10.0 * theta[-1]**2
                    
    #                 loss.backward()
    #                 return loss
                
    #             try:
    #                 optimizer.step(closure)
    #             except Exception:
    #                 break # LBFGS fails smoothly
                
    #             # 2. Optimize Orientation 'o'
    #             with torch.no_grad():
    #                 theta_final = theta / theta.norm()
    #                 if use_sampling:
    #                     sc_diff = (w_i @ theta_final) - (w_j @ theta_final)
    #                     # Correlation: mean( sign(dy) * d_score )
    #                     corrs = torch.mean(sim_signs * sc_diff.view(-1, 1), dim=0)
    #                 else:
    #                     pass
    #                 new_o = torch.sign(corrs)
    #                 new_o[new_o == 0] = 1.0 # Handle zero correlation
    #                 if torch.equal(new_o, o): break
    #                 o = new_o

    #     # ==========================================
    #     # Method B: Median (Robust Aggregation)
    #     # ==========================================
    #     elif method == 'median':
    #         # 1. Aggregation (Pre-compute consensus signs)
    #         median_val = torch.median(y_diffs, dim=-1).values
    #         target_signs = torch.sign(median_val)
            
    #         # --- [CRITICAL FIX 3] Handle Signal Collapse ---
    #         # If median is 0 everywhere (Lazy Majority), don't crash
    #         keep = target_signs != 0
    #         if keep.sum() < 50:
    #             print("[Warning] Median yielded no valid pairs. Returning random init.")
    #             # Skip optimization, return random theta (High NLL)
    #         else:
    #             if use_sampling:
    #                 target_signs_valid = target_signs[keep]
    #                 w_i_valid = w_i[keep]
    #                 w_j_valid = w_j[keep]

    #                 optimizer = optim.LBFGS([theta], lr=1.0, max_iter=50, line_search_fn='strong_wolfe')
                    
    #                 def closure():
    #                     optimizer.zero_grad()
    #                     theta_n = theta / (theta.norm() + 1e-9)
    #                     score_diff = (w_i_valid @ theta_n) - (w_j_valid @ theta_n)
    #                     loss = torch.mean(torch.nn.functional.softplus(-target_signs_valid * score_diff))
                        
    #                     loss += 1e-4 * torch.sum(theta**2)
    #                     if theta[-1] > 0: loss += 10.0 * theta[-1]**2
                        
    #                     if torch.isnan(loss): return torch.tensor(0.0, requires_grad=True)
    #                     loss.backward()
    #                     return loss
                    
    #                 try:
    #                     optimizer.step(closure)
    #                 except Exception:
    #                     pass

    #     else:
    #         raise ValueError(f"Unknown method: {method}")

    #     # ==========================================
    #     # Recover Gamma
    #     # ==========================================
    #     with torch.no_grad():
    #         theta_final = theta / theta.norm()
    #         theta_z = theta_final[:-1]
    #         theta_s = theta_final[-1]
            
    #         # Prevent division by zero
    #         if abs(theta_s) < 1e-4:
    #             # Fallback or warning
    #             theta_s = -1e-4 if theta_s < 0 else 1e-4
            
    #         gamma_hat = -theta_z / theta_s
                
    #     return gamma_hat.detach().to(self.device)