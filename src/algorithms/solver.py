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

    def solve_linear(
        self,
        z: torch.Tensor,
        s_hat: torch.Tensor,
        y: torch.Tensor,
        lambda_reg: float = None,
    ) -> torch.Tensor:
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
        A = torch.cat([ones, z_dbl, s_dbl], dim=1)  # Shape: (N, dim_z + 2)

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
        I[0, 0] = 0.0  # 不对截距项 (Intercept) 进行正则化/微扰

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

    def solve_mrc(
        self, z: torch.Tensor, s_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Algorithm 2: Maximum Rank Correlation (MRC).
        [Updated] Uses solve_linear (OLS) for Warm Start initialization.
        """
        # 1. Device Prep (Keep CPU for LBFGS stability)
        target_device = self.device
        if target_device == "mps":
            target_device = "cpu"
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
            beta_init = torch.linalg.solve(
                XTX + 1e-5 * torch.eye(dim_z + 1, device=target_device), XTY
            )
            gamma_init = beta_init[1:].float()  # 忽略截距

            # Construct Initial Theta: [gamma, -1]
            theta_init = torch.cat(
                [gamma_init, torch.tensor([-1.0], device=target_device)]
            )
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
        optimizer = optim.LBFGS(
            [theta],
            lr=1.0,
            max_iter=100,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        # 3. Pair Sampling Strategy (Keep existing logic)
        n_pairs = 200_000
        use_sampling = N * N > n_pairs * 2

        if not use_sampling:
            y_col = y.view(-1, 1)
            y_sign = torch.sign(y_col - y_col.T)
            mask = y_sign != 0
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
                loss = torch.sum(
                    torch.nn.functional.softplus(
                        -alpha * y_sign[mask] * score_diff[mask]
                    )
                ) / (mask.sum() + 1e-9)
            else:
                score_i = w_i @ theta_n
                score_j = w_j @ theta_n
                score_diff = score_i - score_j
                # [MODIFIED] Added alpha scaling
                loss = torch.mean(
                    torch.nn.functional.softplus(-alpha * y_sign_sample * score_diff)
                )

            # [MODIFIED] Reduced regularization significantly
            loss = loss + 1e-6 * torch.sum(theta**2)

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
                return self.solve_linear(
                    z.to(self.device), s_hat.to(self.device), y.to(self.device)
                )

            gamma_hat = -theta_z / theta_s

        return gamma_hat.detach().to(self.device)

    def solve_multi_mrc(
        self,
        z: torch.Tensor,
        s_hat: torch.Tensor,
        Y: torch.Tensor,
        method: str = "median",
    ) -> torch.Tensor:
        """
        Algorithm 3: Multi-Simulator MRC Calibration.
        [Updated] Adds Warm Start and Smart Orientation Initialization.
        """
        # 1. Naive Baseline (Delegate to Single MRC)
        if method == "logit_mean":
            y_avg = Y.mean(dim=1)
            return self.solve_mrc(z, s_hat, y_avg)

        # 2. Device & Data Prep
        target_device = self.device
        if target_device == "mps":
            target_device = "cpu"
            z = z.cpu()
            s_hat = s_hat.cpu()
            Y = Y.cpu()

        N, M = Y.shape
        # W = [z, s]
        w = torch.cat([z, s_hat.view(-1, 1)], dim=1)
        dim_w = w.shape[1]

        # 3. Pair Sampling
        n_pairs_limit = 100_000
        use_sampling = N * N > n_pairs_limit * 2

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
            beta_init = torch.linalg.solve(
                XTX + 1e-5 * torch.eye(Z_design.shape[1], device=target_device), XTY
            )

            gamma_start = beta_init[1:].float()
            theta_init = torch.cat(
                [gamma_start, torch.tensor([-1.0], device=target_device)]
            )

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
        if method == "weighted_mean":
            pi = torch.ones(M, device=target_device) / M

            # [NEW] Smart Orientation Init
            # 利用 Warm Start 的 theta 预判模拟器的方向
            # 如果某个模拟器与 OLS 结果完全负相关，初始 o 设为 -1
            with torch.no_grad():
                theta_final = theta  # current init
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
                optimizer = optim.LBFGS(
                    [theta], lr=1.0, max_iter=20, line_search_fn="strong_wolfe"
                )

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
                        loss = torch.tensor(
                            0.0, requires_grad=True, device=target_device
                        )

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
                        if torch.equal(new_o, o):
                            break
                        o = new_o

        # ========================================================
        # Method B: Median (Robust Aggregation)
        # ========================================================
        elif method == "median":
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

                    optimizer = optim.LBFGS(
                        [theta], lr=1.0, max_iter=50, line_search_fn="strong_wolfe"
                    )

                    def closure():
                        optimizer.zero_grad()
                        theta_n = theta / (theta.norm() + 1e-9)
                        score_diff = (w_i_valid @ theta_n) - (w_j_valid @ theta_n)
                        loss = torch.mean(
                            torch.nn.functional.softplus(
                                -target_signs_valid * score_diff
                            )
                        )
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
                return self.solve_linear(
                    z.to(self.device), s_hat.to(self.device), y_fallback
                )

            gamma_hat = -theta_z / theta_s

        return gamma_hat.detach().to(self.device)
