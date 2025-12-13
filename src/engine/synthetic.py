import torch
import numpy as np
from typing import Dict, Any, Tuple

from ..config import ExpConfig
from ..utils.data_structs import TensorBatch

from ..modules.sources import SourcePipeline
from ..modules.z_mappers import BaseZMapper
from ..modules.u_mappers import BaseUtilityMapper, LinearUtilityMapper
from ..modules.y_mappers import BaseYMapper


class SyntheticDataEngine:
    """
    The Orchestrator.

    Features:
    1. Parameter Scaling (1/sqrt(d)) for Assumption 6.
    2. Finite Retry Mechanism for Assumptions 2 & 3 & 6.
    3. Robust CPU-based Validation.
    """

    def __init__(
        self,
        cfg: ExpConfig,
        source_pipeline: SourcePipeline,
        z_mapper: BaseZMapper,
        u_mapper: BaseUtilityMapper,
        y_mapper: BaseYMapper,
        norm: bool = False,
    ):

        self.cfg = cfg
        self.source = source_pipeline
        self.z_mapper = z_mapper
        self.u_mapper = u_mapper
        self.y_mapper = y_mapper

        # --- Latent Truth (God's Parameters) ---
        # Scaling for Assumption 6
        scaling_factor = 1.0
        if norm or cfg.dim_z > 1:
            scaling_factor = 1.0 / np.sqrt(cfg.dim_z)

        self.gamma_star = torch.randn(cfg.dim_z, device=cfg.device) * scaling_factor

    def generate(self) -> Dict[str, Any]:
        """
        Public interface: Tries to generate valid data up to `max_retries` times.
        """
        max_retries = 10

        for attempt in range(max_retries):
            # 1. 尝试生成一次
            data_dict, validation_meta = self._generate_once()

            # 2. 检查假设
            z = data_dict["inputs"]["z"]
            s = data_dict["inputs"]["s_hat"]
            eta = data_dict["truth"]["eta"]

            is_valid, fail_reason = self._validate_assumptions(z, s, eta)

            if is_valid:
                return data_dict
            else:
                # 只有调试时才打印
                # if attempt == max_retries - 1:
                #     print(f"[Engine] Generation failed last attempt: {fail_reason}")
                pass

        # 3. 失败抛出异常
        raise RuntimeError(
            f"Data generation failed assumptions {max_retries} times.\n"
            f"Last Reason: {fail_reason}"
        )

    def _generate_once(self) -> Tuple[Dict[str, Any], Dict]:
        """
        Internal: Single pass of generation logic.
        """
        # 1. Source Generation (S -> X)
        batch: TensorBatch = self.source()

        # 2. Feature Mapping (X -> z)
        z_val = self.z_mapper(batch)

        # 3. Utility Mapping (X -> u)
        u_val = self.u_mapper(batch)

        # 4. Calculate Latent Truth
        s_true = torch.logsumexp(u_val, dim=1)
        u0_true = z_val @ self.gamma_star
        eta_true = u0_true - s_true
        p0_true = torch.sigmoid(eta_true)

        # 5. Simulator Process (eta* -> y)
        y_obs = self.y_mapper(eta_true)

        # 6. Estimation Error Injection
        s_hat, beta_hat_export = self._inject_estimation_error(u_val, s_true, batch)

        data = {
            "inputs": {"z": z_val, "s_hat": s_hat, "y": y_obs, "batch": batch},
            "truth": {
                "p0": p0_true,
                "gamma": self.gamma_star,
                "s": s_true,
                "eta": eta_true,
                "u0": u0_true,
                "beta_hat": beta_hat_export,
            },
        }
        return data, {}

    def _inject_estimation_error(self, u_val, s_true, batch):
        """Helper to handle utility noise logic."""
        sigma = self.cfg.est_noise_sigma
        dist_type = self.cfg.noise_distribution
        beta_hat_export = None
        s_hat = s_true

        if self.cfg.utility_mode == "structural":
            if isinstance(self.u_mapper, LinearUtilityMapper):
                beta_true = self.u_mapper.beta
                if sigma > 0:
                    d = self.cfg.dim_item_feat
                    if dist_type == "gaussian":
                        raw = torch.randn(d, device=self.cfg.device)
                        delta_beta = raw / (raw.norm() + 1e-9) * sigma
                    elif dist_type == "uniform":
                        bound = sigma / np.sqrt(d)
                        delta_beta = (
                            torch.rand(d, device=self.cfg.device) * 2 - 1
                        ) * bound
                    else:
                        delta_beta = torch.zeros_like(beta_true)

                    beta_hat_export = beta_true + delta_beta

                    # Propagate error
                    delta_u = torch.einsum("nld,d->nl", batch.items, delta_beta)
                    delta_u = delta_u * batch.mask
                    u_hat = u_val + delta_u
                    s_hat = torch.logsumexp(u_hat, dim=1)
                else:
                    beta_hat_export = beta_true

        elif self.cfg.utility_mode == "additive":
            if hasattr(self.u_mapper, "beta"):
                beta_hat_export = self.u_mapper.beta

            if sigma > 0:
                if dist_type == "gaussian":
                    noise = torch.randn_like(s_true) * sigma
                elif dist_type == "uniform":
                    noise = (torch.rand_like(s_true) * 2 - 1) * sigma
                s_hat = s_true + noise

        return s_hat, beta_hat_export

    def _validate_assumptions(self, z, s, eta) -> Tuple[bool, str]:
        """
        [Updated] Runs checks on CPU + Double Precision to avoid MPS crashes.
        """
        # --- Critical: Move to CPU/Double for numerical stability check ---
        # MPS eigvalsh is notorious for crashing on perfectly valid matrices.
        try:
            z_cpu = z.detach().cpu().double()
            s_cpu = s.detach().cpu().double().view(-1, 1)
            eta_cpu = eta.detach().cpu().double()
        except Exception as e:
            return False, f"Validation Data Transfer Failed: {e}"

        N = z_cpu.shape[0]

        # --- 1. Check Assumption 2: Covariance Non-degeneracy ---
        W = torch.cat([z_cpu, s_cpu], dim=1)
        W_mean = W.mean(dim=0, keepdim=True)
        W_centered = W - W_mean
        Sigma = (W_centered.T @ W_centered) / (N - 1)

        try:
            eigvals = torch.linalg.eigvalsh(Sigma)
            min_eig = eigvals.min().item()
            check_min_eig = getattr(self.cfg, "check_min_eig", 1e-6)  # Relaxed slightly

            if min_eig < check_min_eig:
                return (
                    False,
                    f"Asm 2: Covariance singular (min_eig={min_eig:.2e} < {check_min_eig})",
                )
        except Exception as e:
            # If this still crashes on CPU, the data contains NaNs/Infs
            if torch.isnan(Sigma).any():
                return False, "Asm 2: Sigma contains NaNs."
            if torch.isinf(Sigma).any():
                return False, "Asm 2: Sigma contains Infs."
            return False, f"Asm 2: Eigenvalue check crashed on CPU: {e}"

        # --- 2. Check Assumption 6: Logit Dispersion ---
        eta_std = eta_cpu.std().item()
        # Relaxed ranges for robustness
        min_std = getattr(self.cfg, "check_min_logit_std", 0.2)
        max_std = getattr(self.cfg, "check_max_logit_std", 8.0)

        if eta_std < min_std:
            return False, f"Asm 6: Logit variance too low ({eta_std:.2f} < {min_std})"
        if eta_std > max_std:
            return False, f"Asm 6: Logit variance too high ({eta_std:.2f} > {max_std})"

        # --- 3. Check Assumption 3: Simulator Slope ---
        if hasattr(self.cfg, "sim_bias_b"):
            if abs(self.cfg.sim_bias_b) < 1e-3:
                return False, "Asm 3: Simulator slope is zero."

        return True, "OK"
