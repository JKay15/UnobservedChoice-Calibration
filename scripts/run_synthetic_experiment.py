import sys
import os
import time
import datetime
import itertools
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dataclasses import replace
from matplotlib.ticker import ScalarFormatter, NullFormatter

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import ExpConfig
from src.engine.factory import EngineFactory
from src.algorithms.solver import CalibrationSolver
from src.modules.y_mappers import MonotoneYMapper
from src.utils.metrics import compute_p0_from_logits

# Imports for optimization and regret calculation
from src.utils.optimization import solve_optimal_assortment, calculate_revenue

# ==========================================
# Global Setup
# ==========================================
RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================================
# [AESTHETICS] Global Style Configuration
# ==========================================
import matplotlib.ticker as ticker

# 1. 颜色映射 (Color Map) - 语义绑定
COLOR_MAP = {
    "mrc": "tab:red",  # 红色
    "linear": "tab:blue",  # 蓝色
    "median": "tab:green",  # 绿色
    "weighted": "tab:orange",  # 橙色
    "mean": "tab:purple",  # 紫色
    "naive": "tab:gray",  # 灰色
    "sim": "tab:gray",  # 灰色
    "default": "black",
}

# 2. 纹理库 (Texture Bank) - 字典序分配
# 逻辑: (LineStyle, Marker)
# [User Request]: Index 3 (4th) is 'x', Index 4 (5th) is 'D'
TEXTURE_BANK = [
    ("-", "o"),  # 1. 实线 + 圆圈 (最强)
    ("--", "s"),  # 2. 虚线 + 方块
    ("-.", "^"),  # 3. 点划线 + 三角
    (":", "x"),  # 4. 点线 + 叉号 (Swapped)
    ("-", "D"),  # 5. 实线 + 菱形 (Swapped)
    ("--", "*"),  # 6. 虚线 + 星号
    ("-.", "v"),  # 7. 点划线 + 倒三角
    (":", "P"),  # 8. 点线 + 加号
]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 1. Unified Atomic Trial Runner
# ==========================================
def run_single_trial(
    cfg: ExpConfig,
    task_type: str = "standard",
    algo_type: str = "mrc",
    multi_sim_method: str = "median",
    regret_need: bool = False,
    y_type: str = "linear",
    z_type: str = "independent",  # [UPDATED DEFAULT]
    u_type: str = "linear",
    context_type: str = "concat",
    z_model_path: str = None,
    norm: bool = True,
    sim_noise_type: str = "gaussian",
) -> Dict[str, float]:  # norm=True for theory validation
    trial_cfg = replace(cfg, sim_noise_type=sim_noise_type)
    # 1. Build Engine & Generate Data
    engine = EngineFactory.build_synthetic_engine(
        cfg,
        z_type=z_type,
        u_type=u_type,
        y_type=y_type,
        context_mapper_type=context_type,
        norm=norm,
    )
    # Generate data (Auto-validates assumptions internally)
    data = engine.generate()

    inputs = data["inputs"]
    truth = data["truth"]
    solver = CalibrationSolver(cfg)

    avg_regret = float("nan")
    duration = 0.0

    # --- [NEW] Oracle Logic Handling ---
    # Detect if this is an oracle run
    is_oracle = ("oracle" in algo_type) or ("oracle" in multi_sim_method)
    # Strip suffix to get the actual solver method ('linear' or 'mrc')
    base_algo = (
        algo_type.replace("_oracle", "")
        if task_type != "multi_sim"
        else multi_sim_method.replace("_oracle", "")
    )
    # Select Input Data:
    # Standard: Use estimated s_hat (contains noise)
    # Oracle: Use true s (perfect utility knowledge)
    s_input = truth["s"] if is_oracle else inputs["s_hat"]

    # ==========================================
    # Branch A: Multi-Simulator
    # ==========================================
    if task_type == "multi_sim" and (
        multi_sim_method in ["logit_mean", "weighted_mean", "median"]
    ):
        eta_true = truth["eta"]

        # Simulators with varying quality
        # Sim 1: Good
        mapper1 = MonotoneYMapper(cfg)
        # Sim 2: Slight Shift
        mapper2 = MonotoneYMapper(
            replace(
                cfg, sim_bias_a=cfg.sim_bias_a + 0.5, sim_bias_b=cfg.sim_bias_b * 0.5
            )
        )
        # Sim 3: Another Shift
        mapper3 = MonotoneYMapper(
            replace(
                cfg, sim_bias_a=cfg.sim_bias_a - 0.5, sim_bias_b=cfg.sim_bias_b * 0.5
            )
        )
        # Sim 4: High Variance
        mapper4 = MonotoneYMapper(
            replace(
                cfg, sim_bias_a=cfg.sim_bias_a + 1.0, sim_bias_b=cfg.sim_bias_b * 1.0
            )
        )

        # Sim 5: [Adversarial] Anti-Monotone or Pure Noise
        # This is key to showing Median robustness
        mapper5 = MonotoneYMapper(replace(cfg, sim_bias_a=0.0, sim_bias_b=-2.5))

        # Generate Ys
        y1 = mapper1(eta_true) + torch.randn_like(eta_true) * 0.5
        y2 = mapper2(eta_true) + torch.randn_like(eta_true) * 0.5
        y3 = mapper3(eta_true) + torch.randn_like(eta_true) * 0.5
        y4 = mapper4(eta_true) + torch.randn_like(eta_true) * 1.0
        y5 = mapper5(eta_true) + torch.randn_like(eta_true) * (-2.5)

        Y_multi = torch.stack([y1, y2, y3, y4, y5], dim=1)

        start_time = time.time()
        gamma_hat = solver.solve_multi_mrc(
            inputs["z"], s_input, Y_multi, method=multi_sim_method
        )
        duration = time.time() - start_time

        with torch.no_grad():
            p0_pred = compute_p0_from_logits(inputs["z"], s_input, gamma_hat)

    # ==========================================
    # Branch B: Standard Calibration (+ Regret)
    # ==========================================
    else:
        start_time = time.time()
        if base_algo == "linear":
            gamma_hat = solver.solve_linear(inputs["z"], s_input, inputs["y"])
        elif base_algo == "mrc":
            gamma_hat = solver.solve_mrc(inputs["z"], s_input, inputs["y"])
        else:
            raise ValueError(f"Unknown algo_type: {algo_type}")
        duration = time.time() - start_time

        with torch.no_grad():
            p0_pred = compute_p0_from_logits(inputs["z"], s_input, gamma_hat)

        # ==========================================
        # 3. Downstream Assortment Regret (Test Phase)
        # ==========================================
        if regret_need:
            # 只有 Linear Utility 才能精确定义 beta 误差
            if u_type != "linear":
                pass  # nan
            else:
                n_test_decisions = 100
                n_items_pool = 50

                # 1. 获取 True Beta
                beta_true = engine.u_mapper.beta.detach()

                # [NEW] Oracle Regret Logic
                # If Oracle: We assume we know True Beta (Perfect Stage 1)
                # If Standard: We use the Noisy Beta (Imperfect Stage 1)
                if is_oracle:
                    beta_hat = beta_true
                else:
                    beta_hat = truth.get("beta_hat")
                    if beta_hat is None:
                        beta_hat = beta_true

                beta_hat = beta_hat.to(cfg.device)

                regret_list = []

                for _ in range(n_test_decisions):
                    # A. 生成随机测试环境 (Context Z & Items)
                    # [Attention] Test Z must come from same distribution (Independent Gaussian)
                    if cfg.dim_context > 0:
                        ctx = torch.randn(1, cfg.dim_context, device=cfg.device).clamp_(
                            -3, 3
                        )
                        z_test = engine.z_mapper.proj(ctx).squeeze(0)
                    else:
                        z_test = torch.randn(cfg.dim_z, device=cfg.device)

                    items_x = torch.randn(
                        n_items_pool, cfg.dim_item_feat, device=cfg.device
                    )
                    prices = torch.rand(n_items_pool, device=cfg.device) * 90 + 10

                    # B. 计算 Utilities
                    u_items_true = items_x @ beta_true
                    u_items_hat = items_x @ beta_hat  # Plug-in

                    # C. Optimization (决策)
                    # Oracle: True Gamma, True Beta
                    mask_opt, r_opt = solve_optimal_assortment(
                        truth["gamma"], z_test, prices, u_items_true
                    )

                    # Estimated: Hat Gamma, Hat Beta
                    # "误差抵消" (Error Cancellation) should happen here if calibrated correctly
                    mask_hat, _ = solve_optimal_assortment(
                        gamma_hat, z_test, prices, u_items_hat
                    )

                    # D. Evaluation (评估)
                    # Realized Revenue evaluated on TRUE parameters
                    r_hat_realized = calculate_revenue(
                        mask_hat, truth["gamma"], z_test, prices, u_items_true
                    )

                    # E. Regret Calculation
                    if r_opt > 1e-6:
                        regret = (r_opt - r_hat_realized) / r_opt
                    else:
                        regret = 0.0

                    regret_list.append(regret)

                avg_regret = np.mean(regret_list) * 100.0

    return {
        "gamma_hat": gamma_hat,
        "gamma_true": truth["gamma"],
        "p0_pred": p0_pred,
        "p0_true": truth["p0"],
        "time": duration,
        "regret": avg_regret,
    }


# ==========================================
# 2. Universal Experiment Runner (Corrected)
# ==========================================
def run_experiment_grid(
    base_cfg: ExpConfig,
    x_axis_name: str,
    x_values: List[Any],
    compare_axis_name: Any = None,
    compare_values: List[Any] = None,
    cross_product: bool = True,
    n_seeds: int = 5,
    regret_need: bool = False,
    norm: bool = True,  # [Default True] for Assumption 6
    sim_noise_type: str = "gaussian",
    # Defaults
    default_task_type: str = "standard",
    default_algo_type: str = "mrc",
    default_multi_sim_method: str = "median",
    default_y_type: str = "monotone",
    default_z_type: str = "independent",  # [CRITICAL UPDATE]
    default_u_type: str = "linear",
    default_context_type: str = "concat",
) -> pd.DataFrame:

    # ... (Keep existing iteration logic unchanged) ...
    # Just copying the loop structure for completeness

    if compare_axis_name is None:
        comp_keys = []
        comp_iter = [()]
    elif isinstance(compare_axis_name, str):
        comp_keys = [compare_axis_name]
        comp_iter = [(v,) for v in compare_values]
    else:
        comp_keys = compare_axis_name
        if cross_product:
            comp_iter = list(itertools.product(*compare_values))
        else:
            comp_iter = compare_values

    print(f"\n=== Running Grid: X={x_axis_name} | Cross Validating: {comp_keys} ===")

    records = []
    total_iters = len(x_values) * len(comp_iter) * n_seeds
    pbar = tqdm(total=total_iters, desc="Progress")

    for x_val in x_values:
        for comp_vals in comp_iter:
            current_comp_params = dict(zip(comp_keys, comp_vals))

            if not comp_keys:
                combo_label = "Default"
            else:
                combo_label = "-".join([str(v) for v in comp_vals])

            for seed in range(n_seeds):
                current_cfg_args = {"seed": seed + 1000, x_axis_name: x_val}

                params = {
                    "task_type": default_task_type,
                    "algo_type": default_algo_type,
                    "multi_sim_method": default_multi_sim_method,
                    "y_type": default_y_type,
                    "z_type": default_z_type,
                    "u_type": default_u_type,
                    "context_type": default_context_type,
                    "regret_need": regret_need,
                    "norm": norm,
                    "sim_noise_type": sim_noise_type,
                }

                if x_axis_name in params:
                    params[x_axis_name] = x_val

                for k, v in current_comp_params.items():
                    if k in params:
                        params[k] = v

                cfg_args = current_cfg_args.copy()
                for k, v in params.items():
                    if hasattr(base_cfg, k):
                        cfg_args[k] = v

                if hasattr(base_cfg, x_axis_name):
                    cfg_args[x_axis_name] = x_val

                cfg = ExpConfig(**{**base_cfg.__dict__, **cfg_args})
                set_seed(cfg.seed)
                try:
                    # Run Trial with Error Handling
                    metrics = run_single_trial(
                        cfg,
                        task_type=params["task_type"],
                        algo_type=params["algo_type"],
                        multi_sim_method=params["multi_sim_method"],
                        regret_need=params["regret_need"],
                        y_type=params["y_type"],
                        z_type=params["z_type"],
                        u_type=params["u_type"],
                        context_type=params["context_type"],
                        norm=params["norm"],
                        sim_noise_type=params["sim_noise_type"],
                    )

                    record = {
                        x_axis_name: x_val,
                        "seed": seed,
                        "combo_label": combo_label,
                        **current_comp_params,
                        **metrics,
                    }
                    record["n_samples"] = cfg.n_samples
                    records.append(record)

                except RuntimeError as e:
                    # If Engine max_retries failed
                    print(f"\n[Skip] Trial failed: {e}")

                pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)

    save_name = f"grid_{x_axis_name}"
    if comp_keys:
        comp_str = "_".join(comp_keys)
        save_name += f"_vs_{comp_str}"

    pkl_path = LOG_DIR / f"{save_name}_{RUN_ID}.pkl"
    df.to_pickle(pkl_path)
    print(f"Saved full data to: {pkl_path}")

    return df


# ==========================================
# 3. Universal Plotter (Kept mostly same)
# ==========================================

def plot_metric_scaling(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_label: str,
    hue_col: Optional[str] = None,
    title: str = "Scaling Analysis",
    y_top_margin: Optional[float] = None,
    legend_loc: str = "upper right",
    log_y: Optional[bool] = None,
    log_x: Optional[bool] = None,
):
    print(f"\n>>> [Plot] {y_label} vs {x_label}...")

    # --- 1. Data Prep (Unchanged) ---
    tensor_cols = ["p0_pred", "p0_true", "gamma_hat", "gamma_true"]

    def move_to_cpu(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu()
        return val

    plot_df = df.copy()
    for col in tensor_cols:
        if col in plot_df.columns:
            if len(plot_df) > 0 and isinstance(plot_df[col].iloc[0], torch.Tensor):
                plot_df[col] = plot_df[col].apply(move_to_cpu)

    # --- 2. Metric Computation (Unchanged) ---
    def compute_pooled_metric(sub_df):
        if len(sub_df) == 0:
            return float("nan")

        if "p0" in y_label or "nll" in y_label:
            all_preds = torch.cat(list(sub_df["p0_pred"])).float().numpy()
            all_trues = torch.cat(list(sub_df["p0_true"])).float().numpy()

            if "nll" in y_label:
                epsilon = 1e-7
                all_preds = np.clip(all_preds, epsilon, 1 - epsilon)
                nll = -(
                    all_trues * np.log(all_preds)
                    + (1 - all_trues) * np.log(1 - all_preds)
                )
                return np.mean(nll)
            elif "error" in y_label:
                abs_diff = np.abs(all_preds - all_trues)
                return np.quantile(abs_diff, 0.70)

        elif "gamma" in y_label:
            hat_matrix = torch.stack(list(sub_df["gamma_hat"]))
            true_matrix = torch.stack(list(sub_df["gamma_true"]))
            l2_errors = torch.norm(hat_matrix - true_matrix, p=2, dim=1)
            return l2_errors.mean().item()

        elif "regret" in y_label:
            valid_regrets = sub_df["regret"].dropna()
            return valid_regrets.mean() if not valid_regrets.empty else float("nan")
        return 0.0

    if plot_df.empty:
        print("[Warning] No data to plot.")
        return

    df_agg = (
        plot_df.groupby([x_col, hue_col] if hue_col else [x_col])
        .apply(compute_pooled_metric, include_groups=False)
        .reset_index(name=y_label)
    )

    # --- 3. Style Setup ---
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 20,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "lines.linewidth": 2.5,
            "lines.markersize": 9,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
        }
    )

    LABEL_MAP = {
        "n_samples": "n",
        "est_noise_sigma": x_label,
        "sim_noise_sigma": r"$\sigma_{\epsilon}$",
        "dim_z": "d",
        "max_assortment_size": r"$|\mathcal{S}|_{\max}$",
        "gamma_error": "Parameter Estimation Error",
        "p0_error": "Empirical p0 Error",
        "nll": "Negative Log-Likelihood",
        "regret": "Sub-optimality (%)",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- 4. Plotting Loop (UPDATED LOGIC) ---
    # Sort hues alphabetically to ensure deterministic order
    unique_hues = sorted(df_agg[hue_col].unique()) if hue_col else [None]

    for idx, hue_val in enumerate(unique_hues):
        if hue_val:
            subset = df_agg[df_agg[hue_col] == hue_val].sort_values(x_col)
            label_str = str(hue_val)
        else:
            subset = df_agg.sort_values(x_col)
            label_str = "Default"

        # [NEW LOGIC START] =========================================

        # A. Texture by Dictionary Order (Strict Indexing)
        # 无论是什么 Label，只要排第几，就用第几号纹理
        tex_idx = idx % len(TEXTURE_BANK)
        ls, marker = TEXTURE_BANK[tex_idx]

        # B. Color by Semantic Name
        s_lower = label_str.lower()
        color = COLOR_MAP["default"]
        # Find matching color key
        for k, v in COLOR_MAP.items():
            if k in s_lower:
                color = v
                break

        # C. Label Cleanup
        # e.g. "linear-monotone" -> "Linear (Monotone Sim)"
        clean_label = label_str.replace("_", " ").title()
        clean_label = clean_label.replace("Mrc", "MRC").replace("Linear", "Lin")
        if "-" in clean_label:
            parts = clean_label.split("-")
            clean_label = f"{parts[0]} ({parts[1]} Sim)"

        # Oracle cleanup
        if "Oracle" in clean_label:
            # 保持纹理分配不变，只在图例文字上做点微调（可选）
            pass

        # [NEW LOGIC END] ===========================================

        ax.plot(
            subset[x_col],
            subset[y_label],
            label=clean_label,
            color=color,
            linestyle=ls,
            marker=marker,
            alpha=0.9,
        )

    # --- 5. Axes & Ticks Logic (Preserved) ---
    # X-Axis Log Logic
    use_log_x = False
    if log_x is not None:
        use_log_x = log_x
        x_min, x_max = df_agg[x_col].min(), df_agg[x_col].max()
        if x_col == "n_samples" or "sigma" in x_col:
            if x_min > 1e-9 and (x_max / x_min > 10):
                use_log_x = True
    if use_log_x:
        ax.set_xscale("log")
        custom_ticks = {x_min, x_max}
        if x_col == "n_samples" and x_min < 1000 < x_max:
            custom_ticks.add(1000)

        if ("sigma" in x_col or 'sim_bias_b' in x_col) and x_min < 1.0 < x_max:
            custom_ticks.add(1.0)

        ax.set_xticks(sorted(list(custom_ticks)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Y-Axis Log Logic
    use_log_y = False
    if log_y is not None:
        use_log_y = log_y
    else:
        y_min, y_max = df_agg[y_label].min(), df_agg[y_label].max()
        if "error" in y_label or "nll" in y_label or "regret" in y_label:
            if y_min > 1e-9 and (y_max / y_min > 20):
                use_log_y = True

    if use_log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)

    # Grid
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.3)

    # --- 6. Layout & Legend ---
    ax.set_xlabel(LABEL_MAP.get(x_col, x_label), fontweight="bold")
    ax.set_ylabel(LABEL_MAP.get(y_label, y_label), fontweight="bold")
    # ax.set_title(title, pad=15, fontweight='bold')

    if y_top_margin:
        curr_bottom, curr_top = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(curr_bottom, curr_top * (10**y_top_margin))
        else:
            ax.set_ylim(curr_bottom, curr_top * y_top_margin)

    ax.legend(
        loc=legend_loc, frameon=True, edgecolor="black", framealpha=0.95, fancybox=False
    )

    clean_title = (
        title.lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("$", "")
        .replace("\\", "")
    )
    RUN_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = FIG_DIR / f"{clean_title}_{RUN_id}.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved to {save_path}")


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Starting Experiments. ID: {RUN_ID}")

    # Base Configuration
    # 使用较小的 dim_z (例如 10-24) 以便更快收敛
    base_cfg = ExpConfig(
        n_samples=2000,
        dim_z=24,
        dim_context=24,  # Match dim_z for independent mapping
        est_noise_sigma=1.5,
        sim_noise_sigma=0.2,
        outlier_prob=0.1,
        utility_mode="structural",  # Better for Regret/Tau experiments
        noise_distribution="gaussian",
        sim_bias_a=1.0,
        sim_bias_b=2.0,  # Stronger slope for Monotone effect
    )

    # ------------------------------------------------------
    # Exp 1: Convergence vs Sample Size (N)
    # ------------------------------------------------------
    def exp1_1():
        df1 = run_experiment_grid(
            base_cfg=base_cfg,
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            # compare_axis_name='algo_type',
            # compare_values=['linear', 'mrc'],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),  # Baseline: Lin on Lin
                ("linear", "monotone"),  # Lin on Monotone (Should fail/bias)
                ("linear_oracle", "monotone"),
                ("linear_oracle", "linear"),
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",  # MRC should win here
            n_seeds=10,
            # sim_noise_type='cauchy'
        )
        plot_metric_scaling(
            df1,
            "n_samples",
            "N",
            "p0_error",
            "combo_label",
            "Linear Convergence Rate",
            y_top_margin=1.4,
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(df1, 'n_samples', 'N', 'gamma_error', 'combo_label', "Linear Convergence Rate")

    def exp1_2():
        df2 = run_experiment_grid(
            base_cfg=replace(
                replace(base_cfg, noise_distribution="uniform"), outlier_prob=0.0
            ),
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            # compare_axis_name='algo_type',
            # compare_values=['linear', 'mrc'],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("mrc", "linear"),  # MRC on Lin (Should be good)
                ("mrc", "monotone"),  # MRC on Monotone (Should succeed)
                ("mrc_oracle", "linear"),
                ("mrc_oracle", "monotone"),
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",  # MRC should win here
            n_seeds=30,
            # sim_noise_type='cauchy'
        )
        plot_metric_scaling(
            df2,
            "n_samples",
            "N",
            "p0_error",
            "combo_label",
            "MRC Convergence Rate",
            y_top_margin=1.4,
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(df2, 'n_samples', 'N', 'gamma_error', 'combo_label', "MRC Convergence Rate")

    # ------------------------------------------------------
    # Exp 2: Robustness to Utility Noise (tau)
    # ------------------------------------------------------
    def exp2_1():
        df1 = run_experiment_grid(
            base_cfg=base_cfg,
            x_axis_name="est_noise_sigma",
            x_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            compare_axis_name="algo_type",
            compare_values=[
                "linear",
                "linear_oracle",
            ],
            cross_product=False,  # Explicit list
            default_y_type="linear",
            n_seeds=10,
            # sim_noise_type='cauchy'
        )
        plot_metric_scaling(
            df1,
            "est_noise_sigma",
            r"$\sqrt{\bar{\tau}}$",
            "p0_error",
            "algo_type",
            "Linear Utility Noise Robustness",
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(df1, 'est_noise_sigma', r'$\tau$', 'gamma_error', 'algo_type', "Linear Utility Noise Robustness")

    def exp2_2():
        df2 = run_experiment_grid(
            base_cfg=replace(
                replace(base_cfg, sim_noise_sigma=2.0), noise_distribution="uniform"
            ),
            x_axis_name="est_noise_sigma",
            x_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            compare_axis_name="algo_type",
            compare_values=[
                "mrc",
                "mrc_oracle",
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",
            n_seeds=50,
        )
        plot_metric_scaling(
            df2,
            "est_noise_sigma",
            r"$\tau_s$",
            "p0_error",
            "algo_type",
            "MRC Utility Noise Robustness",
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(df2, 'est_noise_sigma', r'$\tau$', 'gamma_error', 'algo_type', "MRC Utility Noise Robustness")

    # ------------------------------------------------------
    # Exp 3: Assortment Regret
    # ------------------------------------------------------
    def exp3_1():
        df1 = run_experiment_grid(
            base_cfg=base_cfg,
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),  # Baseline: Lin on Lin
                ("linear", "monotone"),  # Lin on Monotone (Should fail/bias)
                ("linear_oracle", "linear"),
                ("linear_oracle", "monotone"),
            ],
            cross_product=False,  # Explicit list
            regret_need=True,
            default_y_type="monotone",
            n_seeds=10,
        )
        plot_metric_scaling(
            df1,
            "n_samples",
            "N",
            "regret",
            "combo_label",
            "Linear Revenue Regret",
            y_top_margin=1.4,
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )

    def exp3_2():
        df2 = run_experiment_grid(
            base_cfg=replace(
                replace(base_cfg, noise_distribution="uniform"), outlier_prob=0.0
            ),
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("mrc", "linear"),  # MRC on Lin (Should be good)
                ("mrc", "monotone"),  # MRC on Monotone (Should succeed)
                ("mrc_oracle", "linear"),
                ("mrc_oracle", "monotone"),
            ],
            cross_product=False,  # Explicit list
            regret_need=True,
            default_y_type="monotone",
            n_seeds=30,
        )
        plot_metric_scaling(
            df2,
            "n_samples",
            "N",
            "regret",
            "combo_label",
            "MRC Revenue Regret",
            y_top_margin=1.2,
            log_y=False,
            log_x=False,
        )

    # ------------------------------------------------------
    # Exp 4: Multi-Simulator Robustness
    # ------------------------------------------------------
    def exp4():
        df = run_experiment_grid(
            base_cfg=replace(base_cfg, est_noise_sigma=0.1),
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            compare_axis_name="multi_sim_method",
            compare_values=["logit_mean", "weighted_mean", "median", "mrc_oracle"],
            default_y_type="monotone",
            default_task_type="multi_sim",
            n_seeds=30,
        )
        plot_metric_scaling(
            df,
            "n_samples",
            "N",
            "p0_error",
            "multi_sim_method",
            "Multi-Sim Aggregation",
            log_y=False,
        )
        # plot_metric_scaling(df, 'n_samples', 'N', 'gamma_error', 'multi_sim_method', "Multi-Sim Aggregation")

    # ------------------------------------------------------
    # Exp 5: Simulator Noise Impact (Sigma)
    # ------------------------------------------------------
    def exp5_1():
        df1 = run_experiment_grid(
            base_cfg=base_cfg,
            x_axis_name="sim_noise_sigma",
            x_values=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),
                ("linear", "monotone"),
                ("linear_oracle", "linear"),
                ("linear_oracle", "monotone"),
            ],
            cross_product=False,
            default_y_type="monotone",
            n_seeds=10,
        )
        plot_metric_scaling(
            df1,
            x_col="sim_noise_sigma",
            x_label=r"$\sigma_{sim}$",
            y_label="p0_error",
            hue_col="combo_label",
            title="Linear Robustness to Simulator Noise",
            y_top_margin=1.2,
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(
        #     df1, x_col='sim_noise_sigma', x_label=r'$\sigma_{sim}$', y_label='gamma_error', hue_col='combo_label',
        #     title="Linear Robustness to Simulator Noise"
        # )

    def exp5_2():
        df2 = run_experiment_grid(
            base_cfg=replace(base_cfg, est_noise_sigma=1.0),
            x_axis_name="sim_noise_sigma",
            x_values=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("mrc", "linear"),  # MRC on Lin (Should be good)
                ("mrc", "monotone"),  # MRC on Monotone (Should succeed)
                ("mrc_oracle", "linear"),
                ("mrc_oracle", "monotone"),
            ],
            cross_product=False,
            default_y_type="monotone",  # Test on the harder case
            n_seeds=10,
        )
        plot_metric_scaling(
            df2,
            x_col="sim_noise_sigma",
            x_label=r"$\sigma_{sim}$",
            y_label="p0_error",
            hue_col="combo_label",
            title="MRC Robustness to Simulator Noise",
            y_top_margin=1.5,
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(
        #     df2, x_col='sim_noise_sigma', x_label=r'$\sigma_{sim}$', y_label='gamma_error', hue_col='combo_label',
        #     title="MRC Robustness to Simulator Noise"
        # )

    # ------------------------------------------------------
    # Exp 6: Assortment Size Impact (Market Density)
    # ------------------------------------------------------
    def exp6_1():
        df1 = run_experiment_grid(
            base_cfg=base_cfg,
            x_axis_name="max_assortment_size",
            x_values=[5, 10, 20, 50, 100],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),  # Baseline: Lin on Lin
                ("linear", "monotone"),  # Lin on Monotone (Should fail/bias)
                ("linear_oracle", "linear"),
                ("linear_oracle", "monotone"),
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",
            n_seeds=10,
        )
        plot_metric_scaling(
            df1,
            x_col="max_assortment_size",
            x_label=r"Max $|S|$",
            y_label="p0_error",
            hue_col="combo_label",
            title="Linear Impact of Assortment Size",
            y_top_margin=1.5,
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(
        #     df1, x_col='max_assortment_size', x_label=r'Max $|S|$', y_label='gamma_error', hue_col='combo_label',
        #     title="Linear Impact of Assortment Size"
        # )

    def exp6_2():
        df2 = run_experiment_grid(
            base_cfg=base_cfg,
            x_axis_name="max_assortment_size",
            x_values=[5, 10, 20, 50, 100],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("mrc", "linear"),  # MRC on Lin (Should be good)
                ("mrc", "monotone"),  # MRC on Monotone (Should succeed)
                ("mrc_oracle", "linear"),
                ("mrc_oracle", "monotone"),
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",
            n_seeds=10,
        )
        plot_metric_scaling(
            df2,
            x_col="max_assortment_size",
            x_label=r"Max $|S|$",
            y_label="p0_error",
            hue_col="combo_label",
            title="MRC Impact of Assortment Size",
            y_top_margin=1.2,
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(
        #     df2, x_col='max_assortment_size', x_label=r'Max $|S|$', y_label='gamma_error', hue_col='combo_label',
        #     title="MRC Impact of Assortment Size"
        # )

    # ------------------------------------------------------
    # Exp 7: Outliers' impact on solver
    # ------------------------------------------------------

    def exp7_1():
        df1 = run_experiment_grid(
            base_cfg=replace(base_cfg, outlier_prob=0),
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),  # Baseline: Lin on Lin
                ("mrc", "linear"),  # MRC on Lin (Should be good)
                ("linear_oracle", "linear"),
                ("mrc_oracle", "linear"),
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",
            n_seeds=10,
        )
        plot_metric_scaling(
            df1,
            "n_samples",
            "N",
            "p0_error",
            "combo_label",
            "Linear Outlier Impact",
            y_top_margin=1.5,
            legend_loc="upper left",
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(df1, 'n_samples', 'N', 'gamma_error', 'combo_label', "Linear Outlier Impact")

    def exp7_2():
        df2 = run_experiment_grid(
            base_cfg=replace(base_cfg, outlier_prob=0),
            x_axis_name="n_samples",
            x_values=[200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "monotone"),
                ("mrc", "monotone"),
                ("linear_oracle", "monotone"),
                ("mrc_oracle", "monotone"),
            ],
            cross_product=False,  # Explicit list
            default_y_type="monotone",
            n_seeds=10,
        )
        plot_metric_scaling(
            df2,
            "n_samples",
            "N",
            "p0_error",
            "combo_label",
            "Monotone Outlier Impact",
            y_top_margin=1.4,
            log_y=False,
            log_x=False,
        )
        # plot_metric_scaling(df2, 'n_samples', 'N', 'gamma_error', 'combo_label', "Monotone Outlier Impact")

    # ------------------------------------------------------
    # Exp 8: Feature Dimension Scalability (d)
    # ------------------------------------------------------
    def exp8():
        df_dim = run_experiment_grid(
            base_cfg=replace(base_cfg, n_samples=8000),
            x_axis_name="dim_z",
            x_values=[5, 10, 20, 50, 100, 200],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),
                ("mrc", "monotone"),
                ("linear_oracle", "linear"),
                ("mrc_oracle", "monotone"),
            ],
            cross_product=False,
            default_context_type="concat",
            n_seeds=10,
        )

        plot_metric_scaling(
            df_dim,
            x_col="dim_z",
            x_label=r"Dimension $d$",
            y_label="gamma_error",
            hue_col="combo_label",
            title="Impact of Feature Dimension",
            log_y=False,
            log_x=False,
        )
        
    def exp9_1():
        """
        Tests the robustness against the magnitude of the bias (Slope b*).
        Small b* = Weak signal (harder for Linear).
        Large b* = Amplified signal.
        """
        df_bias = run_experiment_grid(
            base_cfg=base_cfg, # Fixed moderate noise
            x_axis_name="sim_bias_b",
            # Covering: Weak Signal (0.1) -> Identity (1.0) -> Amplified (5.0)
            x_values=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("linear", "linear"),      # Should handle scaling perfectly
                ("linear", "monotone"),    # Should fail as b* changes curvature
                ("linear_oracle", "linear"),
                ("linear_oracle", "monotone"),       # Should be robust
            ],
            cross_product=False,
            default_y_type="monotone",
            n_seeds=10, # 20 seeds to ensure stable std deviation
        )

        plot_metric_scaling(
            df_bias,
            x_col="sim_bias_b",
            x_label=r"$b^*$",
            y_label="p0_error",
            hue_col="combo_label",
            title="Robustness to Bias Magnitude",
            y_top_margin=1.4,
            log_y=False,
            log_x=True, # Log scale X-axis makes it easier to see 0.1 vs 5.0
        )
    def exp9_2():
        df_bias = run_experiment_grid(
            base_cfg=base_cfg, # Fixed moderate noise
            x_axis_name="sim_bias_b",
            # Covering: Weak Signal (0.1) -> Identity (1.0) -> Amplified (5.0)
            x_values=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
            compare_axis_name=["algo_type", "y_type"],
            compare_values=[
                ("mrc", "linear"),      # Should handle scaling perfectly
                ("mrc", "monotone"),    # Should fail as b* changes curvature
                ("mrc_oracle", "linear"),
                ("mrc_oracle", "monotone"),       # Should be robust
            ],
            cross_product=False,
            default_y_type="monotone",
            n_seeds=50, 
        )

        plot_metric_scaling(
            df_bias,
            x_col="sim_bias_b",
            x_label=r"$b^*$",
            y_label="p0_error",
            hue_col="combo_label",
            title="Robustness to Bias Magnitude",
            y_top_margin=1.4,
            log_y=False,
            log_x=True, # Log scale X-axis makes it easier to see 0.1 vs 5.0
        )
    

    # ------------------------------------------------------
    # Run Selection
    # ------------------------------------------------------
    # Choose which experiments to run
    experiments = [
        exp1_1,
        exp1_2,
        exp2_1,
        exp2_2,
        exp3_1,
        exp3_2,
        exp4,
        exp5_1,
        exp5_2,
        exp9_1,
        exp9_2,
    ]

    for func in experiments:
        print(f"\n{'='*40}")
        print(f">>> Running {func.__name__}...")
        print(f"{'='*40}")
        func()

    print("All Experiments Finished.")
