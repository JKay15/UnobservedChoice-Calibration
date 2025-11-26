import sys
import os
import time
import datetime
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import ExpConfig
from src.engine.factory import EngineFactory
from src.algorithms.solver import CalibrationSolver
from src.utils.metrics import compute_parameter_error, compute_rmse
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

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. Unified Atomic Trial Runner
# ==========================================
def run_single_trial(cfg: ExpConfig, 
                     task_type: str = 'standard',      # 'standard' or 'multi_sim'
                     algo_type: str = 'mrc',           # 'linear' or 'mrc'
                     multi_sim_method: str = 'median', # 'mean' or 'median'
                     y_type: str = 'linear',
                     z_type: str = 'stats',
                     u_type: str = 'linear',
                     z_model_path: str = None) -> Dict[str, float]:
    """
    Executes ONE single experiment run.
    Handles both Standard Calibration (with Regret) and Multi-Simulator Robustness.
    """
    # 1. Build Engine & Generate Data
    engine = EngineFactory.build_synthetic_engine(
        cfg, z_type=z_type, u_type=u_type, y_type=y_type, z_model_path=z_model_path
    )
    data = engine.generate()
    inputs = data['inputs']
    truth = data['truth']
    solver = CalibrationSolver(cfg)
    
    # ==========================================
    # Branch A: Multi-Simulator Robustness Task
    # ==========================================
    if task_type == 'multi_sim':
        eta_true = truth['eta']
        
        # Sim 1: Good (Monotone)
        y1 = eta_true + torch.randn_like(eta_true) * 0.5
        
        # Sim 2: Good (Monotone)
        y2 = eta_true + torch.randn_like(eta_true) * 0.5
        
        # Sim 3: Adversarial & Extreme Scale (The "Attacker")
        # [CRITICAL] We multiply by -20.0. 
        # This makes the absolute values HUGE and REVERSED.
        # "Logit Mean" will be dominated by this one (-20 + 1 + 1 => Negative).
        # "Median" will pick one of the good ones (+1).
        y3 = -20.0 * eta_true + torch.randn_like(eta_true) * 2.0
        
        Y_multi = torch.stack([y1, y2, y3], dim=1)
        
        start_time = time.time()
        gamma_hat = solver.solve_multi_mrc(inputs['z'], inputs['s_hat'], Y_multi, method=multi_sim_method)
        duration = time.time() - start_time
        
        param_error = compute_parameter_error(gamma_hat, truth['gamma'])
        
        # Return specific metrics for multi-sim
        return {
            'param_error': param_error,
            'time': duration,
            'sqrt_d': np.sqrt(cfg.dim_z)
        }

    # ==========================================
    # Branch B: Standard Calibration (+ Regret)
    # ==========================================
    else:
        start_time = time.time()
        if algo_type == 'linear':
            gamma_hat = solver.solve_linear(inputs['z'], inputs['s_hat'], inputs['y'])
        elif algo_type == 'mrc':
            gamma_hat = solver.solve_mrc(inputs['z'], inputs['s_hat'], inputs['y'])
        else:
            raise ValueError(f"Unknown algo_type: {algo_type}")
        duration = time.time() - start_time
        
        # 1. Basic Metrics
        param_error = compute_parameter_error(gamma_hat, truth['gamma'])
        
        u0 = inputs['z'] @ gamma_hat
        eta = u0 - inputs['s_hat']
        p0_pred = torch.sigmoid(eta)
        rmse = compute_rmse(p0_pred, truth['p0'])
        
        # 2. Downstream Assortment Regret
        # Simulate a small "Test Set" of decisions
        n_test_decisions = 20
        n_items_pool = 15
        regret_list = []
        
        # Extract True Beta from engine
        beta_true = engine.u_mapper.beta 
        
        for _ in range(n_test_decisions):
            z_test = torch.randn(cfg.dim_z, device=cfg.device)
            items_feat = torch.randn(n_items_pool, cfg.dim_item_feat, device=cfg.device)
            prices = torch.rand(n_items_pool, device=cfg.device) * 90 + 10 
            
            u_items_true = items_feat @ beta_true
            
            # Solve Oracle
            mask_opt, r_opt = solve_optimal_assortment(truth['gamma'], z_test, prices, u_items_true)
            # Solve Plug-in
            mask_hat, _ = solve_optimal_assortment(gamma_hat, z_test, prices, u_items_true)
            # Evaluate Realized Revenue
            r_hat_realized = calculate_revenue(mask_hat, truth['gamma'], z_test, prices, u_items_true)
            
            if r_opt > 1e-6:
                regret_list.append((r_opt - r_hat_realized) / r_opt)
            else:
                regret_list.append(0.0)
                
        avg_regret = np.mean(regret_list) * 100 # Percentage

        return {
            'param_error': param_error,
            'rmse': rmse,
            'regret': avg_regret,
            'time': duration,
            'sqrt_d': np.sqrt(cfg.dim_z)
        }

# ==========================================
# 2. Universal Experiment Runner (Corrected)
# ==========================================
def run_experiment_grid(
    base_cfg: ExpConfig,
    x_axis_name: str,
    x_values: List[Any],
    compare_axis_name: Optional[str] = None,
    compare_values: Optional[List[Any]] = None,
    n_seeds: int = 5,
    # Defaults (Expanded to include task_type and multi_sim_method)
    default_task_type: str = 'standard',       
    default_algo_type: str = 'mrc',
    default_multi_sim_method: str = 'median',  
    default_y_type: str = 'monotone',
    default_z_type: str = 'stats',
    default_u_type: str = 'linear'
) -> pd.DataFrame:
    """
    Generic grid runner. Fully supports Multi-Sim and Standard tasks.
    """
    print(f"\n=== Running Grid Experiment: X={x_axis_name} | Hue={compare_axis_name} ===")
    
    records = []
    
    if compare_axis_name is None:
        compare_iters = [(None, None)]
    else:
        compare_iters = [(compare_axis_name, val) for val in compare_values]
    
    total_iters = len(x_values) * len(compare_iters) * n_seeds
    pbar = tqdm(total=total_iters, desc="Progress")
    
    for x_val in x_values:
        for comp_name, comp_val in compare_iters:
            for seed in range(n_seeds):
                # 1. Config Setup
                current_cfg_args = {
                    'seed': seed + 1000,
                    x_axis_name: x_val if hasattr(base_cfg, x_axis_name) else getattr(base_cfg, x_axis_name)
                }
                
                # 2. Dynamic Parameter Resolution
                params = {
                    'task_type': default_task_type,
                    'algo_type': default_algo_type,
                    'multi_sim_method': default_multi_sim_method,
                    'y_type': default_y_type,
                    'z_type': default_z_type,
                    'u_type': default_u_type
                }
                
                def apply_override(name, val):
                    if name in params:
                        params[name] = val
                
                apply_override(x_axis_name, x_val)
                if comp_name:
                    apply_override(comp_name, comp_val)
                
                # 3. Create Config
                cfg_args = current_cfg_args.copy()
                for k in params.keys():
                    cfg_args.pop(k, None) 
                    
                if x_axis_name not in base_cfg.__dict__:
                    cfg_args.pop(x_axis_name, None)

                cfg = ExpConfig(**{**base_cfg.__dict__, **cfg_args})
                set_seed(cfg.seed)
                
                # 4. Run Trial (Now passing all necessary params)
                metrics = run_single_trial(
                    cfg, 
                    task_type=params['task_type'],
                    algo_type=params['algo_type'],
                    multi_sim_method=params['multi_sim_method'],
                    y_type=params['y_type'],
                    z_type=params['z_type'],
                    u_type=params['u_type']
                )
                
                # 5. Record
                record = {
                    x_axis_name: x_val,
                    'seed': seed,
                    **metrics
                }
                record['n_samples'] = cfg.n_samples
                
                if comp_name:
                    record[comp_name] = comp_val
                
                records.append(record)
                pbar.update(1)
                
    pbar.close()
    df = pd.DataFrame(records)
    
    save_name = f"grid_{x_axis_name}"
    if compare_axis_name: save_name += f"_vs_{compare_axis_name}"
    df.to_csv(LOG_DIR / f"{save_name}_{RUN_ID}.csv", index=False)
    
    return df

# ==========================================
# 3. Universal Plotter (Updated)
# ==========================================
def plot_metric_scaling(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = "Scaling Analysis",
    ylabel: str = "Error",
    add_theory_line: bool = False,
    theory_anchor_group: str = 'mrc',
    theory_scale_multiplier: float = 1.0 
):
    plt.figure(figsize=(8, 6))
    
    # 1. Empirical Data Plot
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=hue_col,
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette='viridis' if hue_col else None
    )
    
    # 2. Theoretical Line Logic
    if add_theory_line:
        min_x = df[x_col].min()
        max_x = df[x_col].max()
        
        if hue_col and theory_anchor_group in df[hue_col].unique():
            base_val = df[(df[x_col] == min_x) & (df[hue_col] == theory_anchor_group)][y_col].mean()
        else:
            if hue_col:
                first_group = df[hue_col].unique()[0]
                base_val = df[(df[x_col] == min_x) & (df[hue_col] == first_group)][y_col].mean()
            else:
                base_val = df[df[x_col] == min_x][y_col].mean()
        
        t_x = np.linspace(min_x, max_x, 100)
        t_y = None
        label_str = ""

        if 'dim' in x_col:
            # Case A: Error ~ sqrt(d)
            # Formula: y = C * sqrt(x)
            base_sqrt = np.sqrt(min_x)
            scale = (base_val / (base_sqrt + 1e-9)) * theory_scale_multiplier
            t_y = scale * np.sqrt(t_x)
            label_str = r'Theory $\propto \sqrt{d}$'
            
        elif 'n_samples' in x_col:
            # Case B: Error ~ 1/sqrt(n)
            # Formula: y = C * (1 / sqrt(x))
            # base_val = C * (1 / sqrt(min_x))  =>  C = base_val * sqrt(min_x)
            scale_c = (base_val * np.sqrt(min_x)) * theory_scale_multiplier
            t_y = scale_c * (1.0 / np.sqrt(t_x))
            label_str = r'Theory $\propto 1/\sqrt{n}$'

        if t_y is not None:
            if theory_scale_multiplier != 1.0:
                label_str += f' (x{theory_scale_multiplier})'
            plt.plot(t_x, t_y, 'r--', label=label_str, linewidth=2, alpha=0.6)

    # 3. Title Formatting
    if x_col == 'n_samples':
        final_title = title
    else:
        n_val = df.iloc[0].get('n_samples', 'Unknown')
        final_title = f"{title} (N={n_val})"
        
    plt.title(final_title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    
    clean_title = title.lower().replace(" ", "_").replace(":", "")
    save_path = FIG_DIR / f"{clean_title}_{RUN_ID}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"Starting Experiments. ID: {RUN_ID}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}")

    base_cfg = ExpConfig(
        n_samples=2000,
        dim_z=20,
        est_noise_sigma=0.1,
        sim_bias_a=1.0,
        sim_bias_b=15
    )
    regret_cfg = ExpConfig(
        n_samples=2000,       
        
        # Since we normalized u0, we need a larger bias_b to create distortion
        sim_bias_b=15.0,       
        sim_bias_a=0.0,
        est_noise_sigma=0.1,
        
        # Increase item feature dim so u_i has more variance relative to u_0
        dim_item_feat=10,      
        
        # Increase assortment size so the combinatorial space is larger
        min_assortment_size=10,
        max_assortment_size=30,
        n_items_pool=2000
    )

    # # ------------------------------------------------------
    # # Exp 1: Convergence vs Sample Size (n)
    # # ------------------------------------------------------
    df_n = run_experiment_grid(
        base_cfg=base_cfg,
        x_axis_name='n_samples',
        x_values=[50, 100, 200, 500, 1000, 2000,4000], 
        compare_axis_name='algo_type',
        compare_values=['linear', 'mrc'],
        default_y_type='monotone'
    )
    plot_metric_scaling(
        df_n, x_col='n_samples', y_col='param_error', hue_col='algo_type',
        title="Exp 1: Convergence vs Sample Size",
        ylabel="L2 Error",
        add_theory_line=True,
        theory_scale_multiplier=10.0 
    )

    # ------------------------------------------------------
    # Exp 2: Scaling vs Dimension (d)
    # ------------------------------------------------------
    df_d = run_experiment_grid(
        base_cfg=base_cfg,
        x_axis_name='dim_z',
        x_values=[5, 10, 20, 40, 80, 160],
        compare_axis_name='algo_type',
        compare_values=['linear', 'mrc'],
        default_y_type='monotone'
    )
    # [FIX] Scale multiplier = 2.0
    plot_metric_scaling(
        df_d, x_col='dim_z', y_col='param_error', hue_col='algo_type',
        title="Exp 2: Scaling vs Dimension (d)",
        ylabel="L2 Error",
        add_theory_line=True,
        theory_scale_multiplier=2.0 
    )

    # ------------------------------------------------------
    # Exp 3: Robustness to Utility Noise (tau)
    # ------------------------------------------------------
    df_tau = run_experiment_grid(
        base_cfg=base_cfg,
        x_axis_name='est_noise_sigma', 
        x_values=[0.0, 0.1, 0.2, 0.4, 0.8],
        compare_axis_name='algo_type',
        compare_values=['linear', 'mrc'],
        default_y_type='monotone'
    )
    plot_metric_scaling(
        df_tau, x_col='est_noise_sigma', y_col='param_error', hue_col='algo_type',
        title="Exp 3: Robustness to Utility Noise (tau)",
        ylabel="L2 Error"
    )

    # ------------------------------------------------------
    # Exp 4: Bias Type Impact (Linear vs Monotone)
    # ------------------------------------------------------
    
    df_bias = run_experiment_grid(
        base_cfg=regret_cfg,
        x_axis_name='dim_z', 
        x_values=[10, 20, 40, 80, 160],
        compare_axis_name='y_type', 
        compare_values=['linear', 'monotone'],
        default_algo_type='mrc',
        n_seeds=20
    )
    plot_metric_scaling(
        df_bias, x_col='dim_z', y_col='param_error', hue_col='y_type',
        title="Exp 4: MRC Robustness across Simulator Types",
        ylabel="L2 Error",
        add_theory_line=True,
        theory_scale_multiplier=2.0
    )

    # ------------------------------------------------------
    # Exp 5: Assortment Optimization Regret
    # ------------------------------------------------------
    # Note: This uses the SAME data frame as Exp 2 or Exp 4 conceptually, 
    # but we re-run it to be clean (or we could reuse df_d if we want to save time).
    # Here we run explicitly to ensure 'regret' metric is focused.
    # Exp 5: Assortment Regret
    # [FIX] Use specific config to increase difficulty
    
    df_regret = run_experiment_grid(
        base_cfg=regret_cfg, # Use the harder config
        x_axis_name='dim_z',
        x_values=[10, 20, 40, 80, 160,180,200,240,300],
        compare_axis_name='algo_type',
        compare_values=['linear', 'mrc'],
        default_y_type='monotone'
    )
    plot_metric_scaling(
        df_regret, x_col='dim_z', y_col='regret', hue_col='algo_type',
        title="Exp 5: Assortment Regret Analysis",
        ylabel="Relative Revenue Regret (%)"
    )

    # ------------------------------------------------------
    # Exp 6: Multi-Simulator Robustness (Mean vs Median)
    # ------------------------------------------------------
    df_multi = run_experiment_grid(
        base_cfg=base_cfg,
        x_axis_name='dim_z', 
        x_values=[10, 20, 40, 80,160],
        compare_axis_name='multi_sim_method', 
        compare_values=['logit_mean', 'median'], # [FIX] Compare Naive vs Robust
        default_task_type='multi_sim'
    )
    plot_metric_scaling(
        df_multi, x_col='dim_z', y_col='param_error', hue_col='multi_sim_method',
        title="Exp 6: Multi-Simulator Robustness",
        ylabel="L2 Error",
        add_theory_line=True,
        theory_scale_multiplier=0.3
    )
    # ------------------------------------------------------
    # Exp 7: Simulator Noise Impact (Testing Flip Probability)
    # ------------------------------------------------------
    df_noise = run_experiment_grid(
        base_cfg=base_cfg,
        x_axis_name='sim_noise_sigma',
        x_values=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0], # 噪声从微小到巨大
        compare_axis_name='algo_type',
        compare_values=['linear', 'mrc'],
        default_y_type='monotone'
    )
    
    plot_metric_scaling(
        df_noise, x_col='sim_noise_sigma', y_col='param_error', hue_col='algo_type',
        title="Exp 7: Robustness to Simulator Noise",
        ylabel="L2 Error",
        add_theory_line=False
    )

    # ------------------------------------------------------
    # Exp 8: Assortment Size Impact (Market Density)
    # ------------------------------------------------------
    
    df_size = run_experiment_grid(
        base_cfg=base_cfg,
        x_axis_name='max_assortment_size',
        x_values=[5, 10, 20, 50, 100], 
        compare_axis_name='algo_type',
        compare_values=['linear', 'mrc'],
        default_y_type='monotone'
    )
    
    plot_metric_scaling(
        df_size, x_col='max_assortment_size', y_col='param_error', hue_col='algo_type',
        title="Exp 8: Impact of Assortment Size",
        ylabel="L2 Error",
        add_theory_line=False
    )

    print("All Experiments Finished.")