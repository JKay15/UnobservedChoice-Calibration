import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import RealExpConfig
from src.datasets.real_preprocessing import ExpediaPreprocessor
from src.datasets.data_loader import ExpediaDataLoader
from src.algorithms.solver import CalibrationSolver
from src.utils.metrics import compute_p0_from_logits, compute_nll

# ==========================================
# [NEW] Helper Metrics Functions
# ==========================================
def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures the weighted average absolute difference between predicted probability 
    and empirical accuracy across bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Select samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            # Empirical accuracy in this bin
            acc_in_bin = np.mean(y_true[in_bin])
            # Average predicted prob in this bin
            conf_in_bin = np.mean(y_prob[in_bin])
            
            # Weighted absolute difference
            ece += np.abs(acc_in_bin - conf_in_bin) * (n_in_bin / total_samples)
            
    return ece

# # ==========================================
# # 1. MNL Model Definition
# # ==========================================

class LinearUtilityModel(torch.nn.Module):
    """Classic MNL: u = beta * x"""
    def __init__(self, input_dim):
        super().__init__()
        self.beta = torch.nn.Linear(input_dim, 1, bias=False)
        torch.nn.init.normal_(self.beta.weight, mean=0.0, std=0.01)
        
    def forward(self, x, mask):
        u = self.beta(x).squeeze(-1) 
        u = u.masked_fill(mask == 0, -1e9)
        return u
    
    def forward_flat(self, x):
        return self.beta(x).squeeze(-1)

class NeuralUtilityModel(torch.nn.Module):
    """Neural MNL: u = MLP(x)"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        # Init last layer near zero
        torch.nn.init.normal_(self.net[2].weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.net[2].bias)
        
    def forward(self, x, mask):
        u = self.net(x).squeeze(-1)
        u = u.masked_fill(mask == 0, -1e9)
        return u

    def forward_flat(self, x):
        return self.net(x).squeeze(-1)

# ==========================================
# 2. Training Helper (Restored Visualization)
# ==========================================
def train_mnl(cfg, data_loader, item_dim):
    print(f"[Estimator] Training MNL (Type: {cfg.utility_model_type})...")
    
    batch = data_loader.get_pytorch_data(filter_booking_only=True)
    device = torch.device(cfg.device)
    items = batch['items'].to(device)
    mask = batch['mask'].to(device)
    labels = batch['labels'].to(device)
    
    N_samples = items.shape[0]
    print(f"   Training samples: {N_samples}")
    
    if cfg.utility_model_type == 'linear':
        model = LinearUtilityModel(item_dim).to(device)
    elif cfg.utility_model_type == 'neural':
        model = NeuralUtilityModel(item_dim, hidden_dim=64).to(device)
    else:
        raise ValueError(f"Unknown utility model type: {cfg.utility_model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.mnl_lr)
    
    bs = cfg.mnl_batch_size
    n_batches = (N_samples + bs - 1) // bs
    
    model.train()
    for epoch in range(cfg.mnl_epochs):
        total_loss = 0
        indices = torch.randperm(N_samples)
        
        for i in range(n_batches):
            idx = indices[i*bs : (i+1)*bs]
            logits = model(items[idx], mask[idx])
            loss = torch.nn.functional.cross_entropy(logits, labels[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"   Epoch {epoch+1}/{cfg.mnl_epochs} | Loss: {avg_loss:.4f}")
            
    model.eval()
    model.cpu() 
    return model

# ==========================================
# 3. Main Pipeline (With Diagnostics)
# ==========================================
def run_pipeline(
    utility_model_type: str = 'linear',
    mnl_lr: float = None,
    mnl_epochs: int = None,
    return_preds: bool = False
):
    print(f"\n>>> Initializing Real Data Experiment [{utility_model_type.upper()}]...")
    cfg = RealExpConfig()
    
    # Config Override
    cfg.utility_model_type = utility_model_type
    if mnl_lr is not None: cfg.mnl_lr = mnl_lr
    if mnl_epochs is not None: cfg.mnl_epochs = mnl_epochs
    
    preprocessor = ExpediaPreprocessor(cfg)
    train_df, test_df = preprocessor.load_and_process(nrows=None)
    
    all_feat_cols = preprocessor.final_feat_cols
    cat_indices = preprocessor.cat_features_indices
    
    # --- Train Simulator ---
    print("\n>>> Step 1: Training Simulator on Historical CLICK Data...")
    train_loader = ExpediaDataLoader(cfg, train_df, all_feat_cols)
    X_train, y_train, _ = train_loader.get_catboost_data(target_col='booking_bool', only_clicks=True)
    
    sim_model = CatBoostClassifier(
        iterations=cfg.cat_iterations,
        depth=cfg.cat_depth,
        learning_rate=cfg.cat_learning_rate,
        loss_function='Logloss',
        verbose=100, 
        task_type="CPU",
        allow_writing_files=False,
        random_seed=cfg.seed
    )
    sim_model.fit(X_train, y_train, cat_features=cat_indices)
    print("   Simulator Trained.")

    # --- Calibration Phase ---
    print("\n>>> Step 2: Calibration on Current Data (Observed Sales)...")
    test_loader = ExpediaDataLoader(cfg, test_df, all_feat_cols)
    dim_item_feat = len(test_loader.item_feat_indices)
    
    # 1. Train Utility Model
    utility_model = train_mnl(cfg, test_loader, dim_item_feat)
    
    # 2. Construct Vectors
    print("   Constructing calibration vectors (Vectorized)...")
    booked_srch_ids = test_df[test_df['booking_bool'] == 1]['srch_id'].unique()
    current_sales_df = test_df[test_df['srch_id'].isin(booked_srch_ids)].copy()
    
    # Calculate s_hat
    item_col_names = [all_feat_cols[i] for i in test_loader.item_feat_indices]
    sales_items_x = torch.tensor(current_sales_df[item_col_names].values).float()
    
    with torch.no_grad():
        sales_u = utility_model.forward_flat(sales_items_x).numpy()
    
    current_sales_df['u_hat'] = sales_u
    print("   Aggregating s_hat...")
    s_series = current_sales_df.groupby('srch_id')['u_hat'].apply(lambda x: logsumexp(x.values))
    
    ctx_df = current_sales_df.groupby('srch_id')[list(cfg.context_cols)].first()
    
    # Simulator Preds
    current_sales_df['sim_p_book'] = sim_model.predict_proba(current_sales_df[all_feat_cols])[:, 1]
    sim_p_nobuy = 1.0 - current_sales_df.groupby('srch_id')['sim_p_book'].sum()
    sim_p_nobuy = np.clip(sim_p_nobuy, 0.001, 0.999)
    y_series = np.log(sim_p_nobuy / (1 - sim_p_nobuy))
    
    # Align
    common_index = s_series.index
    z_df = ctx_df.loc[common_index]
    y_data = y_series.loc[common_index]
    
    z_tensor = torch.tensor(z_df.values).float().to(cfg.device)
    s_tensor = torch.tensor(s_series.values).float().to(cfg.device)
    y_tensor = torch.tensor(y_data.values).float().to(cfg.device)
    
    # 3. Run Solvers
    print("   Solving Calibration...")
    solver = CalibrationSolver(cfg)
    gamma_lin = solver.solve_linear(z_tensor, s_tensor, y_tensor)
    gamma_mrc = solver.solve_mrc(z_tensor, s_tensor, y_tensor)
    
    print(f"   Gamma Norms | Linear: {gamma_lin.norm().item():.2f}, MRC: {gamma_mrc.norm().item():.2f}")
    # [Diagnostic] Print first few weights to see if they differ
    print(f"   Gamma Head (Lin): {gamma_lin[:3].cpu().numpy()}")
    print(f"   Gamma Head (MRC): {gamma_mrc[:3].cpu().numpy()}")

    # --- Final Evaluation ---
    print("\n>>> Step 3: Evaluation on FULL Test Set...")
    
    y_true = test_df.groupby('srch_id')['booking_bool'].sum().apply(lambda x: 1 if x==0 else 0).values
    
    # Update U and S with the trained model
    test_items_x = torch.tensor(test_df[item_col_names].values).float()
    with torch.no_grad():
        test_u = utility_model.forward_flat(test_items_x).numpy()
    test_df['u_hat'] = test_u
    
    print("   Calculating s_hat for all test sessions...")
    test_s_hat = test_df.groupby('srch_id')['u_hat'].apply(lambda x: logsumexp(x.values))
    
    test_ctx = test_df.groupby('srch_id')[list(cfg.context_cols)].first()
    eval_z = torch.tensor(test_ctx.values).float().to(cfg.device)
    eval_s = torch.tensor(test_s_hat.values).float().to(cfg.device)
    
    
    # 1. Get Probabilities (For plotting) - Move to CPU numpy for visualization
    p_lin = compute_p0_from_logits(eval_z, eval_s, gamma_lin).cpu().numpy()
    p_mrc = compute_p0_from_logits(eval_z, eval_s, gamma_mrc).cpu().numpy()
    
    # [Diagnostic] Check Predictions
    print(f"   [Diag] P_Lin: Mean={p_lin.mean():.4f}, Min={p_lin.min():.6f}, Max={p_lin.max():.6f}")
    print(f"   [Diag] P_MRC: Mean={p_mrc.mean():.4f}, Min={p_mrc.min():.6f}, Max={p_mrc.max():.6f}")
    
    # Simulator Baseline
    test_df['sim_p'] = sim_model.predict_proba(test_df[all_feat_cols])[:, 1]
    sim_p_nobuy = 1.0 - test_df.groupby('srch_id')['sim_p'].sum()
    sim_p_nobuy = np.clip(sim_p_nobuy, 0.001, 0.999).values
    
    nll_sim = compute_nll(sim_p_nobuy, y_true)
    nll_lin = compute_nll(p_lin, y_true)
    nll_mrc = compute_nll(p_mrc, y_true)
    
    # [NEW] ECE
    ece_sim = compute_ece(y_true, sim_p_nobuy)
    ece_lin = compute_ece(y_true, p_lin)
    ece_mrc = compute_ece(y_true, p_mrc)
    
    # [NEW] AUC (Note: y_true=1 means No-Purchase)
    # We want higher prob for y_true=1.
    auc_sim = roc_auc_score(y_true, sim_p_nobuy)
    auc_lin = roc_auc_score(y_true, p_lin)
    auc_mrc = roc_auc_score(y_true, p_mrc)
    
    print("-" * 65)
    print(f" RESULTS: [{utility_model_type.upper()}] Utility Model")
    print("-" * 65)
    print(f" {'Method':<12} | {'NLL':<8} | {'ECE':<8} | {'AUC':<8}")
    print("-" * 65)
    print(f" {'Simulator':<12} | {nll_sim:.5f}   | {ece_sim:.5f}   | {auc_sim:.5f}")
    print(f" {'Linear':<12} | {nll_lin:.5f}   | {ece_lin:.5f}   | {auc_lin:.5f}")
    print(f" {'MRC':<12}    | {nll_mrc:.5f}   | {ece_mrc:.5f}   | {auc_mrc:.5f}")
    print("-" * 40)
    
    result = {
        'model': utility_model_type, 
        'nll_mrc': nll_mrc, 
        'nll_lin': nll_lin,
        'nll_sim': nll_sim,
        'ece_sim': ece_sim,
        'ece_lin': ece_lin,
        'ece_mrc': ece_mrc,
        'auc_sim': auc_sim,
        'auc_lin': auc_lin,
        'auc_mrc': auc_mrc,
    }
    # [NEW] Return predictions for plotting
    if return_preds:
        result.update({
            'y_true': y_true,
            'p_sim': sim_p_nobuy,
            'p_lin': p_lin,
            'p_mrc': p_mrc
        })
    
    return result
def plot_real_data_results(res_lin, res_nn):
    """
    Generates Separate Bar Charts (NLL, ECE, AUC) with Hatching + Calibration Curves.
    """
    # 1. Setup Style
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"], 
        "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 22, 
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
        "lines.linewidth": 3.0, "lines.markersize": 10,
        "axes.grid": True, "grid.alpha": 0.3
    })
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Prepare Data
    methods = ['Simulator', 'Linear\n(Lin)', 'Linear\n(NN)', 'MRC\n(Lin)', 'MRC\n(NN)']
    
    # 颜色 (与 Synthetic 保持一致的 Palette)
    # Sim=Gray, Linear=Blue, MRC=Red/Orange
    c_sim = '#7f7f7f'
    c_lin = '#0072B2'
    c_mrc = '#D55E00'
    
    # Helper for lightness
    import matplotlib.colors as mc
    def lighten(c, amount=0.5):
        c = mc.to_rgb(c)
        return mc.to_hex((c[0] + (1-c[0])*amount, c[1] + (1-c[1])*amount, c[2] + (1-c[2])*amount))

    colors = [
        c_sim, 
        lighten(c_lin, 0.4), c_lin, 
        lighten(c_mrc, 0.4), c_mrc
    ]
    
    # [HATCHING] 纹理映射: Sim(/), Linear(x), MRC(.)
    hatches = ['//', 'xx', 'xx', '..', '..']

    # Metrics Data
    metrics = {
        'NLL': ([res_lin['nll_sim'], res_lin['nll_lin'], res_nn['nll_lin'], res_lin['nll_mrc'], res_nn['nll_mrc']], True),
        'ECE': ([res_lin['ece_sim'], res_lin['ece_lin'], res_nn['ece_lin'], res_lin['ece_mrc'], res_nn['ece_mrc']], True),
        'AUC': ([res_lin['auc_sim'], res_lin['auc_lin'], res_nn['auc_lin'], res_lin['auc_mrc'], res_nn['auc_mrc']], False)
    }
    
    # Label Map
    metric_labels = {
        'NLL': 'Negative Log-Likelihood',
        'ECE': 'Expected Calibration Error',
        'AUC': 'AUC Score'
    }

    # 3. Bar Chart Generator
    def save_bar_chart(metric_key):
        data, is_lower_better = metrics[metric_key]
        ylabel = metric_labels[metric_key]
        
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        
        # Draw Bars with Hatching
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, data, color=colors, edgecolor='black', linewidth=1.5, width=0.65)
        
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Decorations
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=15)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f"{metric_key} Comparison", pad=15, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Scaling
        if is_lower_better:
            ax.set_ylim(0, max(data) * 1.2)
            txt_off = max(data) * 0.02
        else:
            y_min = min(data) * 0.95
            ax.set_ylim(y_min, 1.005)
            txt_off = (1 - y_min) * 0.02

        # Text Labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + txt_off,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_dir / f"real_{metric_key.lower()}_{timestamp}.png")
        plt.close()
        print(f"   Saved {metric_key} Chart.")

    # Generate Bars
    save_bar_chart('NLL')
    save_bar_chart('ECE')
    save_bar_chart('AUC')

    # 4. Calibration Curve
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect", linewidth=2.0)
    
    # Config for lines
    line_cfgs = [
        (res_nn, 'p_sim', 'Simulator', c_sim, ':', ''),
        (res_lin, 'p_lin', 'Linear (Lin)', lighten(c_lin, 0.4), '--', 's'),
        (res_lin, 'p_mrc', 'MRC (Lin)', lighten(c_mrc, 0.4), '-', '^'),
        (res_nn, 'p_mrc', 'MRC (Neural)', c_mrc, '-', 'D')
    ]
    
    for (res, key, name, c, ls, m) in line_cfgs:
        y_true, y_prob = res['y_true'], res[key]
        fp, mp = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(mp, fp, label=name, color=c, linestyle=ls, marker=m, 
                 linewidth=2.5, markersize=8, alpha=0.85)

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual No-Purchase Rate")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left", frameon=True, edgecolor='black', fancybox=False)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.ylim(0, 1.4)
    plt.tight_layout()
    plt.savefig(save_dir / f"real_calibration_{timestamp}.png")
    plt.close()
    print(f"   Saved Calibration Curve.")


if __name__ == "__main__":
    res_lin = run_pipeline(utility_model_type='linear',return_preds=True)
    res_nn = run_pipeline(utility_model_type='neural', mnl_lr=0.001, mnl_epochs=50,return_preds=True)
    
    print("\n" + "="*50)
    print(" FINAL COMPARISON: Linear vs Neural Utility ")
    print("="*50)
    print(f" Linear MNL -> MRC NLL: {res_lin['nll_mrc']:.5f}")
    print(f" Neural MNL -> MRC NLL: {res_nn['nll_mrc']:.5f}")
    
    plot_real_data_results(res_lin, res_nn)
    
    improvement = res_lin['nll_mrc'] - res_nn['nll_mrc']
    if improvement > 0:
        print(f" --> Neural Utility Improved NLL by {improvement:.5f}")
    else:
        print(f" --> Neural Utility did NOT improve (-{-improvement:.5f})")