import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pathlib import Path
from sklearn.calibration import calibration_curve

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import RealExpConfig
from src.datasets.real_preprocessing import ExpediaPreprocessor
from src.datasets.data_loader import ExpediaDataLoader
from src.algorithms.solver import CalibrationSolver

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
            
        # [RESTORED] Print every 5 epochs for better visibility
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
    
    def get_p0(gamma):
        u0 = eval_z @ gamma
        eta = u0 - eval_s
        return torch.sigmoid(eta).cpu().numpy()
    
    p_lin = get_p0(gamma_lin)
    p_mrc = get_p0(gamma_mrc)
    
    # [Diagnostic] Check Predictions
    print(f"   [Diag] P_Lin: Mean={p_lin.mean():.4f}, Min={p_lin.min():.6f}, Max={p_lin.max():.6f}")
    print(f"   [Diag] P_MRC: Mean={p_mrc.mean():.4f}, Min={p_mrc.min():.6f}, Max={p_mrc.max():.6f}")
    
    # Simulator Baseline
    test_df['sim_p'] = sim_model.predict_proba(test_df[all_feat_cols])[:, 1]
    sim_p_nobuy = 1.0 - test_df.groupby('srch_id')['sim_p'].sum()
    sim_p_nobuy = np.clip(sim_p_nobuy, 0.001, 0.999).values
    
    nll_sim = log_loss(y_true, sim_p_nobuy)
    nll_lin = log_loss(y_true, p_lin)
    nll_mrc = log_loss(y_true, p_mrc)
    
    print("-" * 40)
    print(f" RESULTS: [{utility_model_type.upper()}] Utility Model")
    print("-" * 40)
    print(f" Simulator NLL: {nll_sim:.5f}")
    print(f" Linear Calib:  {nll_lin:.5f}")
    print(f" MRC Calib:     {nll_mrc:.5f}")
    print("-" * 40)
    
    result = {
        'model': utility_model_type, 
        'nll_mrc': nll_mrc, 
        'nll_lin': nll_lin,
        'nll_sim': nll_sim
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

# ==========================================
# 4. Visualization Functions
# ==========================================
def plot_real_data_results(res_lin, res_nn):
    """
    Generates NLL Comparison Bar Chart and Calibration Curves.
    Args:
        res_lin: Result dictionary from Linear Utility run (must contain preds)
        res_nn: Result dictionary from Neural Utility run (must contain preds)
    """
    # Setup
    sns.set_theme(style="whitegrid")
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n>>> Generating Visualizations...")

    # -------------------------------------------------------
    # Plot 1: NLL Comparison Bar Chart
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Prepare Data
    # We compare:
    # 1. Simulator (Uncalibrated) - take from linear result (same for both)
    # 2. Linear Calibration (Linear Utility) - The traditional baseline
    # 3. MRC (Linear Utility) - Algo improvement
    # 4. MRC (Neural Utility) - Model improvement (Best)
    
    methods = [
        'Simulator\n(Baseline)', 
        'Linear Calib\n(Linear Util)', 
        'MRC\n(Linear Util)', 
        'MRC\n(Neural Util)'
    ]
    
    nlls = [
        res_lin['nll_sim'], 
        res_lin['nll_lin'], 
        res_lin['nll_mrc'], 
        res_nn['nll_mrc']
    ]
    
    # Colors: Gray for baseline, Blue for Linear, Green for MRC-Lin, Red for MRC-Neural
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
    
    ax = sns.barplot(x=methods, y=nlls, hue=methods,palette=colors,legend=False)
    
    # Decoration
    plt.title('Negative Log Likelihood (NLL) on Real Data', fontsize=16, fontweight='bold')
    plt.ylabel('NLL (Lower is Better)', fontsize=14)
    plt.ylim(0, max(nlls) * 1.15) # Leave space for text
    
    # Add value labels on top of bars
    for i, v in enumerate(nlls):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    plt.tight_layout()
    save_path_bar = save_dir / f"real_nll_comparison_{timestamp}.png"
    plt.savefig(save_path_bar, dpi=300)
    print(f"   Saved NLL Bar Chart to {save_path_bar}")
    plt.close()

    # -------------------------------------------------------
    # Plot 2: Calibration Curve (Reliability Diagram)
    # -------------------------------------------------------
    plt.figure(figsize=(8, 8))
    
    # A. Perfect Calibration (Diagonal)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", linewidth=2)
    
    # Helper to plot curve
    def add_curve(y_true, y_prob, label, color, fmt='-o'):
        # calibration_curve bins the data and calculates (mean_pred, fraction_true)
        frac_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(mean_predicted_value, frac_of_positives, fmt, 
                 color=color, label=label, linewidth=2, markersize=6)

    # B. Simulator (Raw) - Usually miscalibrated
    add_curve(res_nn['y_true'], res_nn['p_sim'], 
              "Simulator (Uncalibrated)", "#95a5a6", '--^')
    
    # C. Linear Calibration (Linear Util) - The baseline method
    add_curve(res_lin['y_true'], res_lin['p_lin'], 
              "Linear Calib (Lin Util)", "#3498db", '-.s')
    
    # D. MRC (Neural Util) - Our Best Model
    add_curve(res_nn['y_true'], res_nn['p_mrc'], 
              "MRC (Neural Util)", "#e74c3c", '-o')
    
    # Decoration
    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Fraction of Positives (Actual No-Purchase Rate)", fontsize=14)
    plt.title("Calibration Curve (Reliability Diagram)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.4)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    save_path_curve = save_dir / f"real_calibration_curve_{timestamp}.png"
    plt.savefig(save_path_curve, dpi=300)
    print(f"   Saved Calibration Curve to {save_path_curve}")
    plt.close()

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