import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import RealExpConfig
from src.datasets.real_preprocessing import ExpediaPreprocessor
from src.datasets.data_loader import ExpediaDataLoader
from src.algorithms.solver import CalibrationSolver

# ==========================================
# 1. MNL Model Definition
# ==========================================
class ConditionalMNL(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Beta: Weights for item features (X) -> Utility
        self.beta = torch.nn.Linear(input_dim, 1, bias=False)
        # Initialize close to zero for stability
        torch.nn.init.normal_(self.beta.weight, mean=0.0, std=0.01)
        
    def forward(self, x, mask):
        """
        x: (Batch, Max_Items, Feat_Dim)
        mask: (Batch, Max_Items) - 1.0 for valid, 0.0 for padding
        """
        # u = X * beta
        u = self.beta(x).squeeze(-1) # (B, L)
        
        # Masking: Set padded items to -inf so exp(u) becomes 0
        # We use a large negative number safe for float32
        u = u.masked_fill(mask == 0, -1e9)
        return u

# ==========================================
# 2. Training Helper
# ==========================================
def train_mnl(cfg, data_loader, item_dim):
    """
    Trains MNL on 'Booking' sessions to learn Internal Utilities (Beta).
    Using M3 Pro MPS acceleration.
    """
    print("[Estimator] preparing data for MNL (Booking Sessions only)...")
    
    # 1. Get Data (Full Batch Tensor)
    # This uses your optimized Numpy Slicing
    batch = data_loader.get_pytorch_data(filter_booking_only=True)
    
    # Move to MPS (GPU)
    device = torch.device(cfg.device)
    items = batch['items'].to(device)     # (N, L, D)
    mask = batch['mask'].to(device)       # (N, L)
    labels = batch['labels'].to(device)   # (N,)
    
    N_samples = items.shape[0]
    print(f"[Estimator] Training on {N_samples} sessions...")
    
    # 2. Model & Optimizer
    model = ConditionalMNL(item_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.mnl_lr)
    
    # 3. Mini-batch Loop
    bs = cfg.mnl_batch_size
    n_batches = (N_samples + bs - 1) // bs
    
    model.train()
    for epoch in range(cfg.mnl_epochs):
        total_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(N_samples)
        
        for i in range(n_batches):
            idx = indices[i*bs : (i+1)*bs]
            
            x_batch = items[idx]
            m_batch = mask[idx]
            y_batch = labels[idx]
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(x_batch, m_batch)
            
            # Loss (Cross Entropy selects the booked item)
            loss = torch.nn.functional.cross_entropy(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"   Epoch {epoch+1}/{cfg.mnl_epochs} | Loss: {avg_loss:.4f}")
            
    return model.beta.weight.detach() # Keep on Device for now

# ==========================================
# 3. Main Pipeline
# ==========================================
def run_pipeline():
    # --- A. Setup ---
    print(">>> Initializing Real Data Experiment...")
    cfg = RealExpConfig()
    preprocessor = ExpediaPreprocessor(cfg)
    
    # --- B. Load & Process ---
    # Set nrows=None for full run, or 200_000 for fast testing
    train_df, test_df = preprocessor.load_and_process(nrows=None)
    
    # Identify Feature Columns indices for PyTorch slicing
    # Note: DataLoader handles this mapping internally, we just need the column list
    all_feat_cols = preprocessor.final_feat_cols
    cat_indices = preprocessor.cat_features_indices
    
    # --- C. Train Simulator (CatBoost) ---
    print("\n>>> Step 1: Training Simulator on Historical CLICK Data...")
    # Bias Source: Training on Clicks (Window Shopping), Predicting Bookings
    
    train_loader = ExpediaDataLoader(cfg, train_df, all_feat_cols)
    X_train, y_train, _ = train_loader.get_catboost_data(target_col='booking_bool', only_clicks=True)
    
    print(f"   Simulator Training Samples: {len(X_train)}")
    
    # Initialize CatBoost (CPU optimized)
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

    # --- D. Calibration Phase ---
    print("\n>>> Step 2: Calibration on Current Data (Observed Sales)...")
    
    # 1. Prepare DataLoader for Test Set
    test_loader = ExpediaDataLoader(cfg, test_df, all_feat_cols)
    
    # 2. Estimate Beta (Inside Utility) using MNL
    # We need the dimension of Item Features (X)
    # data_loader.item_feat_indices gives us the count
    dim_item_feat = len(test_loader.item_feat_indices)
    
    beta_hat = train_mnl(cfg, test_loader, dim_item_feat)
    
    # 3. Construct Calibration Vectors (Z, s_hat, y_sim)
    # We need these for the BOOKING sessions only
    print("   Constructing calibration vectors...")
    
    batch_booking = test_loader.get_pytorch_data(filter_booking_only=True)
    
    # Z (Context): Already in batch
    Z_calib = batch_booking['context'].to(cfg.device)
    
    # s_hat (Inclusive Value): Need to compute using beta_hat
    # s = LogSumExp(X * beta)
    items_calib = batch_booking['items'].to(cfg.device) # (N, L, D)
    mask_calib = batch_booking['mask'].to(cfg.device)   # (N, L)
    
    # Compute Utilities U = X * beta
    # beta_hat: (1, D) -> Transpose to (D, 1)
    U_calib = (items_calib @ beta_hat.t()).squeeze(-1) # (N, L)
    
    # Mask padding before LogSumExp
    U_calib = U_calib.masked_fill(mask_calib == 0, -1e9)
    s_calib = torch.logsumexp(U_calib, dim=1) # (N,)
    
    # y_sim (Simulator Logit):
    # Strategy: 
    # 1. Predict P(Book) for ALL rows in Test DF using CatBoost
    # 2. Aggregate by srch_id to get P(NoBuy) for each session
    # 3. Select only the sessions corresponding to batch_booking['unique_ids']
    
    print("   Running Simulator on all test data...")
    # CatBoost needs flat data. Use loader helper.
    X_test_flat, _, _ = test_loader.get_catboost_data(only_clicks=False)
    
    # Predict Proba (Class 1 = Booking)
    sim_preds_flat = sim_model.predict_proba(X_test_flat)[:, 1]
    
    # Add to DataFrame temporarily for efficient grouping
    test_df['sim_p_book'] = sim_preds_flat
    
    # Aggregate: P(NoBuy) = 1 - sum(P_book_items)
    # (Heuristic approximation for point-wise simulators)
    sim_session_p_book = test_df.groupby('srch_id')['sim_p_book'].sum()
    sim_session_p_nobuy = 1.0 - sim_session_p_book
    
    # Clip to avoid log(0) or log(negative)
    sim_session_p_nobuy = sim_session_p_nobuy.clip(0.001, 0.999)
    sim_session_logit = np.log(sim_session_p_nobuy / (1 - sim_session_p_nobuy))
    
    # Align with PyTorch Batch
    # batch_booking['unique_ids'] contains the srch_ids in the tensor order
    target_srch_ids = batch_booking['unique_ids']
    
    # Select and convert to Tensor
    y_calib_np = sim_session_logit.loc[target_srch_ids].values
    y_calib = torch.from_numpy(y_calib_np).float().to(cfg.device)
    
    # 4. Run Solvers
    print("   Solving Calibration...")
    solver = CalibrationSolver(cfg)
    
    # Linear
    gamma_lin = solver.solve_linear(Z_calib, s_calib, y_calib)
    
    # MRC
    gamma_mrc = solver.solve_mrc(Z_calib, s_calib, y_calib)
    
    print(f"   Gamma Norms | Linear: {gamma_lin.norm().item():.2f}, MRC: {gamma_mrc.norm().item():.2f}")

    # --- E. Final Evaluation ---
    print("\n>>> Step 3: Evaluation on FULL Test Set (Hidden No-Purchases)...")
    
    # We want to predict P(No Purchase) for ALL sessions in Test (Booked + Not Booked)
    # And compare with Ground Truth.
    
    # 1. Ground Truth Labels
    # If a session has 0 bookings, label=1 (No Purchase). Else 0.
    gt_labels = test_df.groupby('srch_id')['booking_bool'].sum().apply(lambda x: 1 if x==0 else 0)
    y_true = gt_labels.values
    
    # 2. Simulator Predictions (Baseline)
    # We already calculated sim_session_p_nobuy for everyone
    # Make sure order aligns with y_true (groupby sorts by key by default)
    p_sim = sim_session_p_nobuy.values
    
    # 3. Model Predictions (Linear & MRC)
    # We need Z and s_hat for ALL sessions (not just booked ones)
    
    # Get Full Batch
    batch_full = test_loader.get_pytorch_data(filter_booking_only=False)
    
    # Z (Full)
    Z_full = batch_full['context'].to(cfg.device)
    
    # s (Full) - Recompute using beta_hat
    items_full = batch_full['items'].to(cfg.device)
    mask_full = batch_full['mask'].to(cfg.device)
    
    U_full = (items_full @ beta_hat.t()).squeeze(-1)
    U_full = U_full.masked_fill(mask_full == 0, -1e9)
    s_full = torch.logsumexp(U_full, dim=1)
    
    # Helper: Prob Function
    # Logit(P0) = u0 - s = Z*gamma - s
    def get_p0(gamma, Z, s):
        u0 = Z @ gamma
        eta = u0 - s
        return torch.sigmoid(eta).cpu().numpy()
    
    p_lin = get_p0(gamma_lin, Z_full, s_full)
    p_mrc = get_p0(gamma_mrc, Z_full, s_full)
    
    # 4. Compute NLL (Log Loss)
    nll_sim = log_loss(y_true, p_sim)
    nll_lin = log_loss(y_true, p_lin)
    nll_mrc = log_loss(y_true, p_mrc)
    
    print("\n" + "="*40)
    print(f" EVALUATION RESULTS (Test Set N={len(y_true)})")
    print(" Negative Log Likelihood (Lower is Better)")
    print("="*40)
    print(f" 1. Simulator (Uncalibrated):  {nll_sim:.5f}")
    print(f" 2. Linear Calibration:        {nll_lin:.5f}")
    print(f" 3. MRC Calibration (Ours):    {nll_mrc:.5f}")
    print("="*40)
    
    # Interpretation
    print("\n[Analysis]")
    if nll_mrc < nll_sim:
        print("SUCCESS: MRC improved over the raw simulator.")
    if nll_mrc < nll_lin:
        print("SUCCESS: MRC outperformed Linear calibration (Robustness verified).")
    else:
        print("NOTE: Linear performed similarly/better. Check if Simulator bias is effectively linear.")

if __name__ == "__main__":
    run_pipeline()