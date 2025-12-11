import torch
from dataclasses import dataclass
from typing import Literal, Optional

def get_default_device() -> str:
    """
    Select the best available device: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
# ==========================================
# 1. Synthetic Data Configuration
# ==========================================
@dataclass
class ExpConfig:
    # ==========================================
    # 1. Infrastructure
    # ==========================================
    seed: int = 42
    device: str = get_default_device()
    
    # ==========================================
    # 2. Data Dimensions
    # ==========================================
    n_samples: int = 2000           # Number of samples (N)
    dim_context:int =24
    outlier_prob: float = 0.0
    outlier_scale: float = 50.0
    sim_noise_type: str = 'gaussian'
    # Assortment Size Range (for variable size S)
    # Data generation will sample size uniformly from [min, max]
    min_assortment_size: int = 5
    max_assortment_size: int = 15
    
    # Feature Dimensions
    dim_item_feat: int = 3          # Dimension of atomic item features
    # Number of items
    n_items_pool: int = 1000  
    
    # Feature Covariance Control
    item_feat_corr: float = 0.5
    
    # Dimension of the mapped feature z(X)
    # Needed to initialize the downstream regression/MRC weights
    dim_z: int = 24

    # ==========================================
    # 3. Utility & Estimation Protocol
    # ==========================================
    # Controls the source of the estimated inclusive value \hat{s}
    # 'noise_injection': Direct s_hat = s_true + N(0, tau). Verifies Theorem 2.
    # 'mle_estimation':  Simulate choices -> Run Conditional MNL. Verifies end-to-end pipeline.
    utility_mode: Literal['structural', 'additive'] = 'structural'
    
    # [NEW] Noise Distribution Type
    # 'gaussian': Normal distribution N(0, sigma^2). Targets MSE (bar_tau).
    # 'uniform':  Uniform distribution U[-sigma, sigma]. Targets Max Error (tau_s).
    # Note: For 'structural' mode, this dictates the distribution of delta_beta.
    noise_distribution: Literal['gaussian', 'uniform'] = 'gaussian'
    
    # Parameter for 'noise_injection' mode:
    # Standard deviation of the estimation error (controls \tau)
    est_noise_sigma: float = 0.1        
    
    # ==========================================
    # 4. Simulator Settings (y generation)
    # ==========================================
    # Bias parameters (intercept a, slope b)
    sim_bias_a: float = 1.0
    sim_bias_b: float = 1.5
    
    # Simulator Noise (epsilon)
    sim_noise_sigma: float = 0.5  

    # ==========================================
    # 5. Real Data
    # ==========================================
    # Path to the processed real data file (e.g., .csv or .pt)
    # If None, the system runs in Synthetic mode.
    real_data_path: Optional[str] = None
    
# ==========================================
# 2. Real Data Configuration
# ==========================================
@dataclass
class RealExpConfig:
    """Configuration for Real Data Experiments (Expedia)"""
    seed: int = 42
    device: str = get_default_device()
    
    # Data Source (We load data, so we need paths)
    raw_data_path: str = "data/train.csv"
    split_ratio: float = 0.7 
    
    # Feature Engineering (Kaggle Style)
    group_stats_cols: tuple = (
        'price_usd', 
        'prop_starrating', 
        'prop_review_score', 
        'prop_location_score1', 
        'prop_location_score2'
    )

    context_cols: tuple = (
        'site_id', 
        'visitor_location_country_id', 
        'srch_length_of_stay', 
        'srch_booking_window',
        'srch_adults_count', 
        'srch_children_count', 
        'srch_room_count', 
        'srch_saturday_night_bool',
        'random_bool' 
    )
    
    base_item_cols: tuple = (
        'prop_starrating', 
        'prop_review_score', 
        'prop_brand_bool',
        'prop_location_score1', 
        'prop_location_score2', 
        'price_usd', 
        'promotion_flag'
    )
    
    # Model Hyperparameters (ML Models instead of Math Formulas)
    # CatBoost
    cat_iterations: int = 500 
    cat_depth: int = 6
    cat_learning_rate: float = 0.1
    cat_early_stopping: int = 50
    
    # Conditional MNL
    utility_model_type: str = 'linear'
    mnl_epochs: int = 30
    mnl_lr: float = 0.05
    mnl_batch_size: int = 4096