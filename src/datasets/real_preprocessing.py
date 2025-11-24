import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ..config import RealExpConfig

class ExpediaPreprocessor:
    """
    Handles loading, cleaning, temporal splitting, and Kaggle-style feature engineering
    for the Expedia dataset.
    """
    def __init__(self, cfg: RealExpConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        
        # Metadata to be populated after loading
        self.final_feat_cols = []      
        self.cat_features_indices = [] 

    def load_and_process(self, nrows=None):
        """
        Main execution pipeline.
        Args:
            nrows: Limit rows for debugging (e.g. 100_000). None for full data.
        Returns:
            train_df (Historical), test_df (Current)
        """
        print(f"[Preprocessor] Loading raw data from {self.cfg.raw_data_path}...")
        
        # 1. Define columns to read
        # We need IDs, Time, Labels, plus all features defined in Config
        cols_to_load = list(set(
            ['srch_id', 'date_time', 'booking_bool', 'click_bool', 'prop_id'] + 
            list(self.cfg.context_cols) + 
            list(self.cfg.base_item_cols) + 
            list(self.cfg.group_stats_cols)
        ))
        
        # Load CSV
        df = pd.read_csv(self.cfg.raw_data_path, usecols=cols_to_load, nrows=nrows)
        print(f"[Preprocessor] Loaded {len(df)} rows.")
        
        # --- 2. Basic Cleaning ---
        print("[Preprocessor] Cleaning data...")
        
        # Fill NaNs for Review Score (Mean Imputation)
        if 'prop_review_score' in df.columns:
            mean_score = df['prop_review_score'].mean()
            df['prop_review_score'] = df['prop_review_score'].fillna(mean_score)
            
        # Fill NaNs for Location Score 2 (Zero Imputation - standard for this dataset)
        if 'prop_location_score2' in df.columns:
            df['prop_location_score2'] = df['prop_location_score2'].fillna(0)
            
        # Log-transform Price (Handle heavy tails)
        if 'price_usd' in df.columns:
            # Clip negative prices to 0, add 1 to avoid log(0)
            df['price_usd'] = np.log1p(df['price_usd'].clip(lower=0))
            
        # --- 3. Temporal Split ---
        print(f"[Preprocessor] Sorting and splitting by time (Ratio: {self.cfg.split_ratio})...")
        
        # Sort by time
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.sort_values('date_time').reset_index(drop=True)
        
        # Split Logic: Must NOT split a single session (srch_id) in half.
        # We split based on unique srch_ids.
        unique_ids = df['srch_id'].unique()
        split_idx = int(len(unique_ids) * self.cfg.split_ratio)
        
        # Identify IDs belonging to History (Train)
        train_srch_ids = set(unique_ids[:split_idx])
        
        # Create Masks
        is_train = df['srch_id'].isin(train_srch_ids)
        
        train_df = df[is_train].copy()
        test_df = df[~is_train].copy()
        
        print(f"   Train Set (Historical): {len(train_df)} rows")
        print(f"   Test Set (Current):     {len(test_df)} rows")
        
        # --- 4. Feature Engineering (Kaggle Style: Group Stats) ---
        print("[Preprocessor] Generating Prop_ID aggregate features...")
        
        # We calculate stats ONLY on Train to avoid data leakage
        stats_cols = list(self.cfg.group_stats_cols)
        agg_funcs = ['mean', 'std', 'count'] # Mean, Std, and Popularity (Count)
        
        # GroupBy prop_id
        prop_stats = train_df.groupby('prop_id')[stats_cols].agg(agg_funcs)
        
        # Flatten MultiIndex columns (e.g., ('price_usd', 'mean') -> 'price_usd_mean')
        prop_stats.columns = [f"{col}_{stat}" for col, stat in prop_stats.columns]
        prop_stats = prop_stats.reset_index()
        
        # Rename count columns to be cleaner (e.g. price_usd_count -> prop_popularity)
        # Actually, count is the same for all cols, just keep one or rename systematically
        # Let's fill NaNs (e.g. std dev of single item is NaN)
        prop_stats = prop_stats.fillna(0)
        
        # Merge stats into Train and Test
        # Left join preserves the original rows
        train_df = train_df.merge(prop_stats, on='prop_id', how='left')
        test_df = test_df.merge(prop_stats, on='prop_id', how='left')
        
        # Handle "Cold Start" in Test Set
        # (Items appearing in Test but never seen in Train will have NaNs for stats)
        new_feat_cols = list(prop_stats.columns)
        new_feat_cols.remove('prop_id') # Remove ID from feature list
        
        for col in new_feat_cols:
            # Fill with global mean from Train
            global_mean = train_df[col].mean()
            train_df[col] = train_df[col].fillna(global_mean)
            test_df[col] = test_df[col].fillna(global_mean)
            
        # --- 5. Define Final Feature Lists & CatBoost Prep ---
        # Categorical columns that CatBoost handles natively
        cat_cols = ['site_id', 'visitor_location_country_id', 'prop_id']
        
        # Ensure they are integer type (CatBoost requirement) and handle NaNs
        for c in cat_cols:
            if c in train_df.columns:
                # Fill NaN IDs with -1
                train_df[c] = train_df[c].fillna(-1).astype(int)
                test_df[c] = test_df[c].fillna(-1).astype(int)

        # Construct the final list of features to be used by models
        # Base Items + Context + New Group Stats
        # Exclude structural columns
        exclude_cols = {'srch_id', 'date_time', 'booking_bool', 'click_bool'}
        
        self.final_feat_cols = [
            c for c in train_df.columns 
            if c not in exclude_cols
        ]
        
        # Identify indices of categorical features for CatBoost
        self.cat_features_indices = [
            i for i, c in enumerate(self.final_feat_cols) 
            if c in cat_cols
        ]
        
        print(f"[Preprocessor] Final Feature Count: {len(self.final_feat_cols)}")
        print(f"[Preprocessor] Categorical Features Indices: {self.cat_features_indices}")
        
        # --- 6. Scaling (StandardScaler) ---
        print("[Preprocessor] Scaling numerical features...")
        
        # Scale everything EXCEPT the categorical IDs
        num_cols = [c for c in self.final_feat_cols if c not in cat_cols]
        
        if num_cols:
            self.scaler.fit(train_df[num_cols])
            train_df[num_cols] = self.scaler.transform(train_df[num_cols])
            test_df[num_cols]  = self.scaler.transform(test_df[num_cols])
        
        # Clean up unnecessary columns to save memory
        train_df.drop(columns=['date_time'], inplace=True)
        test_df.drop(columns=['date_time'], inplace=True)
        
        return train_df, test_df