import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from ..config import RealExpConfig

class ExpediaDataLoader:
    """
    Efficiently converts Pandas DataFrames into:
    1. Flat format for CatBoost (CPU)
    2. Nested/Padded format for PyTorch MNL (MPS/GPU)
    """
    def __init__(self, cfg: RealExpConfig, df: pd.DataFrame, feat_cols: list):
        self.cfg = cfg
        self.df = df
        self.feat_cols = feat_cols
        
        # Identify sub-feature groups indices for fast slicing
        self.item_feat_indices = [i for i, c in enumerate(feat_cols) 
                                  if c in cfg.base_item_cols or '_mean' in c or '_std' in c]
        
        self.ctx_feat_indices = [i for i, c in enumerate(feat_cols) 
                                 if c in cfg.context_cols]

    def get_catboost_data(self, target_col='booking_bool', only_clicks=False):
        """
        Returns (X, y, cat_indices) for CatBoost training.
        Zero overhead: just passes views of the dataframe.
        """
        target_df = self.df
        if only_clicks:
            target_df = self.df[self.df['click_bool'] == 1]
            
        X = target_df[self.feat_cols]
        y = target_df[target_col]
        
        # Identify categorical feature indices for CatBoost
        # (We calculated these in preprocessor, but recalculate here for safety based on current columns)
        cat_cols = ['site_id', 'visitor_location_country_id', 'prop_id']
        cat_indices = [i for i, c in enumerate(self.feat_cols) if c in cat_cols]
        
        return X, y, cat_indices

    def get_pytorch_data(self, filter_booking_only=True):
        """
        Converts DataFrame to Padded Tensors for MNL.
        Strategy: Sort -> Numpy Conversion -> Index Slicing (10x faster than groupby)
        """
        device = torch.device(self.cfg.device)
        
        # 1. Filter Data
        if filter_booking_only:
            # Find srch_ids that have at least one booking
            valid_srch_ids = self.df[self.df['booking_bool'] == 1]['srch_id'].unique()
            target_df = self.df[self.df['srch_id'].isin(valid_srch_ids)].copy()
        else:
            target_df = self.df.copy()
            
        # 2. Sort by srch_id (Crucial for slicing)
        # Note: It usually comes sorted from preprocessor, but ensure safety
        if not target_df['srch_id'].is_monotonic_increasing:
            target_df = target_df.sort_values('srch_id')
            
        # 3. Convert to Numpy (Huge Speedup over Pandas iterrows)
        # We pull out the big feature matrix once
        all_features = target_df[self.feat_cols].values.astype(np.float32)
        all_srch_ids = target_df['srch_id'].values
        all_bookings = target_df['booking_bool'].values
        
        # 4. Find Session Boundaries
        # unique_ids -> The session IDs
        # start_indices -> Where each session starts in the flat array
        unique_ids, start_indices = np.unique(all_srch_ids, return_index=True)
        
        # Calculate lengths of each session
        # Append total length to handle the last session
        end_indices = np.concatenate([start_indices[1:], [len(target_df)]])
        
        # 5. Vectorized Tensor Construction
        # We construct lists first, then pad. 
        
        items_list = []
        ctx_list = []
        label_list = []
        
        # We loop over sessions (e.g., 20k iters).
        # Since operations inside are numpy slices, this is very fast (<1s).
        
        # Optimization: Pre-slice indices
        item_indices = np.array(self.item_feat_indices)
        ctx_indices = np.array(self.ctx_feat_indices)
        
        for start, end in zip(start_indices, end_indices):
            # Slice rows for this session
            session_rows = all_features[start:end]
            session_bookings = all_bookings[start:end]
            
            # A. Items (X)
            # session_rows[:, item_indices] -> (L, D_item)
            items_list.append(torch.from_numpy(session_rows[:, item_indices]))
            
            # B. Context (Z)
            # Context is constant for the session, take first row
            # session_rows[0, ctx_indices] -> (D_ctx,)
            ctx_list.append(torch.from_numpy(session_rows[0, ctx_indices]))
            
            # C. Label (Index of booked item)
            # If booking exists, argmax gives index. If not, we use -1 (or handle later).
            # Since we filtered booking_only=True, there is always a 1.
            if filter_booking_only:
                label = np.argmax(session_bookings)
            else:
                # If checking for no-buy
                if session_bookings.sum() == 0:
                    label = -1 # Indicator for No Purchase
                else:
                    label = np.argmax(session_bookings)
                    
            label_list.append(label)

        # 6. Padding & Stacking
        # pad_sequence: (B, L_max, D)
        items_padded = pad_sequence(items_list, batch_first=True, padding_value=0.0)
        context_stacked = torch.stack(ctx_list)
        labels_stacked = torch.tensor(label_list, dtype=torch.long)
        
        # Create Mask (B, L_max)
        # Fast way: check where items_padded is not all zero? No, features can be zero.
        # Better: Create mask from lengths
        B, L_max, _ = items_padded.shape
        lengths = torch.tensor([t.shape[0] for t in items_list])
        
        # Vectorized mask creation
        # arange(L_max) < lengths[:, None] -> Broadcasting
        mask = torch.arange(L_max).expand(B, L_max) < lengths.unsqueeze(1)
        mask = mask.float()
        
        # 7. Move to Device (Batch Load)
        
        return {
            'items': items_padded,   # CPU
            'context': context_stacked, # CPU
            'mask': mask,            # CPU
            'labels': labels_stacked, # CPU
            'unique_ids': unique_ids # Track srch_id for aggregation later
        }