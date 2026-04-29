import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from scipy import stats

from backend.config.data_paths import tick_orderbook_dir

# Config
SYMBOL = "BTCUSDT"
DATA_DIR = str(tick_orderbook_dir(SYMBOL))
DEPTHS = [5, 10, 20, 50]
FWD_WINDOWS = [1, 5, 15, 30] # minutes
FOLDS = 14

def load_data(symbol):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    if not files:
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
    
    if not dfs:
        return None
        
    full_df = pd.concat(dfs).sort_values("timestamp")
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], unit="ms")
    full_df.set_index("timestamp", inplace=True)
    return full_df

def compute_imbalance(df, depth):
    # Assuming schema: bid_prices, bid_sizes, ask_prices, ask_sizes (lists/arrays)
    # If they are already pre-computed in the parquet, use them.
    # The schema from previous check: timestamp, bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price, spread, imbalance
    # Let's check if 'imbalance' is already there and what depth it is.
    
    # If we need to compute for specific depths:
    # imbalance = (sum(bid_sizes[:depth]) - sum(ask_sizes[:depth])) / (sum(bid_sizes[:depth]) + sum(ask_sizes[:depth]))
    
    # For now, let's assume the 'imbalance' column is depth 20 (standard for our collector)
    # and we'll use it as the primary feature.
    return df["imbalance"]

def run_deep_dive():
    print(f"Finding {SYMBOL} files...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    if not files:
        print("No data found.")
        return

    print(f"Found {len(files)} files. Processing file by file...")
    
    all_results = []
    
    for f in files:
        try:
            print(f"Processing {os.path.basename(f)}...")
            df = pd.read_parquet(f)
            if df.empty:
                continue
                
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            # Forward returns
            for w in FWD_WINDOWS:
                # Assuming 1s snapshots, shift by w*60 rows
                df[f"fwd_ret_{w}m"] = df["mid_price"].shift(-w * 60).pct_change(w * 60)

            df.dropna(inplace=True)
            if df.empty:
                continue
                
            feature = "imbalance"
            res = {"file": os.path.basename(f), "date": df.index[0].date()}
            
            for w in FWD_WINDOWS:
                target = f"fwd_ret_{w}m"
                corr, p = stats.pearsonr(df[feature], df[target])
                acc = (np.sign(df[feature]) == np.sign(df[target])).mean()
                
                # Extreme imbalance (top/bottom 5%)
                q_high = df[feature].quantile(0.95)
                q_low = df[feature].quantile(0.05)
                
                extreme_high = df[df[feature] > q_high]
                extreme_low = df[df[feature] < q_low]
                
                if not extreme_high.empty and not extreme_low.empty:
                    acc_high = (extreme_high[target] > 0).mean()
                    acc_low = (extreme_low[target] < 0).mean()
                    acc_extreme = (acc_high + acc_low) / 2
                else:
                    acc_extreme = np.nan
                
                res[f"corr_{w}m"] = corr
                res[f"acc_{w}m"] = acc
                res[f"acc_extreme_{w}m"] = acc_extreme
                
            all_results.append(res)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    res_df = pd.DataFrame(all_results)
    res_df.to_csv("ob_imbalance_results.csv", index=False)
    
    print("\n--- SUMMARY ---")
    for w in FWD_WINDOWS:
        print(f"\nWindow: {w}m")
        print(f"Mean Accuracy: {res_df[f'acc_{w}m'].mean():.4f}")
        print(f"Mean Extreme Accuracy: {res_df[f'acc_extreme_{w}m'].mean():.4f}")
        print(f"Mean Correlation: {res_df[f'corr_{w}m'].mean():.4f}")
        print(f"Best Day Accuracy: {res_df[f'acc_{w}m'].max():.4f}")

if __name__ == "__main__":
    run_deep_dive()
