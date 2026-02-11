"""
Calibrate Synthetic Proxies Against Real Data

Compares proxy estimates to actual funding rates, OI, etc.
Outputs optimal scaling factors and correlation metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar
import json

from synthetic_proxies import (
    funding_proxy_basis,
    funding_proxy_vol_adjusted,
    funding_proxy_momentum,
    funding_proxy_lsr,
    oi_proxy_volume,
    oi_proxy_volatility,
    liquidation_proxy_wicks,
    lsr_proxy,
    calculate_atr
)


class ProxyCalibrator:
    """
    Calibrate proxy methods against real data.
    """
    
    def __init__(self, data_dir: str = './collected_data'):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def load_funding_rates(self, symbol: str = 'BTC/USDT') -> pd.DataFrame:
        """Load real funding rate data."""
        pattern = f"funding_rates_{symbol.replace('/', '')}*.csv"
        files = list(self.data_dir.glob(f"**/funding*{symbol.replace('/', '')}*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No funding rate files found for {symbol}")
        
        df = pd.read_csv(files[0], parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        return df
    
    def load_spot_prices(self, symbol: str = 'BTC/USDT') -> pd.DataFrame:
        """Load spot OHLCV data."""
        files = list(self.data_dir.glob(f"**/spot*{symbol.replace('/', '')}*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No spot files found for {symbol}")
        
        df = pd.read_csv(files[0], parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        return df
    
    def load_perp_prices(self, symbol: str = 'BTC/USDT', 
                        perp_dir: str = None) -> pd.DataFrame:
        """Load perpetual OHLCV data."""
        if perp_dir:
            search_path = Path(perp_dir)
        else:
            search_path = Path.home() / 'backtest_data'
        
        files = list(search_path.glob(f"*{symbol.replace('/', '_')}*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No perp files found for {symbol}")
        
        df = pd.read_csv(files[0])
        
        # Handle different timestamp column names
        ts_col = None
        for col in ['timestamp', 'time', 'date', 'datetime']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            # Assume first column is timestamp
            ts_col = df.columns[0]
        
        df['timestamp'] = pd.to_datetime(df[ts_col])
        df = df.sort_values('timestamp').set_index('timestamp')
        
        return df
    
    def align_data(self, funding_df: pd.DataFrame, 
                   ohlcv_df: pd.DataFrame,
                   spot_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Align funding rates with OHLCV data.
        
        Funding is every 8 hours, OHLCV may be hourly.
        Resample OHLCV to 8h to match funding frequency.
        """
        # Resample OHLCV to 8h
        ohlcv_8h = ohlcv_df.resample('8h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Join funding rates
        aligned = ohlcv_8h.join(funding_df[['funding_rate']], how='inner')
        
        # Add spot if available
        if spot_df is not None:
            spot_8h = spot_df.resample('8h').agg({
                'close': 'last'
            }).dropna()
            spot_8h.columns = ['spot_close']
            aligned = aligned.join(spot_8h, how='inner')
        
        return aligned.dropna()
    
    def calculate_all_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all proxy values for aligned data."""
        result = df.copy()
        
        # Funding proxies
        result['proxy_vol_adjusted'] = funding_proxy_vol_adjusted(df)
        result['proxy_momentum'] = funding_proxy_momentum(df)
        result['proxy_lsr'] = funding_proxy_lsr(df)
        
        # Basis proxy if spot available
        if 'spot_close' in df.columns:
            result['proxy_basis'] = funding_proxy_basis(df['close'], df['spot_close'])
        
        return result.dropna()
    
    def find_optimal_scale(self, proxy: pd.Series, 
                          real: pd.Series) -> Tuple[float, float]:
        """
        Find optimal scaling factor via linear regression.
        
        Returns:
            (scale, intercept)
        """
        # Clean data
        mask = ~(proxy.isna() | real.isna())
        x = proxy[mask].values
        y = real[mask].values
        
        if len(x) < 10:
            return 1.0, 0.0
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return slope, intercept
    
    def calculate_metrics(self, proxy: pd.Series, 
                         real: pd.Series,
                         scale: float = 1.0,
                         intercept: float = 0.0) -> Dict:
        """
        Calculate accuracy metrics.
        """
        # Clean and align
        mask = ~(proxy.isna() | real.isna())
        proxy_clean = proxy[mask] * scale + intercept
        real_clean = real[mask]
        
        if len(proxy_clean) < 10:
            return {'error': 'Insufficient data'}
        
        # Correlation
        corr = proxy_clean.corr(real_clean)
        
        # R-squared
        ss_res = ((real_clean - proxy_clean) ** 2).sum()
        ss_tot = ((real_clean - real_clean.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Direction accuracy
        proxy_sign = np.sign(proxy_clean)
        real_sign = np.sign(real_clean)
        direction_acc = (proxy_sign == real_sign).mean()
        
        # Extreme detection (top/bottom 20%)
        q20 = real_clean.quantile(0.2)
        q80 = real_clean.quantile(0.8)
        
        real_extreme_high = real_clean > q80
        real_extreme_low = real_clean < q20
        proxy_extreme_high = proxy_clean > proxy_clean.quantile(0.8)
        proxy_extreme_low = proxy_clean < proxy_clean.quantile(0.2)
        
        extreme_high_acc = (proxy_extreme_high == real_extreme_high).mean()
        extreme_low_acc = (proxy_extreme_low == real_extreme_low).mean()
        
        # MSE and MAE
        mse = ((real_clean - proxy_clean) ** 2).mean()
        mae = (real_clean - proxy_clean).abs().mean()
        
        return {
            'correlation': corr,
            'r_squared': r_squared,
            'direction_accuracy': direction_acc,
            'extreme_high_accuracy': extreme_high_acc,
            'extreme_low_accuracy': extreme_low_acc,
            'mse': mse,
            'mae': mae,
            'samples': len(proxy_clean),
            'optimal_scale': scale,
            'optimal_intercept': intercept
        }
    
    def calibrate_funding_proxies(self, symbol: str = 'BTC/USDT',
                                  perp_dir: str = None) -> Dict:
        """
        Calibrate all funding rate proxies.
        """
        print(f"\n{'='*60}")
        print(f"Calibrating Funding Rate Proxies for {symbol}")
        print(f"{'='*60}\n")
        
        # Load data
        funding_df = self.load_funding_rates(symbol)
        print(f"Loaded {len(funding_df)} funding rate records")
        
        try:
            spot_df = self.load_spot_prices(symbol)
            print(f"Loaded {len(spot_df)} spot candles")
        except:
            spot_df = None
            print("No spot data available")
        
        perp_df = self.load_perp_prices(symbol, perp_dir)
        print(f"Loaded {len(perp_df)} perp candles")
        
        # Align data
        aligned = self.align_data(funding_df, perp_df, spot_df)
        print(f"Aligned data: {len(aligned)} matching records")
        
        # Calculate proxies
        with_proxies = self.calculate_all_proxies(aligned)
        print(f"Calculated proxies: {len(with_proxies)} records with all values")
        
        # Calibrate each proxy
        results = {}
        real_funding = with_proxies['funding_rate']
        
        proxy_columns = [c for c in with_proxies.columns if c.startswith('proxy_')]
        
        for proxy_col in proxy_columns:
            proxy_name = proxy_col.replace('proxy_', '')
            proxy_values = with_proxies[proxy_col]
            
            # Find optimal scaling
            scale, intercept = self.find_optimal_scale(proxy_values, real_funding)
            
            # Calculate metrics
            metrics = self.calculate_metrics(proxy_values, real_funding, scale, intercept)
            
            results[proxy_name] = metrics
            
            print(f"\n{proxy_name.upper()}:")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  R²: {metrics['r_squared']:.4f}")
            print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
            print(f"  Optimal Scale: {scale:.6f}")
        
        self.results['funding'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate calibration report."""
        lines = [
            "# Proxy Calibration Report",
            f"Generated: {pd.Timestamp.now()}",
            "",
            "## Funding Rate Proxies",
            "",
            "| Proxy | Correlation | R² | Direction Acc | Extreme Acc | Optimal Scale |",
            "|-------|-------------|-----|--------------|-------------|---------------|"
        ]
        
        if 'funding' in self.results:
            for name, metrics in self.results['funding'].items():
                if 'error' in metrics:
                    continue
                lines.append(
                    f"| {name} | {metrics['correlation']:.3f} | "
                    f"{metrics['r_squared']:.3f} | {metrics['direction_accuracy']:.1%} | "
                    f"{(metrics['extreme_high_accuracy']+metrics['extreme_low_accuracy'])/2:.1%} | "
                    f"{metrics['optimal_scale']:.6f} |"
                )
        
        lines.extend([
            "",
            "## Recommendations",
            "",
            "Based on calibration results:",
        ])
        
        if 'funding' in self.results:
            # Find best proxy
            best = max(self.results['funding'].items(), 
                      key=lambda x: x[1].get('correlation', 0))
            lines.append(f"- **Best funding proxy:** {best[0]} (r={best[1]['correlation']:.3f})")
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str = 'calibration_results.json'):
        """Save calibration results to JSON."""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        results = convert(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


def run_calibration(symbol: str = 'BTC/USDT',
                   data_dir: str = './collected_data',
                   perp_dir: str = None):
    """
    Run full calibration pipeline.
    """
    calibrator = ProxyCalibrator(data_dir)
    
    # Calibrate funding proxies
    funding_results = calibrator.calibrate_funding_proxies(symbol, perp_dir)
    
    # Generate report
    report = calibrator.generate_report()
    print(f"\n{report}")
    
    # Save results
    calibrator.save_results('calibration_results.json')
    
    return calibrator.results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate synthetic proxies")
    parser.add_argument('--symbol', default='BTC/USDT', help='Symbol to calibrate')
    parser.add_argument('--data-dir', default='./collected_data', help='Data directory')
    parser.add_argument('--perp-dir', default=None, help='Perp OHLCV directory')
    
    args = parser.parse_args()
    
    results = run_calibration(
        symbol=args.symbol,
        data_dir=args.data_dir,
        perp_dir=args.perp_dir
    )
