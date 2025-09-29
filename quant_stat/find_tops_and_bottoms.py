import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, SYMBOL
from fractal_detector import UnifiedZigzagDetector, RealTimeExtremeDetector, FractalType
from plot_minute_data import plot_minute_data

from find_pending_creek import analyze_pending_creeks

# Configuration variables
# CHANGE_PCT is now passed as parameter from main.py


def load_es_data(filename: str) -> pd.DataFrame:
    """Load ES clean data from file"""
    file_path = os.path.join(str(DATA_DIR), filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from {filename}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Time range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    return df

def classify_swing_size(distance_usd: float, distance_bars: int) -> str:
    """
    Classify swing size based on price movement and time duration
    """
    # ES futures typical swing classification
    # Adjust these thresholds based on market volatility and timeframe

    # For ES 1-minute data, these are reasonable thresholds:
    if distance_usd >= 15.0:  # Large price moves
        return "big"
    elif distance_usd >= 5.0:  # Medium price moves
        return "small"
    else:  # Small price moves
        return "noise"

def detect_fractals_zigzag(df: pd.DataFrame, min_change_pct: float = 0.15) -> list:
    """
    Detect fractals using UnifiedZigzagDetector
    """
    print(f"\nDetecting fractals with Zigzag method (min_change: {min_change_pct}%)")

    detector = UnifiedZigzagDetector(min_change_pct=min_change_pct)
    fractals = []

    for idx, row in df.iterrows():
        timestamp = f"{row['date']} {row['time']}"
        pivot = detector.add_candle(
            high=row['high'],
            low=row['low'],
            index=idx,
            timestamp=timestamp
        )

        if pivot:
            fractal_type = "TOP" if pivot.direction.value == "up" else "BOTTOM"

            # Calculate distance from previous fractal
            distance_usd = 0.0
            distance_bars = 0
            distance_ratio = 0.0
            swing_size = "noise"  # Default classification

            if len(fractals) > 0:
                prev_fractal = fractals[-1]
                distance_usd = round(abs(pivot.price - prev_fractal['price']), 2)
                distance_bars = pivot.index - prev_fractal['index']

                # Calculate distance_usd*100/(distance_bars) ratio
                if distance_bars > 0:
                    distance_ratio = round((distance_usd * 100) / distance_bars, 2)

                # Classify swing size
                swing_size = classify_swing_size(distance_usd, distance_bars)

            fractals.append({
                'index': pivot.index,
                'timestamp': pivot.timestamp,
                'price': round(pivot.price, 2),
                'type': fractal_type,
                'method': 'Zigzag',
                'distance_usd': distance_usd,
                'distance_bars': distance_bars,
                'distance_ratio': distance_ratio,
                'swing_size': swing_size
            })
            print(f"  -> {fractal_type} detected at {pivot.timestamp} - Price: {pivot.price:.2f} - Swing: {swing_size}")

    return fractals

def detect_fractals_window(df: pd.DataFrame, window_size: int = 7, confirmation_periods: int = 3) -> list:
    """
    Detect fractals using RealTimeExtremeDetector (window method)
    """
    print(f"\nDetecting fractals with Window method (window: {window_size}, confirmation: {confirmation_periods})")

    detector = RealTimeExtremeDetector(
        window_size=window_size,
        confirmation_periods=confirmation_periods,
        min_strength=0.00001
    )

    all_fractals = []

    for idx, row in df.iterrows():
        timestamp = f"{row['date']} {row['time']}"
        # Use close price for window method
        newly_confirmed = detector.add_price(row['close'], timestamp)

        for fractal in newly_confirmed:
            fractal_type = "TOP" if fractal.fractal_type == FractalType.PEAK else "BOTTOM"

            # Calculate distance from previous fractal
            distance_usd = 0.0
            distance_bars = 0
            distance_ratio = 0.0
            swing_size = "noise"  # Default classification

            if len(all_fractals) > 0:
                prev_fractal = all_fractals[-1]
                distance_usd = round(abs(fractal.price - prev_fractal['price']), 2)
                distance_bars = fractal.index - prev_fractal['index']

                # Calculate distance_usd*100/(distance_bars) ratio
                if distance_bars > 0:
                    distance_ratio = round((distance_usd * 100) / distance_bars, 2)

                # Classify swing size
                swing_size = classify_swing_size(distance_usd, distance_bars)

            all_fractals.append({
                'index': fractal.index,
                'timestamp': fractal.timestamp,
                'price': round(fractal.price, 2),
                'type': fractal_type,
                'method': 'Window',
                'distance_usd': distance_usd,
                'distance_bars': distance_bars,
                'distance_ratio': distance_ratio,
                'swing_size': swing_size
            })
            print(f"  -> {fractal_type} detected at {fractal.timestamp} - Price: {fractal.price:.2f} - Swing: {swing_size}")

    return all_fractals

def save_fractals_to_csv(fractals: list, output_filename: str):
    """Save detected fractals to CSV file"""
    # Create outputs directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    df_fractals = pd.DataFrame(fractals)

    # Calculate 3-period moving average of distance_ratio
    if 'distance_ratio' in df_fractals.columns and len(df_fractals) > 0:
        df_fractals['dist_ratio_avg'] = df_fractals['distance_ratio'].rolling(window=3, min_periods=1).mean().round(2)

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df_fractals.to_csv(output_path, index=False)

    print(f"\nFractals saved to: {output_path}")
    return output_path

def analyze_fractals(fractals: list) -> dict:
    """Analyze detected fractals and return statistics"""
    if not fractals:
        return {
            "total": 0, "tops": 0, "bottoms": 0,
            "top_percentage": 0, "bottom_percentage": 0,
            "highest_top": 0, "lowest_bottom": 0,
            "price_range": 0, "avg_top_price": 0, "avg_bottom_price": 0
        }

    df_fractals = pd.DataFrame(fractals)

    total_fractals = len(fractals)
    tops = len(df_fractals[df_fractals['type'] == 'TOP'])
    bottoms = len(df_fractals[df_fractals['type'] == 'BOTTOM'])

    # Price statistics
    prices = [f['price'] for f in fractals]
    top_prices = [f['price'] for f in fractals if f['type'] == 'TOP']
    bottom_prices = [f['price'] for f in fractals if f['type'] == 'BOTTOM']

    stats = {
        'total': total_fractals,
        'tops': tops,
        'bottoms': bottoms,
        'top_percentage': round((tops / total_fractals * 100), 1) if total_fractals > 0 else 0,
        'bottom_percentage': round((bottoms / total_fractals * 100), 1) if total_fractals > 0 else 0,
        'highest_top': max(top_prices) if top_prices else 0,
        'lowest_bottom': min(bottom_prices) if bottom_prices else 0,
        'price_range': round(max(prices) - min(prices), 2) if prices else 0,
        'avg_top_price': round(sum(top_prices) / len(top_prices), 2) if top_prices else 0,
        'avg_bottom_price': round(sum(bottom_prices) / len(bottom_prices), 2) if bottom_prices else 0
    }

    return stats

def print_fractal_summary(fractals: list, method_name: str):
    """Print a formatted summary of detected fractals"""
    stats = analyze_fractals(fractals)

    print(f"\n{'='*60}")
    print(f"FRACTAL ANALYSIS SUMMARY - {method_name.upper()} METHOD")
    print(f"{'='*60}")
    print(f"Total Fractals Detected: {stats['total']}")
    print(f"  Tops:               {stats['tops']} ({stats['top_percentage']}%)")
    print(f"  Bottoms:            {stats['bottoms']} ({stats['bottom_percentage']}%)")
    print(f"\nPrice Analysis:")
    print(f"  Highest Top:      ${stats['highest_top']}")
    print(f"  Lowest Bottom:    ${stats['lowest_bottom']}")
    print(f"  Price Range:      ${stats['price_range']}")
    print(f"  Avg Top Price:    ${stats['avg_top_price']}")
    print(f"  Avg Bottom Price: ${stats['avg_bottom_price']}")

    if fractals:
        print(f"\nTiming Analysis:")
        df_fractals = pd.DataFrame(fractals)
        first_fractal = df_fractals.iloc[0]
        last_fractal = df_fractals.iloc[-1]
        print(f"  First Fractal: {first_fractal['type']} at {first_fractal['timestamp']} (${first_fractal['price']})")
        print(f"  Last Fractal:  {last_fractal['type']} at {last_fractal['timestamp']} (${last_fractal['price']})")

        # Analyze swing sizes
        print_swing_analysis(fractals)

    print(f"{'='*60}")

def print_swing_analysis(fractals: list):
    """Print detailed analysis of swing sizes"""
    if not fractals:
        return

    df_fractals = pd.DataFrame(fractals)

    # Count swing sizes
    swing_counts = df_fractals['swing_size'].value_counts()

    print(f"\nSWING SIZE ANALYSIS:")
    print(f"{'='*60}")
    for swing_type in ['big', 'small', 'noise']:
        count = swing_counts.get(swing_type, 0)
        percentage = (count / len(fractals) * 100) if len(fractals) > 0 else 0
        print(f"  {swing_type.upper()} swings:     {count:2d} ({percentage:.1f}%)")

    # Detailed analysis of BIG moves
    big_swings = df_fractals[df_fractals['swing_size'] == 'big']

    if len(big_swings) > 0:
        print(f"\nBIG MOVES DETAILS:")
        print(f"{'='*80}")
        print(f"{'#':<2} {'Time':<20} {'Type':<6} {'Price':<8} {'Move $':<8} {'Bars':<6} {'From Previous'}")
        print(f"{'-'*80}")

        for i, (idx, swing) in enumerate(big_swings.iterrows(), 1):
            time_str = swing['timestamp']
            swing_type = swing['type']
            price = f"${swing['price']}"
            move_usd = f"${swing['distance_usd']}"
            bars = swing['distance_bars']

            # Find previous fractal for context
            prev_idx = idx - 1 if idx > 0 else 0
            if prev_idx >= 0 and prev_idx < len(df_fractals):
                prev_fractal = df_fractals.iloc[prev_idx]
                from_prev = f"${prev_fractal['price']} -> ${swing['price']}"
            else:
                from_prev = "First fractal"

            print(f"{i:<2} {time_str:<20} {swing_type:<6} {price:<8} {move_usd:<8} {bars:<6} {from_prev}")

        # Summary statistics for big moves
        total_big_move = big_swings['distance_usd'].sum()
        avg_big_move = big_swings['distance_usd'].mean()
        max_big_move = big_swings['distance_usd'].max()
        avg_bars = big_swings['distance_bars'].mean()

        print(f"{'-'*80}")
        print(f"BIG MOVES SUMMARY:")
        print(f"  Total Movement:     ${total_big_move:.2f}")
        print(f"  Average Move:       ${avg_big_move:.2f}")
        print(f"  Largest Move:       ${max_big_move:.2f}")
        print(f"  Average Duration:   {avg_bars:.1f} bars")

    else:
        print(f"\nNo BIG moves detected with current thresholds (≥$15.00)")

    print(f"{'='*80}")

def print_detailed_fractals(fractals: list, max_display: int = 20):
    """Print detailed list of detected fractals"""
    if not fractals:
        print("No fractals detected.")
        return

    # Convert to DataFrame to access the rolling average if available
    df_fractals = pd.DataFrame(fractals)

    # Calculate 3-period moving average if not already calculated
    if 'distance_ratio' in df_fractals.columns and 'dist_ratio_avg' not in df_fractals.columns:
        df_fractals['dist_ratio_avg'] = df_fractals['distance_ratio'].rolling(window=3, min_periods=1).mean().round(2)

    print(f"\nDETAILED FRACTAL LIST (showing first {min(max_display, len(fractals))} of {len(fractals)}):")
    print(f"{'='*155}")
    print(f"{'Index':<6} {'Timestamp':<20} {'Type':<8} {'Price':<10} {'Dist USD':<10} {'Bars':<6} {'Ratio':<10} {'Avg3':<10} {'Swing':<10} {'Method':<10}")
    print(f"{'-'*155}")

    for i in range(min(max_display, len(df_fractals))):
        fractal = df_fractals.iloc[i]
        type_icon = "^" if fractal['type'] == 'TOP' else "v"
        dist_usd = fractal.get('distance_usd', 0.0)
        dist_bars = fractal.get('distance_bars', 0)
        dist_ratio = fractal.get('distance_ratio', 0.0)
        dist_ratio_avg = fractal.get('dist_ratio_avg', 0.0)
        swing_size = fractal.get('swing_size', 'noise')
        print(f"{fractal['index']:<6} {fractal['timestamp']:<20} {type_icon} {fractal['type']:<6} ${fractal['price']:<9} ${dist_usd:<9} {dist_bars:<6} {dist_ratio:<10} {dist_ratio_avg:<10} {swing_size:<10} {fractal['method']:<10}")

    if len(fractals) > max_display:
        print(f"\n... and {len(fractals) - max_display} more fractals")
    print(f"{'-'*155}")

def main(method='zigzag', change_pct=0.05, window_size=7, confirmation_periods=3):
    """Main function to detect tops and bottoms in ES data"""
    print("FRACTAL DETECTION - TOPS AND BOTTOMS FINDER")
    print("="*60)

    # Configuration
    data_filename = 'es_1min_clean_2023_03_02.csv'

    # Generate filename based on method
    if method == 'zigzag':
        output_filename = f'fractals_2023_03_02_zigzag_{change_pct}.csv'
        print(f"Using ZIGZAG method with {change_pct}% minimum change")
    else:  # window method
        output_filename = f'fractals_2023_03_02_window_{window_size}_{confirmation_periods}.csv'
        print(f"Using WINDOW method with window size {window_size} and {confirmation_periods} confirmation periods")

    try:
        # Load data
        print("Loading ES 1-minute data...")
        df = load_es_data(data_filename)

        # Detect fractals using selected method
        if method == 'zigzag':
            primary_fractals = detect_fractals_zigzag(df, min_change_pct=change_pct)
            method_name = "Zigzag"

            # Also run window method for comparison
            fractals_window = detect_fractals_window(df, window_size=window_size, confirmation_periods=confirmation_periods)
        else:  # window method
            primary_fractals = detect_fractals_window(df, window_size=window_size, confirmation_periods=confirmation_periods)
            method_name = "Window"

            # Also run zigzag method for comparison
            fractals_zigzag = detect_fractals_zigzag(df, min_change_pct=change_pct)

        # Display results
        print_fractal_summary(primary_fractals, method_name)
        print_detailed_fractals(primary_fractals, max_display=25)

        # Save results
        if primary_fractals:
            output_path = save_fractals_to_csv(primary_fractals, output_filename)

            # Show file contents
            print(f"\nContents of {output_filename}:")
            print("-" * 50)
            df_output = pd.read_csv(output_path)
            print(df_output.to_string(index=False))

            # Analyze pending creeks with dynamic movement
            df_fractals = pd.DataFrame(primary_fractals)
            pending_creeks, creek_lines = analyze_pending_creeks(df_fractals, df)

            # Store pending creek data for plotting
            if pending_creeks:
                creek_output_file = output_filename.replace('.csv', '_pending_creeks.csv')
                creek_output_path = os.path.join('outputs', creek_output_file)
                pd.DataFrame(pending_creeks).to_csv(creek_output_path, index=False)
                print(f"Pending creek data saved to: {creek_output_path}")

        else:
            print("No fractals detected to save.")
            pending_creeks = []
            creek_lines = []

        # Compare methods if both were run
        if method == 'zigzag' and 'fractals_window' in locals() and fractals_window:
            print(f"\nMETHOD COMPARISON:")
            print(f"Zigzag Method (ACTIVE):  {len(primary_fractals)} fractals")
            print(f"Window Method:           {len(fractals_window)} fractals")
        elif method == 'window' and 'fractals_zigzag' in locals() and fractals_zigzag:
            print(f"\nMETHOD COMPARISON:")
            print(f"Window Method (ACTIVE):  {len(primary_fractals)} fractals")
            print(f"Zigzag Method:           {len(fractals_zigzag)} fractals")

        print(f"\nFractal detection completed successfully!")

        # Return the fractal data for main.py to use
        return primary_fractals, pending_creeks

    except Exception as e:
        print(f"Error during fractal detection: {e}")
        return [], []

if __name__ == "__main__":
    main()