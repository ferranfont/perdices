import pandas as pd
from typing import List, Dict, Optional, Tuple

class PendingCreekDetector:
    """
    Detects pending creek patterns: small swing tops that occur after big downtrend swings.
    A pending creek is a potential resistance level that may be tested in the future.
    """

    def __init__(self, lookback_bars: int = 60):
        """
        Initialize the pending creek detector

        Args:
            lookback_bars: Number of bars to extend the pending creek line
        """
        self.lookback_bars = lookback_bars
        self.pending_creeks = []

    def check_crossover(self, price_data: pd.DataFrame, creek_price: float, start_index: int, end_index: int) -> bool:
        """
        Check if price crossed over the creek level using close prices

        Args:
            price_data: DataFrame with price data including 'close' column
            creek_price: The creek resistance level
            start_index: Starting index to check from
            end_index: Ending index to check to

        Returns:
            True if price crossed over the creek level
        """
        if 'close' not in price_data.columns:
            return False

        # Get the relevant price range
        price_range = price_data.iloc[start_index:end_index+1]

        # Check if any close price crossed above the creek level
        crossover = (price_range['close'] > creek_price).any()

        return crossover

    def detect_pending_creeks(self, fractals_df: pd.DataFrame, price_data: pd.DataFrame = None) -> List[Dict]:
        """
        Detect pending creek patterns in fractal data with dynamic movement logic

        Args:
            fractals_df: DataFrame with fractal data including swing_size column
            price_data: DataFrame with price data for crossover checking

        Returns:
            List of pending creek dictionaries with details
        """
        pending_creeks = []

        if len(fractals_df) < 2:
            return pending_creeks

        # Reset index to ensure proper iteration
        fractals_df = fractals_df.reset_index(drop=True)

        # Look for pattern: big downtrend followed by small top
        for i in range(1, len(fractals_df)):
            current_fractal = fractals_df.iloc[i]
            previous_fractal = fractals_df.iloc[i-1]

            # Check for the pattern:
            # 1. Previous fractal is a BIG BOTTOM (end of big downtrend)
            # 2. Current fractal is ANY TOP (potential resistance) - no swing size filter
            if (previous_fractal['swing_size'] == 'big' and
                previous_fractal['type'] == 'BOTTOM' and
                current_fractal['type'] == 'TOP'):

                creek_info = {
                    'index': current_fractal['index'],
                    'timestamp': current_fractal['timestamp'],
                    'price': current_fractal['price'],
                    'type': 'PENDING_CREEK',
                    'trigger_bottom_price': previous_fractal['price'],
                    'trigger_bottom_time': previous_fractal['timestamp'],
                    'strength': current_fractal['swing_size'],
                    'distance_from_bottom': round(current_fractal['price'] - previous_fractal['price'], 2),
                    'bars_from_bottom': current_fractal['index'] - previous_fractal['index'],
                    'original_price': current_fractal['price'],  # Store original level
                    'original_index': current_fractal['index'],  # Store original creation index
                    'moved': False  # Track if creek has been moved
                }

                pending_creeks.append(creek_info)

                print(f"  -> PENDING CREEK detected at {creek_info['timestamp']}")
                print(f"     Price: ${creek_info['price']}, Strength: {creek_info['strength']}")
                print(f"     After BIG bottom at ${creek_info['trigger_bottom_price']}")
                print(f"     Creek height: ${creek_info['distance_from_bottom']}")

        # Apply trailing logic if price data is available
        if price_data is not None and len(pending_creeks) > 0:
            pending_creeks = self.apply_creek_trailing(pending_creeks, fractals_df, price_data)

        return pending_creeks

    def apply_creek_trailing(self, pending_creeks: List[Dict], fractals_df: pd.DataFrame, price_data: pd.DataFrame) -> List[Dict]:
        """
        Apply trailing logic: replace creek with new tops that form within active period

        Args:
            pending_creeks: List of detected pending creeks
            fractals_df: DataFrame with fractal data
            price_data: DataFrame with price data

        Returns:
            Updated list of pending creeks with trailing applied
        """
        updated_creeks = []

        for creek in pending_creeks:
            creek_start_index = creek['index']
            creek_original_index = creek.get('original_index', creek_start_index)  # Track original start
            creek_price = creek['price']

            # Creek is only active for 60 bars from its CURRENT position (not original)
            creek_end_index = creek_start_index + self.lookback_bars

            print(f"  -> Checking creek at ${creek_price} (current index: {creek_start_index})")
            print(f"     Active period: {creek_start_index} to {creek_end_index}")

            # Look for new tops that form AFTER the current creek position
            later_fractals = fractals_df[
                (fractals_df['index'] > creek_start_index) &
                (fractals_df['index'] <= creek_end_index)  # Only within active period
            ]

            # Look for new tops that are LOWER than current creek - trail downward only
            new_tops = later_fractals[
                (later_fractals['type'] == 'TOP') &
                (later_fractals['price'] < creek_price)  # Only trail to LOWER tops
            ]

            if len(new_tops) > 0:
                # Get the FIRST lower top within active period (not latest)
                first_lower_top = new_tops.iloc[0]

                print(f"  -> REPLACING CREEK: Delete old creek at ${creek_price}")
                print(f"     CREATE NEW CREEK at ${first_lower_top['price']} at {first_lower_top['timestamp']}")
                print(f"     New creek starts fresh 60-bar period from index {first_lower_top['index']}")

                # Create a completely new creek at the new top location
                new_creek = {
                    'index': first_lower_top['index'],
                    'timestamp': first_lower_top['timestamp'],
                    'price': first_lower_top['price'],
                    'type': 'PENDING_CREEK',
                    'trigger_bottom_price': creek['trigger_bottom_price'],  # Keep original trigger
                    'trigger_bottom_time': creek['trigger_bottom_time'],    # Keep original trigger
                    'strength': first_lower_top['swing_size'],
                    'distance_from_bottom': round(first_lower_top['price'] - creek['trigger_bottom_price'], 2),
                    'original_price': creek['original_price'],  # Keep original for reference
                    'original_index': first_lower_top['index'],  # NEW creek starts here
                    'moved': True  # Mark as moved/replaced
                }

                # Calculate bars from original bottom
                original_bottom_index = fractals_df[
                    (fractals_df['timestamp'] == creek['trigger_bottom_time'])
                ]['index'].iloc[0]
                new_creek['bars_from_bottom'] = first_lower_top['index'] - original_bottom_index

                updated_creeks.append(new_creek)
            else:
                print(f"  -> CREEK at ${creek_price} REMAINS")
                print(f"     No new tops found within active period (ends at index {creek_end_index})")
                # Keep the existing creek as-is
                creek['original_index'] = creek_original_index
                updated_creeks.append(creek)

        return updated_creeks

    def get_creek_lines(self, pending_creeks: List[Dict], max_index: int) -> List[Dict]:
        """
        Generate horizontal line data for each pending creek

        Args:
            pending_creeks: List of detected pending creeks
            max_index: Maximum index in the dataset for line extension

        Returns:
            List of line dictionaries for plotting
        """
        creek_lines = []

        for creek in pending_creeks:
            # For trailing creeks, the line starts from current position and extends 60 bars forward
            current_index = creek['index']
            end_index = min(current_index + self.lookback_bars, max_index)

            line_info = {
                'start_index': current_index,  # Line starts from where creek currently is
                'end_index': end_index,        # Line ends 60 bars after current position
                'price': creek['price'],
                'timestamp_start': creek['timestamp'],
                'type': 'PENDING_CREEK_LINE',
                'strength': creek['strength']
            }

            creek_lines.append(line_info)

        return creek_lines

def analyze_pending_creeks(fractals_df: pd.DataFrame, price_data: pd.DataFrame = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze fractal data for pending creek patterns with dynamic movement

    Args:
        fractals_df: DataFrame with fractal data
        price_data: DataFrame with price data for crossover checking

    Returns:
        Tuple of (pending_creeks, creek_lines)
    """
    detector = PendingCreekDetector()

    print(f"\n{'='*60}")
    print("PENDING CREEK ANALYSIS (WITH DYNAMIC MOVEMENT)")
    print(f"{'='*60}")
    print("Looking for small tops after big downtrends...")
    if price_data is not None:
        print("Checking for crossovers and potential creek movements...")

    pending_creeks = detector.detect_pending_creeks(fractals_df, price_data)

    if pending_creeks:
        print(f"\nFound {len(pending_creeks)} pending creek(s):")
        print(f"{'='*90}")
        print(f"{'#':<2} {'Time':<20} {'Price':<10} {'Strength':<8} {'From Bottom':<12} {'Height $':<10} {'Moved?'}")
        print(f"{'-'*90}")

        for i, creek in enumerate(pending_creeks, 1):
            moved_status = "YES" if creek.get('moved', False) else "NO"
            print(f"{i:<2} {creek['timestamp']:<20} ${creek['price']:<9} {creek['strength']:<8} "
                  f"${creek['trigger_bottom_price']:<11} +${creek['distance_from_bottom']:<9} {moved_status}")

        # Generate creek lines for plotting
        max_index = fractals_df['index'].max() if len(fractals_df) > 0 else 0
        creek_lines = detector.get_creek_lines(pending_creeks, max_index)

        print(f"\nGenerated {len(creek_lines)} pending creek lines for visualization")
        print(f"Each line extends {detector.lookback_bars} bars forward")

        # Show movement summary
        moved_count = sum(1 for creek in pending_creeks if creek.get('moved', False))
        if moved_count > 0:
            print(f"Creek movements: {moved_count} out of {len(pending_creeks)} creeks were moved down")

    else:
        print("No pending creek patterns detected")
        creek_lines = []

    print(f"{'='*90}")

    return pending_creeks, creek_lines

if __name__ == "__main__":
    print("Pending creek module - use from main.py or find_tops_and_bottoms.py")