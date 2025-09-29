import pandas as pd
import webbrowser
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ============================================================================
# MASTER CANDLE DETECTION PARAMETERS
# ============================================================================

# Default configuration parameters
DEFAULT_VOLUME_WINDOW = 60           # Number of bars for volume average calculation
DEFAULT_VOLUME_MULTIPLIER = 1.5     # Minimum volume multiplier requirement
DEFAULT_PASSING_SCORE = 50          # Minimum breakout quality score to qualify as master candle

# Factor 4 (Range and Tail Analysis) parameters
DEFAULT_RANGE_MULTIPLIER = 1.0      # Minimum range multiplier (crossover range vs avg previous range)
DEFAULT_TAIL_THRESHOLD = 0.18        # Maximum tail percentage (20% of total range)
DEFAULT_RANGE_LOOKBACK = 60         # Number of previous candles for range comparison

# ============================================================================
# FACTOR WEIGHTING SYSTEM - QUINTUPLE FACTOR ANALYSIS
# ============================================================================

FACTOR_WEIGHTS = {
    'factor_1': 0.10,   # 15% - Volumen de la vela de rotura
    'factor_2': 0.15,   # 45% - Volumen acumulativo desde el último swing (mayor peso)
    'factor_3': 0.20,   # 15% - Volumen de todo el rango
    'factor_4': 0.25,   # 10% - Análisis de rango y cola desde el último swing a la rotura
    'factor_5': 0.30    # 25% - Análisis de rango y cola de la vela de rotura del creek
}

# Validate weights sum to 100%
assert sum(FACTOR_WEIGHTS.values()) == 1.0, f"Factor weights must sum to 1.0, got {sum(FACTOR_WEIGHTS.values())}"

# ============================================================================
# VOLUME SCORING SCALE CONFIGURATION
# ============================================================================

VOLUME_SCORING_SCALE = {
    # Format: volume_ratio_threshold: (score, description)
    5.0: (10, "Exceptional volume"),
    4.0: (9,  "Very high volume"),
    3.5: (8,  "High volume"),
    3.0: (7,  "Significant volume"),
    2.5: (6,  "Moderate high volume"),
    2.0: (5,  "Minimum required volume"),
    1.5: (3,  "Low volume"),
    1.0: (1,  "Normal volume"),
    0.0: (0,  "Insufficient volume")  # Fallback for any value below 1.0
}

# ============================================================================
# CUMULATIVE VOLUME SCORING SCALE CONFIGURATION
# ============================================================================

CUMULATIVE_VOLUME_SCORING_SCALE = {
    # Format: cumulative_volume_ratio_threshold: (score, description)
    5.0: (10, "Exceptional cumulative volume"),
    4.0: (9,  "Very high cumulative volume"),
    3.5: (8,  "High cumulative volume"),
    3.0: (7,  "Significant cumulative volume"),
    2.5: (6,  "Moderate high cumulative volume"),
    2.0: (5,  "Minimum required cumulative volume"),
    1.5: (3,  "Low cumulative volume"),
    1.0: (1,  "Normal cumulative volume"),
    0.0: (0,  "Insufficient cumulative volume")  # Fallback for any value below 1.0
}

class MasterCandleDetector:
    """
    Detects master candles that cross pending creek levels with high-quality characteristics.
    A master candle must have breakout_quality >= passing_score based on volume and other criteria.
    """

    def __init__(self, volume_window: int = 60, volume_multiplier: float = 2.0, passing_score: int = 50):
        """
        Initialize the master candle detector

        Args:
            volume_window: Number of bars to calculate average volume (default 60)
            volume_multiplier: Minimum volume multiplier vs average (default 2.0x)
            passing_score: Minimum breakout quality score to qualify as master candle (default 50)
        """
        self.volume_window = volume_window
        self.volume_multiplier = volume_multiplier
        self.passing_score = passing_score

    def detect_crossovers(self, price_data: pd.DataFrame, creek_data: pd.DataFrame) -> List[Dict]:
        """
        Detect crossovers of pending creek levels using close prices

        Args:
            price_data: DataFrame with price and volume data
            creek_data: DataFrame with pending creek information

        Returns:
            List of crossover dictionaries with detailed information
        """
        crossovers = []

        if creek_data is None or len(creek_data) == 0:
            return crossovers

        print(f"\n{'='*60}")
        print("MASTER CANDLE DETECTION - CROSSOVER ANALYSIS")
        print(f"{'='*60}")
        print(f"Analyzing {len(creek_data)} pending creeks for crossovers...")

        for _, creek in creek_data.iterrows():
            creek_price = creek['price']
            creek_start_time = pd.to_datetime(creek['timestamp'])

            # Find creek start index in price data
            price_data_dates = pd.to_datetime(price_data['date'])
            creek_mask = price_data_dates >= creek_start_time

            if not creek_mask.any():
                continue

            start_idx = price_data[creek_mask].index[0]
            max_extension = min(start_idx + 60, len(price_data) - 1)  # 60 bars max

            # Look for crossover using close prices
            for i in range(start_idx, max_extension + 1):
                candle = price_data.iloc[i]

                if candle['close'] > creek_price:
                    # Found crossover!
                    crossover_info = {
                        'crossover_index': i,
                        'crossover_time': candle['date'],
                        'crossover_close': candle['close'],
                        'crossover_volume': candle['volume'],
                        'crossover_open': candle['open'],
                        'crossover_high': candle['high'],
                        'crossover_low': candle['low'],
                        'creek_price': creek_price,
                        'creek_timestamp': creek['timestamp'],
                        'creek_strength': creek['strength'],
                        'distance_above_creek': round(candle['close'] - creek_price, 2)
                    }

                    crossovers.append(crossover_info)
                    print(f"  -> CROSSOVER detected at {candle['date']}")
                    print(f"     Close: ${candle['close']:.2f} | Creek: ${creek_price:.2f} | Volume: {candle['volume']:,}")
                    break  # Only first crossover per creek

        print(f"\nFound {len(crossovers)} total crossovers")
        return crossovers

    def calculate_volume_score(self, crossover_data: Dict, price_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate volume score for a crossover based on recent volume average

        Args:
            crossover_data: Dictionary with crossover information
            price_data: DataFrame with price and volume data

        Returns:
            Tuple of (volume_score, volume_ratio)
        """
        crossover_idx = crossover_data['crossover_index']
        crossover_volume = crossover_data['crossover_volume']

        # Calculate average volume for the window before crossover
        start_idx = max(0, crossover_idx - self.volume_window)
        volume_window_data = price_data.iloc[start_idx:crossover_idx]

        if len(volume_window_data) == 0:
            return 0, 0

        avg_volume = volume_window_data['volume'].mean()
        volume_ratio = crossover_volume / avg_volume if avg_volume > 0 else 0

        # Volume scoring based on configurable scale
        score = 0
        for threshold in sorted(VOLUME_SCORING_SCALE.keys(), reverse=True):
            if volume_ratio >= threshold:
                score = VOLUME_SCORING_SCALE[threshold][0]
                break

        return score, volume_ratio

    def find_last_bottom_before_crossover(self, crossover_data: Dict, price_data: pd.DataFrame) -> Tuple[int, Optional[Dict]]:
        """
        Find the last bottom (small or big) before the crossover candle

        Args:
            crossover_data: Dictionary with crossover information
            price_data: DataFrame with price and volume data

        Returns:
            Tuple of (bottom_index, bottom_info_dict)
        """
        crossover_idx = crossover_data['crossover_index']

        # Load fractal data to find bottoms
        try:
            import os
            fractal_file = 'fractals_2023_03_01_zigzag_0.1.csv'
            fractal_path = os.path.join('outputs', fractal_file)

            if os.path.exists(fractal_path):
                df_fractals = pd.read_csv(fractal_path)
                df_fractals['timestamp'] = pd.to_datetime(df_fractals['timestamp'])

                # Filter only bottoms
                bottoms = df_fractals[df_fractals['type'] == 'BOTTOM'].copy()

                # Find crossover time
                crossover_time = price_data.iloc[crossover_idx]['date']

                # Find bottoms before crossover time
                bottoms_before = bottoms[bottoms['timestamp'] <= crossover_time]

                if len(bottoms_before) > 0:
                    # Get the most recent bottom
                    last_bottom = bottoms_before.iloc[-1]

                    # Find the index of this bottom in the price data
                    bottom_mask = price_data['date'] >= last_bottom['timestamp']
                    if bottom_mask.any():
                        bottom_idx = price_data[bottom_mask].index[0]
                        return bottom_idx, last_bottom.to_dict()

        except Exception as e:
            print(f"Error finding last bottom: {e}")

        return 0, None

    def calculate_cumulative_volume_score(self, crossover_data: Dict, price_data: pd.DataFrame) -> Tuple[float, float, int]:
        """
        Calculate cumulative volume score from last bottom to crossover

        Args:
            crossover_data: Dictionary with crossover information
            price_data: DataFrame with price and volume data

        Returns:
            Tuple of (cumulative_volume_score, cumulative_volume_ratio, candles_from_bottom)
        """
        crossover_idx = crossover_data['crossover_index']

        # Find last bottom before crossover
        bottom_idx, bottom_info = self.find_last_bottom_before_crossover(crossover_data, price_data)

        if bottom_info is None:
            return 0, 0, 0

        # Calculate cumulative volume from bottom to crossover
        volume_data = price_data.iloc[bottom_idx:crossover_idx + 1]
        cumulative_volume = volume_data['volume'].sum()
        candles_from_bottom = len(volume_data)

        if candles_from_bottom == 0:
            return 0, 0, 0

        # Calculate average volume per candle since bottom
        avg_volume_since_bottom = cumulative_volume / candles_from_bottom

        # Calculate average volume for recent n candles (same window as original factor)
        start_idx = max(0, crossover_idx - self.volume_window)
        volume_window_data = price_data.iloc[start_idx:crossover_idx]

        if len(volume_window_data) == 0:
            return 0, 0, candles_from_bottom

        avg_volume_recent = volume_window_data['volume'].mean()

        # Calculate ratio
        cumulative_volume_ratio = avg_volume_since_bottom / avg_volume_recent if avg_volume_recent > 0 else 0

        # Score based on cumulative volume scale
        score = 0
        for threshold in sorted(CUMULATIVE_VOLUME_SCORING_SCALE.keys(), reverse=True):
            if cumulative_volume_ratio >= threshold:
                score = CUMULATIVE_VOLUME_SCORING_SCALE[threshold][0]
                break

        return score, cumulative_volume_ratio, candles_from_bottom

    def calculate_longterm_volume_score(self, crossover_data: Dict, price_data: pd.DataFrame, fractals_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Factor 3: Long-term Volume Analysis
        Compare average volume from big bottom until crossover vs baseline 100-bar average before big bottom

        Args:
            crossover_data: Crossover information
            price_data: Price and volume data
            fractals_df: Fractal data to find big bottoms

        Returns:
            Tuple of (score, ratio)
        """
        crossover_idx = crossover_data['crossover_index']
        crossover_time = pd.to_datetime(crossover_data['crossover_time'])

        # Find the last big bottom before the crossover
        big_bottoms = fractals_df[
            (fractals_df['type'] == 'BOTTOM') &
            (fractals_df['swing_size'] == 'big') &
            (pd.to_datetime(fractals_df['timestamp']) < crossover_time)
        ].sort_values('timestamp')

        if len(big_bottoms) == 0:
            return 1.0, 1.0  # Minimum score if no big bottom found

        # Get the last big bottom
        last_big_bottom = big_bottoms.iloc[-1]
        big_bottom_idx = last_big_bottom['index']

        # Check if we have enough data for baseline calculation
        baseline_start_idx = max(0, big_bottom_idx - 100)
        if big_bottom_idx < 100:
            return 1.0, 1.0  # Minimum score if not enough baseline data

        # Calculate baseline average volume (100 bars before big bottom)
        baseline_volume_data = price_data.iloc[baseline_start_idx:big_bottom_idx]
        baseline_avg_volume = baseline_volume_data['volume'].mean()

        # Calculate period from big bottom until crossover
        period_volume_data = price_data.iloc[big_bottom_idx:crossover_idx + 1]
        if len(period_volume_data) == 0:
            return 1.0, 1.0  # Minimum score if no period data

        # Calculate average volume from big bottom to crossover
        period_avg_volume = period_volume_data['volume'].mean()

        # Calculate ratio
        if baseline_avg_volume == 0:
            longterm_ratio = 1.0
        else:
            longterm_ratio = period_avg_volume / baseline_avg_volume

        # Score based on ratio (same scale as other factors)
        if longterm_ratio >= 4.0:
            longterm_score = 10
        elif longterm_ratio >= 3.5:
            longterm_score = 9
        elif longterm_ratio >= 3.0:
            longterm_score = 8
        elif longterm_ratio >= 2.5:
            longterm_score = 7
        elif longterm_ratio >= 2.0:
            longterm_score = 6
        elif longterm_ratio >= 1.8:
            longterm_score = 5
        elif longterm_ratio >= 1.5:
            longterm_score = 4
        elif longterm_ratio >= 1.3:
            longterm_score = 3
        elif longterm_ratio >= 1.1:
            longterm_score = 2
        else:
            longterm_score = 1

        return longterm_score, longterm_ratio

    def calculate_range_and_tail_score(self, crossover_data: Dict, price_data: pd.DataFrame, candles_from_bottom: int = 0, range_multiplier: float = DEFAULT_RANGE_MULTIPLIER, tail_threshold: float = DEFAULT_TAIL_THRESHOLD, lookback_periods: int = DEFAULT_RANGE_LOOKBACK) -> Tuple[float, float, float, float, List]:
        """
        Factor 4: Range and Tail Analysis
        Evaluate if ANY candle from last bottom to crossover (excluding crossover) has:
        1. Range (high-low) bigger than average range of previous N candles by multiplier R
        2. Small tail above (high-close) less than T% of total range

        Args:
            crossover_data: Dictionary with crossover information
            price_data: DataFrame with price data
            range_multiplier: Multiplier for range comparison (default 1.0)
            tail_threshold: Maximum tail percentage (default 0.2 = 20%)
            lookback_periods: Number of previous candles to compare (default 60)

        Returns:
            Tuple of (range_score, best_range_ratio, best_tail_percentage, best_range, qualifying_candles)
        """
        crossover_time = pd.to_datetime(crossover_data.get('crossover_time'))

        # Find crossover candle index
        crossover_idx = None
        for i, row in price_data.iterrows():
            if pd.to_datetime(row['date']) == crossover_time:
                crossover_idx = i
                break

        if crossover_idx is None:
            print(f"Factor 4: Crossover candle not found for time {crossover_time}")
            return 1, 0.0, 0.0, 0.0, []

        # Find the range from last bottom to crossover (excluding crossover)
        bottom_idx = crossover_idx - candles_from_bottom
        if bottom_idx < 0:
            bottom_idx = 0

        # Get candles from bottom to crossover (excluding crossover)
        swing_candles = price_data.iloc[bottom_idx:crossover_idx]  # Excludes crossover candle

        if len(swing_candles) == 0:
            print(f"Factor 4: No swing candles found between bottom and crossover")
            return 1, 0.0, 0.0, 0.0, []

        print(f"Factor 4 Debug:")
        print(f"  Crossover time: {crossover_time}")
        print(f"  Crossover index: {crossover_idx}")
        print(f"  Candles from bottom: {candles_from_bottom}")
        print(f"  Bottom index: {bottom_idx}")
        print(f"  Swing candles range: {bottom_idx} to {crossover_idx}")
        print(f"  Analyzing {len(swing_candles)} candles from bottom to crossover (excluding crossover)")
        print(f"  Range multiplier threshold: {range_multiplier}x")
        print(f"  Tail threshold: {tail_threshold:.1%}")

        # Calculate average range for comparison (use lookback_periods before bottom)
        reference_start_idx = max(0, bottom_idx - lookback_periods)
        if reference_start_idx >= bottom_idx:
            print(f"Factor 4: Not enough historical data for range comparison")
            return 1, 0.0, 0.0, 0.0, []

        reference_candles = price_data.iloc[reference_start_idx:bottom_idx]
        if len(reference_candles) == 0:
            print(f"Factor 4: No reference candles found for comparison")
            return 1, 0.0, 0.0, 0.0, []

        reference_ranges = reference_candles['high'] - reference_candles['low']
        avg_reference_range = reference_ranges.mean()
        print(f"  Average reference range ({len(reference_candles)} candles before bottom): {avg_reference_range:.2f}")

        # Analyze each candle in the swing to find qualifying candles
        qualifying_candles = []
        best_range_ratio = 0.0
        best_tail_percentage = 1.0
        best_range = 0.0

        for i, (idx, candle) in enumerate(swing_candles.iterrows()):
            candle_high = candle['high']
            candle_low = candle['low']
            candle_close = candle['close']
            candle_range = candle_high - candle_low
            candle_time = pd.to_datetime(candle['date'])

            # Calculate tail above (high - close)
            tail_above = candle_high - candle_close
            tail_percentage = tail_above / candle_range if candle_range > 0 else 0.0

            # Calculate range ratio
            range_ratio = candle_range / avg_reference_range if avg_reference_range > 0 else 0.0

            # Check if this candle meets criteria
            range_meets_criteria = range_ratio >= range_multiplier
            tail_meets_criteria = tail_percentage <= tail_threshold

            if range_meets_criteria and tail_meets_criteria:
                qualifying_candles.append({
                    'time': candle_time,
                    'high': candle_high,
                    'low': candle_low,
                    'close': candle_close,
                    'range': candle_range,
                    'range_ratio': range_ratio,
                    'tail_percentage': tail_percentage,
                    'index': idx
                })
                print(f"    QUALIFYING CANDLE {i+1}: {candle_time} - Range: {range_ratio:.2f}x, Tail: {tail_percentage:.1%}")

                # Track best ratios for reporting
                if range_ratio > best_range_ratio:
                    best_range_ratio = range_ratio
                if tail_percentage < best_tail_percentage:
                    best_tail_percentage = tail_percentage
                if candle_range > best_range:
                    best_range = candle_range
            else:
                range_check = 'PASS' if range_meets_criteria else 'FAIL'
                tail_check = 'PASS' if tail_meets_criteria else 'FAIL'
                print(f"    Candle {i+1}: {candle_time} - Range: {range_ratio:.2f}x ({range_check}), Tail: {tail_percentage:.1%} ({tail_check})")

        # Scoring based on number and quality of qualifying candles
        range_score = 1  # Default minimum score
        num_qualifying = len(qualifying_candles)

        print(f"  Found {num_qualifying} qualifying candles")

        if num_qualifying > 0:
            # Score based on best ratios found
            if best_range_ratio >= 4.0 and best_tail_percentage <= 0.05:  # Exceptional
                range_score = 10
            elif best_range_ratio >= 3.5 and best_tail_percentage <= 0.10:  # Excellent
                range_score = 9
            elif best_range_ratio >= 3.0 and best_tail_percentage <= 0.15:  # Very Strong
                range_score = 8
            elif best_range_ratio >= 2.5 and best_tail_percentage <= 0.18:  # Strong
                range_score = 7
            elif best_range_ratio >= 2.0 and best_tail_percentage <= 0.20:  # Good
                range_score = 6
            else:  # Meets minimum criteria
                range_score = 5

            # Bonus points for multiple qualifying candles
            if num_qualifying >= 3:
                range_score = min(10, range_score + 2)
            elif num_qualifying >= 2:
                range_score = min(10, range_score + 1)

        print(f"  Best range ratio: {best_range_ratio:.2f}x")
        print(f"  Best tail percentage: {best_tail_percentage:.1%}")
        print(f"  Factor 4 Score: {range_score}/10")

        return range_score, best_range_ratio, best_tail_percentage, best_range, qualifying_candles

    def calculate_crossover_range_and_tail_score(self, crossover_data: Dict, price_data: pd.DataFrame, range_multiplier: float = DEFAULT_RANGE_MULTIPLIER, tail_threshold: float = DEFAULT_TAIL_THRESHOLD, lookback_periods: int = DEFAULT_RANGE_LOOKBACK) -> Tuple[float, float, float, float]:
        """
        Factor 5: Crossover Candle Range and Tail Analysis
        Evaluate if the crossover candle itself has:
        1. Range (high-low) bigger than average range of previous N candles by multiplier R
        2. Small tail above (high-close) less than T% of total range

        Args:
            crossover_data: Dictionary with crossover information
            price_data: DataFrame with price data
            range_multiplier: Multiplier for range comparison (default 1.0)
            tail_threshold: Maximum tail percentage (default 0.2 = 20%)
            lookback_periods: Number of previous candles to compare (default 60)

        Returns:
            Tuple of (range_score, range_ratio, tail_percentage, crossover_range)
        """
        crossover_time = pd.to_datetime(crossover_data.get('crossover_time'))

        # Find crossover candle index
        crossover_idx = None
        for i, row in price_data.iterrows():
            if pd.to_datetime(row['date']) == crossover_time:
                crossover_idx = i
                break

        if crossover_idx is None:
            print(f"Factor 5: Crossover candle not found for time {crossover_time}")
            return 1, 0.0, 0.0, 0.0

        # Get crossover candle OHLC
        crossover_candle = price_data.iloc[crossover_idx]
        crossover_high = crossover_candle['high']
        crossover_low = crossover_candle['low']
        crossover_close = crossover_candle['close']
        crossover_range = crossover_high - crossover_low

        # Calculate tail above (high - close)
        tail_above = crossover_high - crossover_close
        tail_percentage = tail_above / crossover_range if crossover_range > 0 else 0.0

        # Get previous N candles for range comparison
        start_idx = max(0, crossover_idx - lookback_periods)
        if start_idx >= crossover_idx:
            print(f"Factor 5: Not enough historical data for range comparison")
            return 1, 0.0, tail_percentage, crossover_range

        # Calculate average range of previous N candles
        previous_candles = price_data.iloc[start_idx:crossover_idx]
        if len(previous_candles) == 0:
            print(f"Factor 5: No previous candles found for comparison")
            return 1, 0.0, tail_percentage, crossover_range

        previous_ranges = previous_candles['high'] - previous_candles['low']
        avg_previous_range = previous_ranges.mean()

        # Calculate range ratio
        range_ratio = crossover_range / avg_previous_range if avg_previous_range > 0 else 0.0

        print(f"Factor 5 Debug:")
        print(f"  Crossover candle range: {crossover_range:.2f}")
        print(f"  Avg previous range ({lookback_periods} candles): {avg_previous_range:.2f}")
        print(f"  Range ratio: {range_ratio:.2f}x")
        print(f"  Tail above: {tail_above:.2f} ({tail_percentage:.1%} of range)")
        print(f"  Range multiplier threshold: {range_multiplier}x")
        print(f"  Tail threshold: {tail_threshold:.1%}")

        # Scoring logic based on both range and tail criteria
        range_score = 1  # Default minimum score

        # Range must be bigger than previous candles by multiplier
        range_meets_criteria = range_ratio >= range_multiplier

        # Tail must be smaller than threshold percentage
        tail_meets_criteria = tail_percentage <= tail_threshold

        if range_meets_criteria and tail_meets_criteria:
            # Both criteria met - score based on how much they exceed thresholds
            if range_ratio >= 4.0 and tail_percentage <= 0.05:  # Exceptional: 4x range, 5% tail
                range_score = 10
            elif range_ratio >= 3.5 and tail_percentage <= 0.10:  # Excellent: 3.5x range, 10% tail
                range_score = 9
            elif range_ratio >= 3.0 and tail_percentage <= 0.15:  # Very Strong: 3x range, 15% tail
                range_score = 8
            elif range_ratio >= 2.5 and tail_percentage <= 0.18:  # Strong: 2.5x range, 18% tail
                range_score = 7
            elif range_ratio >= range_multiplier and tail_percentage <= tail_threshold:  # Good: meets minimum criteria
                range_score = 6
            else:
                range_score = 5  # Marginal
        elif range_meets_criteria:
            # Only range criteria met
            range_score = 3
        elif tail_meets_criteria:
            # Only tail criteria met
            range_score = 2
        else:
            # Neither criteria met
            range_score = 1

        print(f"  Range meets criteria (>= {range_multiplier}x): {range_meets_criteria}")
        print(f"  Tail meets criteria (<= {tail_threshold:.1%}): {tail_meets_criteria}")
        print(f"  Factor 5 Score: {range_score}/10")

        return range_score, range_ratio, tail_percentage, crossover_range

    def evaluate_mastercandle(self, crossover_data: Dict, price_data: pd.DataFrame, fractals_data: pd.DataFrame = None) -> Dict:
        """
        Evaluate if a crossover qualifies as a master candle based on scoring system

        Args:
            crossover_data: Dictionary with crossover information
            price_data: DataFrame with price and volume data

        Returns:
            Enhanced crossover dictionary with scoring information
        """
        # Calculate volume score (Factor 1)
        volume_score, volume_ratio = self.calculate_volume_score(crossover_data, price_data)

        # Calculate cumulative volume score (Factor 2)
        cumulative_volume_score, cumulative_volume_ratio, candles_from_bottom = self.calculate_cumulative_volume_score(crossover_data, price_data)

        # Calculate long-term volume score (Factor 3)
        if fractals_data is not None:
            longterm_volume_score, longterm_volume_ratio = self.calculate_longterm_volume_score(crossover_data, price_data, fractals_data)
        else:
            longterm_volume_score, longterm_volume_ratio = 0, 0.0

        # Calculate range and tail score (Factor 4)
        range_score, range_ratio, tail_percentage, crossover_range, qualifying_candles = self.calculate_range_and_tail_score(crossover_data, price_data, candles_from_bottom)

        # Calculate crossover range and tail score (Factor 5)
        crossover_range_score, crossover_range_ratio, crossover_tail_percentage, crossover_candle_range = self.calculate_crossover_range_and_tail_score(crossover_data, price_data)

        # Breakout quality score (sum of all five factors)
        breakout_quality = volume_score + cumulative_volume_score + longterm_volume_score + range_score + crossover_range_score

        # Determine if it's a master candle
        is_mastercandle = breakout_quality >= self.passing_score

        # Enhanced crossover data
        enhanced_data = crossover_data.copy()
        enhanced_data.update({
            'volume_score': volume_score,
            'volume_ratio': round(volume_ratio, 2),
            'cumulative_volume_score': cumulative_volume_score,
            'cumulative_volume_ratio': round(cumulative_volume_ratio, 2),
            'longterm_volume_score': longterm_volume_score,
            'longterm_volume_ratio': round(longterm_volume_ratio, 2),
            'range_score': range_score,
            'range_ratio': round(range_ratio, 2),
            'tail_percentage': round(tail_percentage, 3),
            'crossover_range': round(crossover_range, 2),
            'crossover_range_score': crossover_range_score,
            'crossover_range_ratio': round(crossover_range_ratio, 2),
            'crossover_tail_percentage': round(crossover_tail_percentage, 3),
            'crossover_candle_range': round(crossover_candle_range, 2),
            'crossover_high': crossover_data.get('crossover_high', crossover_data.get('crossover_close', 0)),
            'qualifying_candles': qualifying_candles,
            'candles_from_bottom': candles_from_bottom,
            'breakout_quality': breakout_quality,
            'is_mastercandle': is_mastercandle,
            'meets_volume_requirement': volume_ratio >= self.volume_multiplier
        })

        return enhanced_data

    def generate_html_report(self, evaluated_crossovers: List[Dict], data_date: str = None) -> str:
        """
        Generate HTML report with master candle analysis results

        Args:
            evaluated_crossovers: List of evaluated crossover dictionaries
            data_date: Optional date string from the data for filename and title

        Returns:
            Path to generated HTML file
        """
        # Separate master candles from regular crossovers
        master_candles = [c for c in evaluated_crossovers if c['is_mastercandle']]

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Master Candle Detection Report{' - ' + data_date.split(' ')[0] if data_date else ''}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .summary-item {{ display: inline-block; margin: 10px 20px; }}
        .summary-number {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .summary-label {{ font-size: 14px; color: #7f8c8d; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; font-size: 12px; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e8f4f8; }}
        .mastercandle {{ background-color: #d5ead5 !important; font-weight: bold; }}
        .score-excellent {{ color: #27ae60; font-weight: bold; }}
        .score-good {{ color: #f39c12; font-weight: bold; }}
        .score-poor {{ color: #e74c3c; font-weight: bold; }}
        .volume-high {{ background-color: #d4edda; }}
        .volume-medium {{ background-color: #fff3cd; }}
        .volume-low {{ background-color: #f8d7da; }}
        .criteria {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Master Candle Detection Report{' - ' + data_date.split(' ')[0] if data_date else ''}</h1>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-number">{len(evaluated_crossovers)}</div>
                <div class="summary-label">Total Crossovers</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{len(master_candles)}</div>
                <div class="summary-label">Master Candles</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{(len(master_candles)/len(evaluated_crossovers)*100):.1f}%</div>
                <div class="summary-label">Success Rate</div>
            </div>
        </div>

        <div class="criteria">
            <h3>Master Candle Detection System - Quintuple Factor Analysis</h3>

            <div style="margin-bottom: 20px; padding: 15px; background-color: #e8f4f8; border-radius: 5px;">
                <h4 style="color: #2c3e50; margin-top: 0;">📊 Factor 1: Individual Volume Analysis</h4>
                <p><strong>What it measures:</strong> The volume of the crossover candle compared to recent average volume</p>
                <p><strong>Calculation:</strong> Crossover Volume ÷ Average Volume (last {self.volume_window} bars)</p>
                <p><strong>Purpose:</strong> Identifies if the breakout candle has significantly higher volume than normal, indicating strong buying pressure</p>
                <p><strong>Threshold:</strong> Minimum {self.volume_multiplier}x average volume required for significance</p>
                <p><strong>Example:</strong> If recent 60-bar average volume = 3,000 and crossover volume = 8,231, then ratio = 2.85x</p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background-color: #fff3e0; border-radius: 5px;">
                <h4 style="color: #2c3e50; margin-top: 0;">📈 Factor 2: Cumulative Volume Analysis</h4>
                <p><strong>What it measures:</strong> Average volume from the last bottom to crossover vs recent baseline volume</p>
                <p><strong>Calculation:</strong> Average Volume (since last bottom) ÷ Average Volume (last {self.volume_window} bars)</p>
                <p><strong>Purpose:</strong> Detects if there's been sustained volume accumulation building up to the breakout</p>
                <p><strong>Logic:</strong> Strong moves often show gradual volume buildup before the actual breakout</p>
                <p><strong>Example:</strong> If volume has been building since the bottom, this ratio will be > 1.0</p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background-color: #ffeaa7; border-radius: 5px;">
                <h4 style="color: #2c3e50; margin-top: 0;">📊 Factor 3: Long-term Volume Analysis</h4>
                <p><strong>What it measures:</strong> Average volume from big bottom (purple dot) until crossover vs 100-bar baseline before big bottom</p>
                <p><strong>Calculation:</strong> Average Volume (big bottom to crossover) ÷ Average Volume (100 bars before big bottom)</p>
                <p><strong>Purpose:</strong> Identifies sustained institutional accumulation over extended periods</p>
                <p><strong>Logic:</strong> Major moves often show long-term volume increases starting from significant bottoms</p>
                <p><strong>Example:</strong> If volume doubled since the big bottom compared to historical baseline, ratio = 2.0x</p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background-color: #f0f4ff; border-radius: 5px;">
                <h4 style="color: #2c3e50; margin-top: 0;">🎯 Factor 4: Swing Range and Tail Analysis</h4>
                <p><strong>What it measures:</strong> ANY candle from last bottom to crossover (excluding crossover) with big range and small tail</p>
                <p><strong>Range Calculation:</strong> Each Swing Candle Range (high-low) ÷ Average Range (60 candles before bottom)</p>
                <p><strong>Tail Calculation:</strong> Upper Tail (high-close) ÷ Total Range (high-low) for each swing candle</p>
                <p><strong>Purpose:</strong> Identifies accumulation/momentum building during the swing before breakout</p>
                <p><strong>Criteria:</strong> Any swing candle with Range ≥ 1.0x average AND Upper tail ≤ 20% of total range</p>
                <p><strong>Example:</strong> If any candle in the swing has strong range and small rejection, it indicates building momentum</p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background-color: #fff0f5; border-radius: 5px;">
                <h4 style="color: #2c3e50; margin-top: 0;">💥 Factor 5: Crossover Candle Analysis</h4>
                <p><strong>What it measures:</strong> The actual crossover candle characteristics - big range with small upper tail</p>
                <p><strong>Range Calculation:</strong> Crossover Range (high-low) ÷ Average Range (previous 60 candles)</p>
                <p><strong>Tail Calculation:</strong> Upper Tail (high-close) ÷ Total Range (high-low)</p>
                <p><strong>Purpose:</strong> Identifies powerful breakout candles with strong momentum and minimal rejection</p>
                <p><strong>Criteria:</strong> Range ≥ 1.0x average AND Upper tail ≤ 20% of total range</p>
                <p><strong>Example:</strong> If crossover candle range = 5.0 vs avg 2.0 (2.5x) with tail = 0.3 (6%), scores highly</p>
            </div>

            <div style="margin-bottom: 20px; padding: 15px; background-color: #f0f8f0; border-radius: 5px;">
                <h4 style="color: #2c3e50; margin-top: 0;">🎯 Scoring Methodology</h4>
                <p><strong>Each Factor Scores 1-10 Points:</strong></p>
                <ul style="margin: 10px 0;">
                    <li><strong>Ratio ≥ 4.0x:</strong> 10 points (Exceptional)</li>
                    <li><strong>Ratio ≥ 3.5x:</strong> 9 points (Excellent)</li>
                    <li><strong>Ratio ≥ 3.0x:</strong> 8 points (Very Strong)</li>
                    <li><strong>Ratio ≥ 2.5x:</strong> 6-7 points (Strong)</li>
                    <li><strong>Ratio ≥ 2.0x:</strong> 4-5 points (Good)</li>
                    <li><strong>Ratio ≥ 1.5x:</strong> 2-3 points (Moderate)</li>
                    <li><strong>Ratio < 1.5x:</strong> 1 point (Weak)</li>
                </ul>
                <p><strong>Total Score:</strong> Factor 1 + Factor 2 + Factor 3 + Factor 4 + Factor 5 (Maximum: 50 points)</p>
                <p><strong>Master Candle Threshold:</strong> ≥ {self.passing_score} points total</p>
            </div>

            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #3498db;">
                <h4 style="color: #2c3e50; margin-top: 0;">💡 Why This Works</h4>
                <p><strong>Factor 1</strong> catches explosive breakouts with immediate high volume</p>
                <p><strong>Factor 2</strong> catches setups where smart money has been accumulating volume</p>
                <p><strong>Factor 3</strong> identifies long-term institutional accumulation from major bottoms</p>
                <p><strong>Factor 4</strong> detects momentum building during the swing before breakout</p>
                <p><strong>Factor 5</strong> confirms powerful breakout candles with large range and minimal rejection</p>
                <p><strong>Combined:</strong> Creates a comprehensive system that identifies explosive, methodical, institutional, momentum-building, and technically strong breakouts</p>
            </div>
        </div>

        <h2>Factor Weights Configuration</h2>
        <table>
            <tr>
                <th>Factor</th>
                <th>Weight</th>
                <th>Description (Spanish)</th>
                <th>Description (English)</th>
            </tr>
            <tr>
                <td><strong>Factor 1</strong></td>
                <td><strong>{FACTOR_WEIGHTS['factor_1']:.0%}</strong></td>
                <td>Volumen de la vela de rotura</td>
                <td>Individual Volume Analysis</td>
            </tr>
            <tr style="background-color: #fff3cd;">
                <td><strong>Factor 2</strong></td>
                <td><strong>{FACTOR_WEIGHTS['factor_2']:.0%}</strong></td>
                <td>Volumen acumulativo desde el último swing</td>
                <td>Cumulative Volume Analysis (highest weight)</td>
            </tr>
            <tr>
                <td><strong>Factor 3</strong></td>
                <td><strong>{FACTOR_WEIGHTS['factor_3']:.0%}</strong></td>
                <td>Volumen de todo el rango</td>
                <td>Long-term Volume Analysis</td>
            </tr>
            <tr>
                <td><strong>Factor 4</strong></td>
                <td><strong>{FACTOR_WEIGHTS['factor_4']:.0%}</strong></td>
                <td>Análisis de rango y cola desde el último swing a la rotura</td>
                <td>Swing Range and Tail Analysis</td>
            </tr>
            <tr>
                <td><strong>Factor 5</strong></td>
                <td><strong>{FACTOR_WEIGHTS['factor_5']:.0%}</strong></td>
                <td>Análisis de rango y cola de la vela de rotura del creek</td>
                <td>Crossover Candle Range and Tail Analysis</td>
            </tr>
        </table>
        <p style="margin: 10px 0; font-size: 14px; color: #666;">
            <strong>Note:</strong> Each factor score (1-10) is multiplied by its weight to get the final contribution to the total score.
            Total possible score: {sum(FACTOR_WEIGHTS.values()) * 10:.1f} points
        </p>

        <h2>Master Candles</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Close Price</th>
                <th>Creek Level</th>
                <th>Volume</th>
                <th>Factor 1<br/>Vol Ratio</th>
                <th>Factor 1<br/>Score</th>
                <th>Factor 2<br/>Cum Vol Ratio</th>
                <th>Factor 2<br/>Score</th>
                <th>Factor 3<br/>LT Vol Ratio</th>
                <th>Factor 3<br/>Score</th>
                <th>Factor 4<br/>Range Ratio</th>
                <th>Factor 4<br/>Tail %</th>
                <th>Factor 4<br/>Score</th>
                <th>Factor 5<br/>Cross Range Ratio</th>
                <th>Factor 5<br/>Cross Tail %</th>
                <th>Factor 5<br/>Score</th>
                <th>Candles from Bottom</th>
                <th>Breakout Quality<br/>(F1 + F2 + F3 + F4 + F5)</th>
                <th>Status</th>
            </tr>
"""

        # Add master candles to table
        if master_candles:
            for candle in master_candles:
                volume_class = ("volume-high" if candle['volume_ratio'] >= 3.0 else
                              "volume-medium" if candle['volume_ratio'] >= 2.0 else "volume-low")
                cum_volume_class = ("volume-high" if candle['cumulative_volume_ratio'] >= 3.0 else
                                  "volume-medium" if candle['cumulative_volume_ratio'] >= 2.0 else "volume-low")
                score_class = ("score-excellent" if candle['breakout_quality'] >= 50 else
                             "score-good" if candle['breakout_quality'] >= 40 else "score-poor")
                vol_score_class = ("score-excellent" if candle['volume_score'] >= 8 else
                                 "score-good" if candle['volume_score'] >= 6 else "score-poor")
                cum_vol_score_class = ("score-excellent" if candle['cumulative_volume_score'] >= 8 else
                                     "score-good" if candle['cumulative_volume_score'] >= 6 else "score-poor")
                longterm_volume_class = ("volume-high" if candle['longterm_volume_ratio'] >= 3.0 else
                                       "volume-medium" if candle['longterm_volume_ratio'] >= 2.0 else "volume-low")
                longterm_vol_score_class = ("score-excellent" if candle['longterm_volume_score'] >= 8 else
                                          "score-good" if candle['longterm_volume_score'] >= 6 else "score-poor")
                range_class = ("volume-high" if candle['range_ratio'] >= 3.0 else
                             "volume-medium" if candle['range_ratio'] >= 2.0 else "volume-low")
                range_score_class = ("score-excellent" if candle['range_score'] >= 8 else
                                   "score-good" if candle['range_score'] >= 6 else "score-poor")

                html_content += f"""
            <tr class="mastercandle">
                <td>{candle['crossover_time']}</td>
                <td>${candle['crossover_close']:.2f}</td>
                <td>${candle['creek_price']:.2f}</td>
                <td class="{volume_class}">{candle['crossover_volume']:,}</td>
                <td class="{volume_class}">{candle['volume_ratio']:.2f}x</td>
                <td class="{vol_score_class}">{candle['volume_score']}</td>
                <td class="{cum_volume_class}">{candle['cumulative_volume_ratio']:.2f}x</td>
                <td class="{cum_vol_score_class}">{candle['cumulative_volume_score']}</td>
                <td class="{longterm_volume_class}">{candle['longterm_volume_ratio']:.2f}x</td>
                <td class="{longterm_vol_score_class}">{candle['longterm_volume_score']}</td>
                <td class="{range_class}">{candle['range_ratio']:.2f}x</td>
                <td class="{range_class}">{candle['tail_percentage']:.1%}</td>
                <td class="{range_score_class}">{candle['range_score']}</td>
                <td class="{range_class}">{candle['crossover_range_ratio']:.2f}x</td>
                <td class="{range_class}">{candle['crossover_tail_percentage']:.1%}</td>
                <td class="{range_score_class}">{candle['crossover_range_score']}</td>
                <td>{candle['candles_from_bottom']}</td>
                <td class="{score_class}">{candle['breakout_quality']}</td>
                <td>[MASTER CANDLE]</td>
            </tr>
"""
        else:
            html_content += """
            <tr>
                <td colspan="11" style="text-align: center; color: #7f8c8d; font-style: italic;">
                    No master candles detected with current criteria
                </td>
            </tr>
"""

        html_content += """
        </table>

        <h2>All Crossovers Analysis</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Close Price</th>
                <th>Creek Level</th>
                <th>Volume</th>
                <th>Factor 1<br/>Vol Ratio</th>
                <th>Factor 1<br/>Score</th>
                <th>Factor 2<br/>Cum Vol Ratio</th>
                <th>Factor 2<br/>Score</th>
                <th>Factor 3<br/>LT Vol Ratio</th>
                <th>Factor 3<br/>Score</th>
                <th>Factor 4<br/>Range Ratio</th>
                <th>Factor 4<br/>Tail %</th>
                <th>Factor 4<br/>Score</th>
                <th>Factor 5<br/>Cross Range Ratio</th>
                <th>Factor 5<br/>Cross Tail %</th>
                <th>Factor 5<br/>Score</th>
                <th>Candles from Bottom</th>
                <th>Breakout Quality<br/>(F1 + F2 + F3 + F4 + F5)</th>
                <th>Status</th>
            </tr>
"""

        # Add all crossovers to table
        for candle in evaluated_crossovers:
            row_class = "mastercandle" if candle['is_mastercandle'] else ""
            volume_class = ("volume-high" if candle['volume_ratio'] >= 3.0 else
                          "volume-medium" if candle['volume_ratio'] >= 2.0 else "volume-low")
            cum_volume_class = ("volume-high" if candle['cumulative_volume_ratio'] >= 3.0 else
                              "volume-medium" if candle['cumulative_volume_ratio'] >= 2.0 else "volume-low")
            longterm_volume_class = ("volume-high" if candle['longterm_volume_ratio'] >= 3.0 else
                                   "volume-medium" if candle['longterm_volume_ratio'] >= 2.0 else "volume-low")
            vol_score_class = ("score-excellent" if candle['volume_score'] >= 8 else
                             "score-good" if candle['volume_score'] >= 6 else "score-poor")
            cum_vol_score_class = ("score-excellent" if candle['cumulative_volume_score'] >= 8 else
                                 "score-good" if candle['cumulative_volume_score'] >= 6 else "score-poor")
            longterm_vol_score_class = ("score-excellent" if candle['longterm_volume_score'] >= 8 else
                                      "score-good" if candle['longterm_volume_score'] >= 6 else "score-poor")
            range_class = ("volume-high" if candle['range_ratio'] >= 3.0 else
                         "volume-medium" if candle['range_ratio'] >= 2.0 else "volume-low")
            range_score_class = ("score-excellent" if candle['range_score'] >= 8 else
                               "score-good" if candle['range_score'] >= 6 else "score-poor")
            score_class = ("score-excellent" if candle['breakout_quality'] >= 45 else
                         "score-good" if candle['breakout_quality'] >= 35 else "score-poor")
            status = "[MASTER CANDLE]" if candle['is_mastercandle'] else "[Regular Crossover]"

            html_content += f"""
            <tr class="{row_class}">
                <td>{candle['crossover_time']}</td>
                <td>${candle['crossover_close']:.2f}</td>
                <td>${candle['creek_price']:.2f}</td>
                <td class="{volume_class}">{candle['crossover_volume']:,}</td>
                <td class="{volume_class}">{candle['volume_ratio']:.2f}x</td>
                <td class="{vol_score_class}">{candle['volume_score']}</td>
                <td class="{cum_volume_class}">{candle['cumulative_volume_ratio']:.2f}x</td>
                <td class="{cum_vol_score_class}">{candle['cumulative_volume_score']}</td>
                <td class="{longterm_volume_class}">{candle['longterm_volume_ratio']:.2f}x</td>
                <td class="{longterm_vol_score_class}">{candle['longterm_volume_score']}</td>
                <td class="{range_class}">{candle['range_ratio']:.2f}x</td>
                <td class="{range_class}">{candle['tail_percentage']:.1%}</td>
                <td class="{range_score_class}">{candle['range_score']}</td>
                <td class="{range_class}">{candle['crossover_range_ratio']:.2f}x</td>
                <td class="{range_class}">{candle['crossover_tail_percentage']:.1%}</td>
                <td class="{range_score_class}">{candle['crossover_range_score']}</td>
                <td>{candle['candles_from_bottom']}</td>
                <td class="{score_class}">{candle['breakout_quality']}</td>
                <td>{status}</td>
            </tr>
"""

        html_content += f"""
        </table>

        <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Analysis Parameters:</strong> Volume Window: {self.volume_window} bars,
               Volume Multiplier: {self.volume_multiplier}x, Passing Score: {self.passing_score}</p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML file
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if data_date:
            # Use the data date instead of current timestamp
            # Extract just the date part and clean it (e.g., "2023-03-01 00:00:00" -> "20230301")
            date_only = data_date.split(' ')[0]  # Get just the date part
            date_str = date_only.replace('-', '')  # Convert 2023-03-01 to 20230301
            html_filename = f"master_candle_report_{date_str}.html"
        else:
            html_filename = f"master_candle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path = os.path.join(output_dir, html_filename)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

def analyze_mastercandles(price_data: pd.DataFrame, creek_data: pd.DataFrame, fractals_data: pd.DataFrame = None,
                         volume_window: int = 60, volume_multiplier: float = 2.0,
                         passing_score: int = 50) -> Tuple[List[Dict], str]:
    """
    Main function to analyze master candles from crossover data

    Args:
        price_data: DataFrame with price and volume data
        creek_data: DataFrame with pending creek information
        volume_window: Number of bars for volume average calculation
        volume_multiplier: Minimum volume multiplier requirement
        passing_score: Minimum score to qualify as master candle

    Returns:
        Tuple of (master_candles_list, html_report_path)
    """
    detector = MasterCandleDetector(volume_window, volume_multiplier, passing_score)

    print(f"\n{'='*60}")
    print("MASTER CANDLE DETECTION SYSTEM")
    print(f"{'='*60}")
    print(f"Volume Window: {volume_window} bars")
    print(f"Volume Multiplier: {volume_multiplier}x")
    print(f"Passing Score: {passing_score}")

    # Step 1: Detect crossovers
    crossovers = detector.detect_crossovers(price_data, creek_data)

    if not crossovers:
        print("No crossovers detected - skipping master candle analysis")
        return [], ""

    # Step 2: Evaluate each crossover
    print(f"\n{'='*60}")
    print("EVALUATING CROSSOVERS FOR MASTER CANDLE QUALIFICATION")
    print(f"{'='*60}")

    evaluated_crossovers = []
    for i, crossover in enumerate(crossovers, 1):
        print(f"\nCrossover {i}/{len(crossovers)}:")
        print(f"  Time: {crossover['crossover_time']}")
        print(f"  Price: ${crossover['crossover_close']:.2f}")
        print(f"  Volume: {crossover['crossover_volume']:,}")

        evaluated = detector.evaluate_mastercandle(crossover, price_data, fractals_data)
        evaluated_crossovers.append(evaluated)

        print(f"  Volume Ratio: {evaluated['volume_ratio']:.2f}x")
        print(f"  Volume Score: {evaluated['volume_score']}/10")
        print(f"  Breakout Quality: {evaluated['breakout_quality']}")
        print(f"  Master Candle: {'[YES]' if evaluated['is_mastercandle'] else '[NO]'}")

    # Step 3: Generate report
    master_candles = [c for c in evaluated_crossovers if c['is_mastercandle']]

    print(f"\n{'='*60}")
    print("MASTER CANDLE DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Total Crossovers: {len(crossovers)}")
    print(f"Master Candles: {len(master_candles)}")
    print(f"Success Rate: {(len(master_candles)/len(crossovers)*100):.1f}%")

    # Extract data date from the price_data for filename
    data_date = None
    if 'date' in price_data.columns and len(price_data) > 0:
        data_date = str(price_data['date'].iloc[0])  # Get first date from the data

    # Generate HTML report
    html_path = detector.generate_html_report(evaluated_crossovers, data_date)
    print(f"\nHTML Report generated: {html_path}")

    # Open report in browser
    try:
        webbrowser.open('file://' + os.path.realpath(html_path))
        print("Report opened in web browser")
    except Exception as e:
        print(f"Could not open browser: {e}")

    return master_candles, html_path

if __name__ == "__main__":
    print("Master Candle Detection module - use from main.py or find_tops_and_bottoms.py")