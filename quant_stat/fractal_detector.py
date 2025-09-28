from collections import deque
from typing import List, Tuple, Optional
from enum import Enum
import time

class FractalType(Enum):
    PEAK = "peak"      # Top/Maximum
    VALLEY = "valley"  # Bottom/Minimum

class FractalState(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"

class ZigzagDirection(Enum):
    UP = "up"      # Looking for peak
    DOWN = "down"  # Looking for valley

class Fractal:
    def __init__(self, index: int, price: float, fractal_type: FractalType, timestamp: Optional[str] = None):
        self.index = index
        self.price = price
        self.fractal_type = fractal_type
        self.state = FractalState.PENDING
        self.timestamp = timestamp
        self.detection_time = time.time()

class ZigzagPoint:
    def __init__(self, index: int, price: float, direction: ZigzagDirection, timestamp: Optional[str] = None):
        self.index = index
        self.price = price
        self.direction = direction
        self.timestamp = timestamp
        self.confirmed = False

class UnifiedZigzagDetector:
    """
    Unified Zigzag detector that guarantees alternating peaks and valleys
    Based on percentage change detection
    """
    def __init__(self, min_change_pct: float = 0.15):
        self.min_change_pct = min_change_pct / 100.0  # Convert to decimal

        # Detector state
        self.candles = []
        self.current_trend = None  # None, UP (looking for peak), DOWN (looking for valley)
        self.last_pivot = None  # Last confirmed pivot point

        # Tracking buffers
        self.current_high = None
        self.current_high_index = None
        self.current_high_time = None
        self.current_low = None
        self.current_low_index = None
        self.current_low_time = None

        # Detected zigzag points
        self.zigzag_points = []

    def add_candle(self, high: float, low: float, index: int, timestamp: str = None) -> Optional[ZigzagPoint]:
        """
        Add a new candle and return a zigzag point if detected
        """
        candle = {
            'high': high,
            'low': low,
            'index': index,
            'timestamp': timestamp
        }
        self.candles.append(candle)

        # First candle - initialize
        if len(self.candles) == 1:
            self.current_high = high
            self.current_high_index = index
            self.current_high_time = timestamp
            self.current_low = low
            self.current_low_index = index
            self.current_low_time = timestamp
            return None

        # Second candle - determine initial trend
        if len(self.candles) == 2:
            if high > self.current_high:
                self.current_high = high
                self.current_high_index = index
                self.current_high_time = timestamp
            if low < self.current_low:
                self.current_low = low
                self.current_low_index = index
                self.current_low_time = timestamp

            # Establish initial trend based on which moved more
            high_change = (self.current_high - self.candles[0]['high']) / self.candles[0]['high']
            low_change = (self.candles[0]['low'] - self.current_low) / self.candles[0]['low']

            if high_change > low_change:
                self.current_trend = ZigzagDirection.UP  # Looking for peak
                # First point is a valley
                self.last_pivot = ZigzagPoint(
                    self.current_low_index,
                    self.current_low,
                    ZigzagDirection.DOWN,
                    self.current_low_time
                )
            else:
                self.current_trend = ZigzagDirection.DOWN  # Looking for valley
                # First point is a peak
                self.last_pivot = ZigzagPoint(
                    self.current_high_index,
                    self.current_high,
                    ZigzagDirection.UP,
                    self.current_high_time
                )
            return None

        return self._check_for_pivot(candle)

    def _check_for_pivot(self, candle: dict) -> Optional[ZigzagPoint]:
        """
        Check if current candle creates a pivot point
        """
        high = candle['high']
        low = candle['low']
        index = candle['index']
        timestamp = candle['timestamp']

        # Update current extremes
        if high > self.current_high:
            self.current_high = high
            self.current_high_index = index
            self.current_high_time = timestamp
        if low < self.current_low:
            self.current_low = low
            self.current_low_index = index
            self.current_low_time = timestamp

        if not self.last_pivot:
            return None

        # If looking for a peak (uptrend)
        if self.current_trend == ZigzagDirection.UP:
            # Check for significant drop from current high
            if self.current_high > self.last_pivot.price:  # New high
                change_from_high = (self.current_high - low) / self.current_high
                if change_from_high >= self.min_change_pct:
                    # Confirm the peak
                    pivot = ZigzagPoint(
                        self.current_high_index,
                        self.current_high,
                        ZigzagDirection.UP,
                        self.current_high_time
                    )
                    pivot.confirmed = True
                    self.zigzag_points.append(pivot)
                    self.last_pivot = pivot
                    self.current_trend = ZigzagDirection.DOWN  # Now look for valley

                    # Reset tracking for next valley
                    self.current_low = low
                    self.current_low_index = index
                    self.current_low_time = timestamp

                    return pivot

        # If looking for a valley (downtrend)
        elif self.current_trend == ZigzagDirection.DOWN:
            # Check for significant rise from current low
            if self.current_low < self.last_pivot.price:  # New low
                change_from_low = (high - self.current_low) / self.current_low
                if change_from_low >= self.min_change_pct:
                    # Confirm the valley
                    pivot = ZigzagPoint(
                        self.current_low_index,
                        self.current_low,
                        ZigzagDirection.DOWN,
                        self.current_low_time
                    )
                    pivot.confirmed = True
                    self.zigzag_points.append(pivot)
                    self.last_pivot = pivot
                    self.current_trend = ZigzagDirection.UP  # Now look for peak

                    # Reset tracking for next peak
                    self.current_high = high
                    self.current_high_index = index
                    self.current_high_time = timestamp

                    return pivot

        return None

    def get_zigzag_points(self) -> List[ZigzagPoint]:
        """Return all detected zigzag points"""
        return self.zigzag_points.copy()

    def get_fractals(self) -> List[Fractal]:
        """Convert zigzag points to fractal format"""
        fractals = []
        for point in self.zigzag_points:
            fractal_type = FractalType.PEAK if point.direction == ZigzagDirection.UP else FractalType.VALLEY
            fractal = Fractal(point.index, point.price, fractal_type, point.timestamp)
            fractal.state = FractalState.CONFIRMED
            fractals.append(fractal)
        return fractals

class RealTimeExtremeDetector:
    """
    Window-based fractal detector with confirmation periods
    """
    def __init__(self, window_size: int = 10, confirmation_periods: int = 5, min_strength: float = 0.3):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")

        self.window_size = window_size
        self.confirmation_periods = confirmation_periods
        self.half_window = window_size // 2
        self.min_strength = min_strength

        # Price buffer
        self.prices = deque(maxlen=window_size + confirmation_periods)
        self.timestamps = deque(maxlen=window_size + confirmation_periods)
        self.price_index = 0

        # Detected extremes
        self.extremes_pending = []
        self.extremes_confirmed = []

    def add_price(self, price: float, timestamp: str = None) -> List[Fractal]:
        """
        Add new price and return newly confirmed fractals
        """
        self.prices.append(price)
        self.timestamps.append(timestamp)
        self.price_index += 1

        newly_confirmed = []

        # Start detecting when we have enough data
        if len(self.prices) >= self.window_size:
            extremo = self._detect_extremo()
            if extremo:
                self.extremes_pending.append(extremo)

        # Check confirmations
        if len(self.prices) >= self.window_size + self.confirmation_periods:
            confirmed = self._check_confirmations()
            newly_confirmed.extend(confirmed)

        return newly_confirmed

    def _detect_extremo(self) -> Optional[Fractal]:
        """
        Detect if there's an extreme at the center of current window
        """
        if len(self.prices) < self.window_size:
            return None

        # Get current window
        window = list(self.prices)[-self.window_size:]
        window_times = list(self.timestamps)[-self.window_size:]
        center_idx = self.half_window
        center_price = window[center_idx]
        center_time = window_times[center_idx]

        # Check if it's a local maximum
        is_maximum = all(center_price >= price for i, price in enumerate(window) if i != center_idx)
        # Check if it's a local minimum
        is_minimum = all(center_price <= price for i, price in enumerate(window) if i != center_idx)

        # Avoid flat regions
        if self._is_flat_region(window, center_idx):
            return None

        # Calculate strength
        if is_maximum or is_minimum:
            if not self._is_strong_extremo(window, center_idx):
                return None

        if is_maximum:
            global_index = self.price_index - self.window_size + center_idx
            return Fractal(global_index, center_price, FractalType.PEAK, center_time)

        if is_minimum:
            global_index = self.price_index - self.window_size + center_idx
            return Fractal(global_index, center_price, FractalType.VALLEY, center_time)

        return None

    def _is_flat_region(self, window: List[float], center_idx: int, tolerance: float = 1e-6) -> bool:
        """Check if region around center is flat"""
        center_price = window[center_idx]
        for price in window:
            if abs(price - center_price) > tolerance:
                return False
        return True

    def _is_strong_extremo(self, window: List[float], center_idx: int) -> bool:
        """Check if extreme has sufficient strength"""
        center_price = window[center_idx]
        other_prices = [price for i, price in enumerate(window) if i != center_idx]
        avg_price = sum(other_prices) / len(other_prices)

        if avg_price == 0:
            return False

        strength = abs(center_price - avg_price) / avg_price
        return strength >= self.min_strength

    def _check_confirmations(self) -> List[Fractal]:
        """Check which pending extremes can be confirmed"""
        confirmed = []
        extremes_to_remove = []

        for extremo in self.extremes_pending:
            periods_since_detection = self.price_index - extremo.index

            if periods_since_detection >= self.confirmation_periods:
                if self._is_extremo_still_valid(extremo):
                    extremo.state = FractalState.CONFIRMED
                    self.extremes_confirmed.append(extremo)
                    confirmed.append(extremo)
                else:
                    extremo.state = FractalState.INVALIDATED

                extremes_to_remove.append(extremo)

        for extremo in extremes_to_remove:
            self.extremes_pending.remove(extremo)

        return confirmed

    def _is_extremo_still_valid(self, extremo: Fractal) -> bool:
        """Check if pending extreme is still valid after confirmation period"""
        start_idx = max(0, len(self.prices) - (self.price_index - extremo.index))
        subsequent_prices = list(self.prices)[start_idx + 1:]

        if not subsequent_prices:
            return True

        if extremo.fractal_type == FractalType.PEAK:
            return all(price <= extremo.price for price in subsequent_prices[-self.confirmation_periods:])
        else:
            return all(price >= extremo.price for price in subsequent_prices[-self.confirmation_periods:])

    def get_confirmed_fractals(self) -> List[Fractal]:
        """Return all confirmed fractals"""
        return self.extremes_confirmed.copy()