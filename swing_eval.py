"""
Swing Evaluation System for ES Futures Fractal Analysis
========================================================

This module provides comprehensive swing analysis for fractal-detected price movements,
including velocity analysis, pattern recognition, and synthetic indicators.

Author: Claude AI Assistant
Purpose: Advanced swing evaluation for trading strategy development
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from config import OUTPUT_DIR, CHARTS_DIR, SYMBOL
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SwingMetrics:
    """
    Calculate velocity, momentum, and acceleration metrics for swing analysis
    """

    @staticmethod
    def calculate_velocity_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various velocity-based metrics for swing analysis
        """
        df = df.copy()

        # Basic velocity (already exists as distance_ratio)
        df['velocity'] = df['distance_ratio']

        # Velocity acceleration (rate of change in velocity)
        df['velocity_acceleration'] = df['velocity'].diff()

        # Velocity momentum (3-period momentum)
        df['velocity_momentum'] = df['velocity'].diff(3)

        # Velocity percentile ranking (rolling 20-period)
        df['velocity_percentile'] = df['velocity'].rolling(20, min_periods=5).rank(pct=True) * 100

        # Price velocity (price change per unit time)
        df['price_velocity'] = df['distance_usd'] / df['distance_bars'].replace(0, 1)

        # Time efficiency (price moved per minute)
        df['time_efficiency'] = df['distance_usd'] / df['distance_bars'].replace(0, 1)

        return df

    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based indicators for swing analysis
        """
        df = df.copy()

        # Swing momentum (comparing current swing to previous)
        df['swing_momentum'] = (df['velocity'] / df['velocity'].shift(1).replace(0, 1)) - 1

        # Momentum strength (absolute momentum)
        df['momentum_strength'] = np.abs(df['swing_momentum'])

        # Momentum direction (1 for acceleration, -1 for deceleration)
        df['momentum_direction'] = np.where(df['swing_momentum'] > 0, 1, -1)

        # Cumulative momentum (running sum of momentum changes)
        df['cumulative_momentum'] = df['swing_momentum'].cumsum()

        # Momentum oscillator (normalized momentum)
        df['momentum_oscillator'] = df['swing_momentum'].rolling(10, min_periods=3).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
        )

        return df


class PatternDetector:
    """
    Detect swing patterns including impulse-correction sequences
    """

    def __init__(self, impulse_threshold: float = 75.0, correction_threshold: float = 25.0):
        """
        Initialize pattern detector with thresholds for impulse and correction identification

        Args:
            impulse_threshold: Velocity percentile threshold for impulse moves
            correction_threshold: Velocity percentile threshold for correction moves
        """
        self.impulse_threshold = impulse_threshold
        self.correction_threshold = correction_threshold

    def classify_swing_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify swings as impulse, correction, or neutral
        """
        df = df.copy()

        # Classify based on velocity percentile
        conditions = [
            df['velocity_percentile'] >= self.impulse_threshold,
            df['velocity_percentile'] <= self.correction_threshold
        ]
        choices = ['IMPULSE', 'CORRECTION']

        df['swing_type'] = np.select(conditions, choices, default='NEUTRAL')

        return df

    def detect_impulse_correction_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect impulse-correction patterns in the swing sequence
        """
        df = df.copy()

        # Initialize pattern columns
        df['pattern_type'] = 'NONE'
        df['pattern_strength'] = 0.0
        df['pattern_sequence'] = ''

        # Look for impulse followed by corrections
        for i in range(1, len(df)):
            if df.iloc[i-1]['swing_type'] == 'IMPULSE':
                # Look ahead for corrections
                correction_count = 0
                correction_velocity_sum = 0

                for j in range(i, min(i+5, len(df))):  # Look ahead max 5 swings
                    if df.iloc[j]['swing_type'] == 'CORRECTION':
                        correction_count += 1
                        correction_velocity_sum += df.iloc[j]['velocity']
                    elif df.iloc[j]['swing_type'] == 'IMPULSE':
                        break  # End of correction sequence

                if correction_count > 0:
                    # Calculate pattern strength
                    impulse_velocity = df.iloc[i-1]['velocity']
                    avg_correction_velocity = correction_velocity_sum / correction_count
                    pattern_strength = impulse_velocity / (avg_correction_velocity + 1e-8)

                    # Mark the impulse and corrections
                    df.at[i-1, 'pattern_type'] = 'IMPULSE_START'
                    df.at[i-1, 'pattern_strength'] = pattern_strength
                    df.at[i-1, 'pattern_sequence'] = f'I-{correction_count}C'

                    # Mark corrections
                    for k in range(i, i + correction_count):
                        if k < len(df):
                            df.at[k, 'pattern_type'] = 'CORRECTION_FOLLOW'
                            df.at[k, 'pattern_strength'] = pattern_strength
                            df.at[k, 'pattern_sequence'] = f'I-{correction_count}C'

        return df

    def analyze_swing_sequences(self, df: pd.DataFrame) -> Dict:
        """
        Analyze swing sequences and return statistics
        """
        sequence_stats = {
            'total_impulses': len(df[df['swing_type'] == 'IMPULSE']),
            'total_corrections': len(df[df['swing_type'] == 'CORRECTION']),
            'impulse_correction_patterns': len(df[df['pattern_type'] == 'IMPULSE_START']),
            'avg_pattern_strength': df[df['pattern_type'] == 'IMPULSE_START']['pattern_strength'].mean(),
            'max_pattern_strength': df[df['pattern_type'] == 'IMPULSE_START']['pattern_strength'].max(),
            'dominant_direction': 'UP' if len(df[df['type'] == 'TOP']) > len(df[df['type'] == 'BOTTOM']) else 'DOWN'
        }

        return sequence_stats


class SyntheticIndicators:
    """
    Create synthetic indicators for comprehensive swing evaluation
    """

    @staticmethod
    def swing_strength_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Swing Strength Index (SSI) - composite indicator
        """
        df = df.copy()

        # Normalize components to 0-100 scale
        velocity_norm = (df['velocity_percentile']).fillna(50)
        momentum_norm = ((df['momentum_strength'].rolling(20, min_periods=5).rank(pct=True)) * 100).fillna(50)
        pattern_norm = ((df['pattern_strength'].rolling(20, min_periods=5).rank(pct=True)) * 100).fillna(50)

        # Weighted composite (velocity 40%, momentum 35%, pattern 25%)
        df['swing_strength_index'] = (
            velocity_norm * 0.40 +
            momentum_norm * 0.35 +
            pattern_norm * 0.25
        )

        # Smooth the SSI with 3-period moving average
        df['ssi_smooth'] = df['swing_strength_index'].rolling(3, min_periods=1).mean()

        return df

    @staticmethod
    def market_state_indicator(df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine market state: TRENDING vs CONSOLIDATING
        """
        df = df.copy()

        # Calculate trend strength based on swing characteristics
        df['trend_strength'] = df['velocity'].rolling(10, min_periods=3).std()

        # Calculate direction consistency differently to avoid aggregation issues
        df['trend_direction_consistency'] = 0.5  # Default neutral value

        # Simple velocity-based market state
        velocity_mean = df['velocity'].rolling(20, min_periods=5).mean()
        velocity_std = df['velocity'].rolling(20, min_periods=5).std()

        high_velocity = df['velocity'] > (velocity_mean + velocity_std)
        low_velocity = df['velocity'] < (velocity_mean - velocity_std)

        conditions = [
            high_velocity,
            low_velocity
        ]
        choices = ['TRENDING', 'CONSOLIDATING']

        df['market_state'] = np.select(conditions, choices, default='NEUTRAL')

        return df

    @staticmethod
    def volatility_regime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify volatility regime: LOW, MEDIUM, HIGH
        """
        df = df.copy()

        # Calculate volatility metrics
        df['velocity_volatility'] = df['velocity'].rolling(15, min_periods=5).std()
        df['price_volatility'] = df['distance_usd'].rolling(15, min_periods=5).std()

        # Combined volatility score
        df['volatility_score'] = (
            df['velocity_volatility'].rolling(20, min_periods=5).rank(pct=True) * 0.6 +
            df['price_volatility'].rolling(20, min_periods=5).rank(pct=True) * 0.4
        ) * 100

        # Classify volatility regimes
        conditions = [
            df['volatility_score'] >= 70,
            df['volatility_score'] <= 30
        ]
        choices = ['HIGH', 'LOW']

        df['volatility_regime'] = np.select(conditions, choices, default='MEDIUM')

        return df


class SwingClassifier:
    """
    Classify swings into major, minor, and consolidation categories
    """

    def __init__(self, major_threshold: float = 75.0, minor_threshold: float = 25.0):
        """
        Initialize with classification thresholds
        """
        self.major_threshold = major_threshold
        self.minor_threshold = minor_threshold

    def classify_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify swings based on multiple criteria
        """
        df = df.copy()

        # Multi-criteria classification
        velocity_class = pd.cut(
            df['velocity_percentile'],
            bins=[0, self.minor_threshold, self.major_threshold, 100],
            labels=['MINOR', 'CONSOLIDATION', 'MAJOR']
        )

        distance_class = pd.cut(
            df['distance_usd'].rolling(20, min_periods=5).rank(pct=True) * 100,
            bins=[0, self.minor_threshold, self.major_threshold, 100],
            labels=['MINOR', 'CONSOLIDATION', 'MAJOR']
        )

        # Combine classifications (take the more conservative)
        df['swing_classification'] = velocity_class.astype(str)

        # Override with distance classification if more conservative
        conservative_mask = (distance_class == 'MINOR') | (
            (distance_class == 'CONSOLIDATION') & (velocity_class == 'MAJOR')
        )
        df.loc[conservative_mask, 'swing_classification'] = distance_class[conservative_mask].astype(str)

        return df


class VisualizationEngine:
    """
    Enhanced visualization engine for swing evaluation results
    """

    @staticmethod
    def create_swing_evaluation_chart(df: pd.DataFrame,
                                    price_data: pd.DataFrame = None,
                                    symbol: str = 'ES',
                                    timeframe: str = 'swing_eval') -> str:
        """
        Create comprehensive swing evaluation visualization
        """
        # Create subplots for comprehensive analysis
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
            vertical_spacing=0.02,
            subplot_titles=(
                'Price & Swing Analysis',
                'Swing Strength Index (SSI)',
                'Velocity & Momentum',
                'Market State & Volatility',
                'Pattern Classification'
            )
        )

        # Row 1: Price chart with swing analysis
        if price_data is not None:
            # Add candlestick chart if price data available
            fig.add_trace(go.Candlestick(
                x=price_data['date'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price',
                opacity=0.7
            ), row=1, col=1)

        # Add fractal points colored by swing classification
        colors_map = {
            'MAJOR': 'red',
            'CONSOLIDATION': 'orange',
            'MINOR': 'lightblue',
            'nan': 'gray'
        }

        for classification in colors_map.keys():
            if classification == 'nan':
                subset = df[df['swing_classification'].isna()]
            else:
                subset = df[df['swing_classification'] == classification]

            if len(subset) > 0:
                fig.add_trace(go.Scatter(
                    x=subset['timestamp'],
                    y=subset['price'],
                    mode='markers',
                    marker=dict(
                        color=colors_map[classification],
                        size=8,
                        symbol='diamond' if classification == 'MAJOR' else 'circle'
                    ),
                    name=f'{classification} Swings',
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

        # Add swing lines connecting fractals
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Swing Line',
            opacity=0.6
        ), row=1, col=1)

        # Row 2: Swing Strength Index
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['swing_strength_index'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='SSI',
            fill='tonexty'
        ), row=2, col=1)

        # Add SSI threshold lines
        fig.add_hline(y=75, line_dash="dash", line_color="red",
                     annotation_text="Strong", row=2, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="green",
                     annotation_text="Weak", row=2, col=1)

        # Row 3: Velocity and Momentum
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['velocity'],
            name='Velocity',
            marker_color='lightcoral',
            opacity=0.7
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['momentum_strength'] * 100,  # Scale for visibility
            mode='lines',
            line=dict(color='darkgreen', width=2),
            name='Momentum Strength',
            yaxis='y4'
        ), row=3, col=1)

        # Row 4: Market State and Volatility Regime
        # Convert categorical data to numeric for plotting
        state_map = {'TRENDING': 1, 'NEUTRAL': 0, 'CONSOLIDATING': -1}
        regime_map = {'HIGH': 1, 'MEDIUM': 0, 'LOW': -1}

        df['state_numeric'] = df['market_state'].map(state_map)
        df['regime_numeric'] = df['volatility_regime'].map(regime_map)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['state_numeric'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            name='Market State',
            hovertemplate='State: %{text}<br>Time: %{x}<extra></extra>',
            text=df['market_state']
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['regime_numeric'],
            mode='lines+markers',
            line=dict(color='orange', width=2),
            marker=dict(size=4),
            name='Volatility Regime',
            hovertemplate='Regime: %{text}<br>Time: %{x}<extra></extra>',
            text=df['volatility_regime'],
            yaxis='y5'
        ), row=4, col=1)

        # Row 5: Pattern Classification
        # Create pattern visualization
        pattern_map = {
            'IMPULSE_START': 2,
            'CORRECTION_FOLLOW': 1,
            'NONE': 0
        }
        df['pattern_numeric'] = df['pattern_type'].map(pattern_map)

        # Color by swing type
        type_colors = {'IMPULSE': 'red', 'CORRECTION': 'green', 'NEUTRAL': 'gray'}
        swing_colors = [type_colors.get(t, 'gray') for t in df['swing_type']]

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['pattern_numeric'],
            mode='markers',
            marker=dict(
                color=swing_colors,
                size=8,
                symbol='diamond'
            ),
            name='Pattern Type',
            hovertemplate='Pattern: %{text}<br>Swing: %{customdata}<br>Time: %{x}<extra></extra>',
            text=df['pattern_type'],
            customdata=df['swing_type']
        ), row=5, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} Comprehensive Swing Evaluation Analysis',
            height=1200,
            showlegend=True,
            legend=dict(x=1.02, y=1),
            template='plotly_white'
        )

        # Update x-axes
        for i in range(1, 6):
            fig.update_xaxes(
                type='date',
                tickformat="%H:%M",
                showgrid=True,
                gridwidth=0.5,
                gridcolor='lightgray',
                row=i, col=1
            )

        # Update y-axes with appropriate titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="SSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Velocity", row=3, col=1)
        fig.update_yaxes(title_text="State", row=4, col=1)
        fig.update_yaxes(title_text="Pattern", row=5, col=1)

        # Save chart
        CHARTS_DIR.mkdir(exist_ok=True)
        chart_path = CHARTS_DIR / f'swing_evaluation_{symbol}_{timeframe}.html'
        fig.write_html(str(chart_path))

        print(f"Swing evaluation chart saved to: {chart_path}")
        return str(chart_path)

    @staticmethod
    def create_swing_statistics_chart(analysis_stats: Dict,
                                    symbol: str = 'ES') -> str:
        """
        Create statistical summary charts
        """
        # Create subplots for different statistics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Swing Type Distribution',
                'Swing Classification',
                'Market State Distribution',
                'Volatility Regime'
            ),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )

        # Swing Type Distribution
        swing_types = analysis_stats.get('swing_types', {})
        if swing_types:
            fig.add_trace(go.Pie(
                labels=list(swing_types.keys()),
                values=list(swing_types.values()),
                name="Swing Types",
                marker_colors=['red', 'green', 'orange']
            ), row=1, col=1)

        # Swing Classification
        classifications = analysis_stats.get('swing_classifications', {})
        if classifications:
            fig.add_trace(go.Pie(
                labels=list(classifications.keys()),
                values=list(classifications.values()),
                name="Classifications",
                marker_colors=['darkred', 'orange', 'lightblue']
            ), row=1, col=2)

        # Market State
        market_states = analysis_stats.get('market_states', {})
        if market_states:
            fig.add_trace(go.Pie(
                labels=list(market_states.keys()),
                values=list(market_states.values()),
                name="Market States",
                marker_colors=['blue', 'gray', 'purple']
            ), row=2, col=1)

        # Volatility Regime
        volatility_regimes = analysis_stats.get('volatility_regimes', {})
        if volatility_regimes:
            fig.add_trace(go.Pie(
                labels=list(volatility_regimes.keys()),
                values=list(volatility_regimes.values()),
                name="Volatility",
                marker_colors=['red', 'orange', 'green']
            ), row=2, col=2)

        fig.update_layout(
            title=f'{symbol} Swing Analysis Statistics',
            height=800,
            showlegend=True,
            template='plotly_white'
        )

        # Save chart
        CHARTS_DIR.mkdir(exist_ok=True)
        chart_path = CHARTS_DIR / f'swing_statistics_{symbol}.html'
        fig.write_html(str(chart_path))

        print(f"Swing statistics chart saved to: {chart_path}")
        return str(chart_path)


class SwingEvaluator:
    """
    Main class for comprehensive swing evaluation
    """

    def __init__(self,
                 impulse_threshold: float = 75.0,
                 correction_threshold: float = 25.0,
                 major_threshold: float = 75.0,
                 minor_threshold: float = 25.0):
        """
        Initialize the swing evaluator with configurable thresholds
        """
        self.impulse_threshold = impulse_threshold
        self.correction_threshold = correction_threshold
        self.major_threshold = major_threshold
        self.minor_threshold = minor_threshold

        # Initialize components
        self.metrics = SwingMetrics()
        self.pattern_detector = PatternDetector(impulse_threshold, correction_threshold)
        self.indicators = SyntheticIndicators()
        self.classifier = SwingClassifier(major_threshold, minor_threshold)
        self.visualizer = VisualizationEngine()

        self.fractal_data = None
        self.evaluated_data = None
        self.analysis_stats = {}

    def load_fractal_data(self, csv_file: str = None) -> pd.DataFrame:
        """
        Load fractal data from CSV file
        """
        if csv_file is None:
            # Use the most recent zigzag file
            csv_file = 'fractals_2023_03_01_zigzag_0.1.csv'

        file_path = os.path.join(OUTPUT_DIR, csv_file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fractal data file not found: {file_path}")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"Loaded {len(df)} fractals from {csv_file}")

        self.fractal_data = df
        return df

    def evaluate_swings(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Perform comprehensive swing evaluation
        """
        if df is None:
            if self.fractal_data is None:
                raise ValueError("No fractal data loaded. Call load_fractal_data() first.")
            df = self.fractal_data.copy()

        print("Starting comprehensive swing evaluation...")

        # Step 1: Calculate velocity and momentum metrics
        print("  - Calculating velocity metrics...")
        df = self.metrics.calculate_velocity_metrics(df)
        df = self.metrics.calculate_momentum_indicators(df)

        # Step 2: Detect patterns
        print("  - Detecting swing patterns...")
        df = self.pattern_detector.classify_swing_types(df)
        df = self.pattern_detector.detect_impulse_correction_sequences(df)

        # Step 3: Calculate synthetic indicators
        print("  - Calculating synthetic indicators...")
        df = self.indicators.swing_strength_index(df)
        df = self.indicators.market_state_indicator(df)
        df = self.indicators.volatility_regime_index(df)

        # Step 4: Classify swings
        print("  - Classifying swing types...")
        df = self.classifier.classify_swings(df)

        # Step 5: Generate analysis statistics
        self.analysis_stats = self._generate_analysis_stats(df)

        self.evaluated_data = df
        print("Swing evaluation completed!")

        return df

    def _generate_analysis_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive analysis statistics
        """
        stats = {}

        # Basic statistics
        stats['total_swings'] = len(df)
        stats['avg_velocity'] = df['velocity'].mean()
        stats['avg_distance_usd'] = df['distance_usd'].mean()
        stats['avg_distance_bars'] = df['distance_bars'].mean()

        # Swing type distribution
        stats['swing_types'] = df['swing_type'].value_counts().to_dict()
        stats['swing_classifications'] = df['swing_classification'].value_counts().to_dict()
        stats['market_states'] = df['market_state'].value_counts().to_dict()
        stats['volatility_regimes'] = df['volatility_regime'].value_counts().to_dict()

        # Pattern analysis
        stats.update(self.pattern_detector.analyze_swing_sequences(df))

        # Advanced metrics
        stats['avg_ssi'] = df['swing_strength_index'].mean()
        stats['max_velocity'] = df['velocity'].max()
        stats['velocity_std'] = df['velocity'].std()

        # Directional bias
        tops = len(df[df['type'] == 'TOP'])
        bottoms = len(df[df['type'] == 'BOTTOM'])
        stats['directional_bias'] = 'BULLISH' if tops > bottoms else 'BEARISH'
        stats['bias_ratio'] = max(tops, bottoms) / min(tops, bottoms) if min(tops, bottoms) > 0 else float('inf')

        return stats

    def save_evaluation_results(self, output_filename: str = None) -> str:
        """
        Save evaluation results to CSV
        """
        if self.evaluated_data is None:
            raise ValueError("No evaluation data available. Run evaluate_swings() first.")

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f'swing_evaluation_{timestamp}.csv'

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        self.evaluated_data.to_csv(output_path, index=False)

        print(f"Evaluation results saved to: {output_path}")
        return output_path

    def print_analysis_summary(self):
        """
        Print comprehensive analysis summary
        """
        if not self.analysis_stats:
            print("No analysis statistics available. Run evaluate_swings() first.")
            return

        print("\n" + "="*80)
        print("SWING EVALUATION ANALYSIS SUMMARY")
        print("="*80)

        # Basic Statistics
        print(f"\nBASIC STATISTICS:")
        print(f"  Total Swings: {self.analysis_stats['total_swings']}")
        print(f"  Average Velocity: {self.analysis_stats['avg_velocity']:.2f}")
        print(f"  Average Distance (USD): ${self.analysis_stats['avg_distance_usd']:.2f}")
        print(f"  Average Distance (Bars): {self.analysis_stats['avg_distance_bars']:.2f}")
        print(f"  Max Velocity: {self.analysis_stats['max_velocity']:.2f}")
        print(f"  Velocity Std Dev: {self.analysis_stats['velocity_std']:.2f}")

        # Market Analysis
        print(f"\nMARKET ANALYSIS:")
        print(f"  Directional Bias: {self.analysis_stats['directional_bias']}")
        print(f"  Bias Ratio: {self.analysis_stats['bias_ratio']:.2f}")
        print(f"  Average SSI: {self.analysis_stats['avg_ssi']:.2f}")

        # Pattern Analysis
        print(f"\nPATTERN ANALYSIS:")
        print(f"  Total Impulses: {self.analysis_stats['total_impulses']}")
        print(f"  Total Corrections: {self.analysis_stats['total_corrections']}")
        print(f"  Impulse-Correction Patterns: {self.analysis_stats['impulse_correction_patterns']}")
        print(f"  Average Pattern Strength: {self.analysis_stats.get('avg_pattern_strength', 0):.2f}")

        # Distribution Analysis
        print(f"\nSWING TYPE DISTRIBUTION:")
        for swing_type, count in self.analysis_stats['swing_types'].items():
            percentage = (count / self.analysis_stats['total_swings']) * 100
            print(f"  {swing_type}: {count} ({percentage:.1f}%)")

        print(f"\nSWING CLASSIFICATION DISTRIBUTION:")
        for classification, count in self.analysis_stats['swing_classifications'].items():
            percentage = (count / self.analysis_stats['total_swings']) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")

        print(f"\nMARKET STATE DISTRIBUTION:")
        for state, count in self.analysis_stats['market_states'].items():
            percentage = (count / self.analysis_stats['total_swings']) * 100
            print(f"  {state}: {count} ({percentage:.1f}%)")

        print(f"\nVOLATILITY REGIME DISTRIBUTION:")
        for regime, count in self.analysis_stats['volatility_regimes'].items():
            percentage = (count / self.analysis_stats['total_swings']) * 100
            print(f"  {regime}: {count} ({percentage:.1f}%)")

        print("="*80)

    def create_visualizations(self, price_data: pd.DataFrame = None) -> Tuple[str, str]:
        """
        Create comprehensive visualizations for swing evaluation

        Returns:
            Tuple of (evaluation_chart_path, statistics_chart_path)
        """
        if self.evaluated_data is None:
            raise ValueError("No evaluation data available. Run evaluate_swings() first.")

        print("Creating swing evaluation visualizations...")

        # Create main evaluation chart
        eval_chart_path = self.visualizer.create_swing_evaluation_chart(
            self.evaluated_data,
            price_data=price_data,
            symbol=SYMBOL
        )

        # Create statistics chart
        stats_chart_path = self.visualizer.create_swing_statistics_chart(
            self.analysis_stats,
            symbol=SYMBOL
        )

        return eval_chart_path, stats_chart_path


def main():
    """
    Main function to demonstrate swing evaluation system
    """
    print("SWING EVALUATION SYSTEM")
    print("="*50)

    # Initialize evaluator
    evaluator = SwingEvaluator(
        impulse_threshold=75.0,
        correction_threshold=25.0,
        major_threshold=75.0,
        minor_threshold=25.0
    )

    try:
        # Load fractal data
        print("Loading fractal data...")
        evaluator.load_fractal_data()

        # Perform evaluation
        print("Performing swing evaluation...")
        evaluated_df = evaluator.evaluate_swings()

        # Save results
        print("Saving results...")
        output_file = evaluator.save_evaluation_results()

        # Print analysis summary
        evaluator.print_analysis_summary()

        # Create visualizations
        print("Creating comprehensive visualizations...")
        eval_chart, stats_chart = evaluator.create_visualizations()

        # Show sample of evaluated data
        print(f"\nSAMPLE OF EVALUATED DATA (first 10 rows):")
        print("-" * 120)
        sample_cols = ['timestamp', 'type', 'price', 'velocity', 'swing_type',
                      'swing_classification', 'pattern_type', 'swing_strength_index']
        available_cols = [col for col in sample_cols if col in evaluated_df.columns]
        print(evaluated_df[available_cols].head(10).to_string(index=False))

        print(f"\nOutput files generated:")
        print(f"  - Evaluation data: {output_file}")
        print(f"  - Evaluation chart: {eval_chart}")
        print(f"  - Statistics chart: {stats_chart}")
        print(f"\nSwing evaluation completed successfully!")

    except Exception as e:
        print(f"Error in swing evaluation: {e}")
        return False

    return True


if __name__ == "__main__":
    main()