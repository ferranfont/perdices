# Perdices - ES Futures Quantitative Analysis

Advanced multi-layered fractal and master candle detection system for ES (E-mini S&P 500) futures trading analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Data Source

The system analyzes 1-minute ES futures data from the local `data/` folder:
- `es_1min_data_2015_2025.csv` - Full historical 1-min data (2015-2025)
- `es_1D_data.csv` - Daily OHLCV data
- `es_1min_clean_2023_03_01.csv` - Clean intraday data (March 1, 2023)
- `es_1min_clean_2023_03_02.csv` - Clean intraday data (March 2, 2023) **[ACTIVE]**
- `ES_near_tick_data_27_jul_2025.csv` - Tick-level data

## Features

### Core Analysis Pipeline

1. **Fractal Detection** (`quant_stat/find_tops_and_bottoms.py`)
   - ZigZag method (default: 0.1% threshold)
   - Window-based method (configurable)
   - Identifies swing highs and lows with precision

2. **Creek Pattern Detection** (`quant_stat/find_pending_creek.py`)
   - Identifies pending creek formations from fractal data
   - Detects potential reversal zones

3. **Master Candle Analysis** (`quant_stat/find_mastercandle.py`)
   - **Quintuple Factor Scoring System**:
     - **Factor 1** (10%): Breakout candle volume intensity
     - **Factor 2** (15%): Cumulative volume from last swing
     - **Factor 3** (20%): Total range volume analysis
     - **Factor 4** (25%): Range and tail analysis from swing to breakout
     - **Factor 5** (30%): Range and tail analysis of breakout candle
   - Minimum passing score: 50 points
   - Volume confirmation with 60-bar moving average
   - Range multiplier validation: 1.0x minimum
   - Tail rejection: 18% maximum threshold

4. **Advanced Visualization** (`plot_minute_data.py`)
   - Interactive Plotly charts with zoom and pan
   - Multi-layer overlays: price, volume, fractals, creeks, master candles
   - HTML output for detailed analysis

## Usage

### Run Main Analysis

```bash
python main.py
```

### Configuration Parameters (in `main.py`)

```python
# Fractal Detection
FRACTAL_METHOD = 'zigzag'  # or 'window'
CHANGE_PCT = 0.1           # Zigzag threshold

# Window Method
WINDOW_SIZE = 7
CONFIRMATION_PERIODS = 3

# Master Candle Detection (in find_mastercandle.py)
DEFAULT_VOLUME_WINDOW = 60        # Volume avg window
DEFAULT_VOLUME_MULTIPLIER = 1.5   # Volume threshold
DEFAULT_PASSING_SCORE = 50        # Minimum score
```

## Output

The system generates:

### CSV Files (in `outputs/`)
- `fractals_2023_03_01_zigzag_0.1.csv` - Detected fractals
- `fractals_2023_03_01_zigzag_0.1_pending_creeks.csv` - Creek patterns
- Master candle analysis results

### Interactive HTML Reports
- Fractal and creek visualization charts
- Master candle detection reports with scoring breakdown
- Multi-factor analysis overlays

### Console Output
- Data loading and validation status
- Fractal detection summary
- Creek pattern count
- Master candle detection results with scores

## Project Structure

```
perdices/
├── data/                            # Market data files
│   ├── es_1min_data_2015_2025.csv
│   ├── es_1D_data.csv
│   ├── es_1min_clean_2023_03_01.csv
│   ├── es_1min_clean_2023_03_02.csv  ← ACTIVE
│   └── ES_near_tick_data_27_jul_2025.csv
├── quant_stat/                      # Analysis modules
│   ├── find_tops_and_bottoms.py     # Fractal detection
│   ├── find_pending_creek.py        # Creek pattern detection
│   ├── find_mastercandle.py         # Master candle analysis ⭐
│   └── fractal_detector.py          # Core detection logic
├── utils/                           # Data utilities
│   ├── clean_data_and_format_tick_data.py
│   ├── clean_data_minut_format_all_dataframe.py
│   └── clean_data_one_day_data.py
├── charts/                          # Output visualizations
├── outputs/                         # CSV analysis results
├── config.py                        # Central configuration
├── main.py                          # Main pipeline
├── plot_minute_data.py              # Visualization engine
├── plot_chart_volume.py             # Volume charts
├── swing_eval.py                    # Swing evaluation
└── requirements.txt                 # Python dependencies
```

## Master Candle Detection System

The system uses a sophisticated **quintuple factor scoring** approach to identify high-probability master candles at creek breakout levels:

### Factor Breakdown

| Factor | Weight | Analysis Focus |
|--------|--------|----------------|
| Factor 1 | 10% | Breakout candle volume vs. average |
| Factor 2 | 15% | Cumulative volume from last swing |
| Factor 3 | 20% | Total range volume intensity |
| Factor 4 | 25% | Range/tail quality from swing to breakout |
| Factor 5 | 30% | Range/tail quality of breakout candle |

### Scoring Scale

- **50+ points**: Qualifies as master candle
- **70+ points**: High-quality setup
- **80+ points**: Exceptional setup
- **90+ points**: Rare, elite setup

## Recent Updates

- ✅ Quintuple factor scoring system implemented
- ✅ Factor 4: Swing-to-breakout range analysis
- ✅ Factor 5: Breakout candle range/tail validation
- ✅ Weighted scoring with configurable thresholds
- ✅ Interactive HTML reports with factor breakdowns
- ✅ Clean data pipeline for March 2023 intraday analysis
