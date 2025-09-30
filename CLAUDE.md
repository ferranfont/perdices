# CLAUDE.md - Perdices Trading Analysis Project

## Project Overview

This is a quantitative trading analysis project focused on ES (E-mini S&P 500) futures. The system performs multi-layered fractal analysis, creek detection, and master candle identification using high-frequency 1-minute data.

## Current Configuration

### Data Source
- **Location**: `data/` folder (local)
- **Active file**: `es_1min_clean_2023_03_02.csv` (see `main.py:30`)
- **Configuration**: `config.py` manages all paths and settings using `DATA_DIR`
- **Available datasets**:
  - `es_1min_data_2015_2025.csv` - Full historical 1-min data (358 MB)
  - `es_1D_data.csv` - Daily data (207 KB)
  - `es_1min_clean_2023_03_01.csv` - Clean 1-min data (March 1, 2023)
  - `es_1min_clean_2023_03_02.csv` - Clean 1-min data (March 2, 2023)
  - `ES_near_tick_data_27_jul_2025.csv` - Tick data

### Analysis Pipeline

The project runs a multi-stage analysis pipeline:

1. **Fractal Detection** (`quant_stat/find_tops_and_bottoms.py`)
   - Method: ZigZag or Window-based
   - Default: ZigZag with 0.1% change threshold
   - Identifies swing highs and lows
   - Outputs: `fractals_2023_03_01_zigzag_0.1.csv`

2. **Creek Detection** (`quant_stat/find_pending_creek.py`)
   - Identifies pending creek patterns from fractals
   - Outputs: `fractals_2023_03_01_zigzag_0.1_pending_creeks.csv`

3. **Master Candle Analysis** (`quant_stat/find_mastercandle.py`)
   - **Quintuple Factor Scoring System**:
     - Factor 1 (10%): Breakout candle volume
     - Factor 2 (15%): Cumulative volume from last swing
     - Factor 3 (20%): Total range volume analysis
     - Factor 4 (25%): Range/tail analysis from swing to breakout
     - Factor 5 (30%): Range/tail analysis of breakout candle
   - **Default parameters**:
     - Volume window: 60 bars (`DEFAULT_VOLUME_WINDOW:11`)
     - Volume multiplier: 1.5x
     - Passing score: 50 (sum of all factor scores)
     - Range multiplier: 1.0x
     - Tail threshold: 18% max
   - Outputs: HTML report with interactive charts

4. **Visualization** (`plot_minute_data.py`)
   - Generates interactive Plotly charts
   - Displays price, volume, fractals, creeks, and master candles
   - Outputs: HTML files in `charts/` folder

### Key Configuration Files

- `config.py` - Central configuration (paths, timeframes, chart settings)
- `main.py` - Main execution pipeline with configurable parameters
- `.env` - Environment variables (if needed)

### Current Analysis Focus

The project is configured to analyze March 2, 2023 data (`main.py:30`) looking for:
- High-quality master candle breakouts at creek levels
- Volume confirmation with quintuple factor validation
- Optimal entry points at fractal swing levels