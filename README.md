# Perdices - ES Futures Analysis

Trading analysis script for ES (E-mini S&P 500) futures data.

## Installation

```bash
pip install -r requirements.txt
```

## Data Source

The script reads 1-minute ES futures data from the local `data/` folder:
- `es_1min_data_2015_2025.csv` - High frequency ES data (2015-2025)
- `es_1D_data.csv` - Daily ES data

## Features

- **Data Processing**: Loads and processes ES futures data
- **Timeframe Conversion**: Resamples 1-minute data to daily candles
- **Visualization**: Creates price and volume charts using `chart_volume.py`
- **Technical Analysis**: Script designed for finding tops and bottoms in ES futures

## Usage

Run the main analysis:

```bash
python main.py
```

## Output

The script generates:
- Daily OHLCV data from 1-minute bars
- Price and volume charts
- Console output showing data characteristics

## Project Structure

```
perdices/
├── data/                    # Local data files
│   ├── es_1min_data_2015_2025.csv
│   └── es_1D_data.csv
├── main.py                  # Main analysis script
├── chart_volume.py          # Charting functionality
├── requirements.txt         # Python dependencies
└── README.md               # This file
```
