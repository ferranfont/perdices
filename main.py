import pandas as pd
import os
import sys
from pathlib import Path
from config import DATA_DIR, SYMBOL
from plot_minute_data import plot_minute_data

# ⚙️ Configuration =========================================

# Fractal Detection Method: 'zigzag' or 'window'
FRACTAL_METHOD = 'zigzag'  # Change to 'window' to test window-based detection

# Zigzag Method Parameters
CHANGE_PCT = 0.1 # Minimum percentage change for zigzag fractal detection

# Window Method Parameters
WINDOW_SIZE = 7 # Window size for window-based fractal detection
CONFIRMATION_PERIODS = 3  # Confirmation periods for window method

# ==========================================================


# Import fractal detection module
sys.path.append(str(Path(__file__).parent / 'quant_stat'))
from find_tops_and_bottoms import main as fractal_analysis

# ====================================================
# LEER ARCHIVO ES 1 MIN DATA 2023-03-01
# ====================================================
filename = 'es_1min_clean_2023_03_02.csv'
file_path = os.path.join(str(DATA_DIR), filename)

print(f"\n======================== Leyendo archivo: {filename} ===========================")

# Verificar si el archivo existe
if os.path.exists(file_path):
    # Leer el archivo CSV
    df = pd.read_csv(file_path)

    print(f"Archivo leído exitosamente: {file_path}")
    print(f"Número de registros: {len(df)}")
    print(f"Columnas: {list(df.columns)}")

    # Mostrar información básica
    print(f"\nInformación del DataFrame:")
    print(df.info())

    print(f"\nDataframe registros:")
    print(df.head(),"\n")
    print(df.tail())

    # Si hay columna de fecha, mostrar rango temporal
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"\nRango temporal:")
        print(f"Desde: {df['date'].min()}")
        print(f"Hasta: {df['date'].max()}")

    # Si hay columna dow (day of week), convertir números a nombres
    if 'dow' in df.columns:
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['dow'] = df['dow'].map(day_names)
        print(f"\nDía de la semana: {df['dow'].iloc[0]}")

    # Mostrar muestra actualizada
    print(f"\nMuestra de datos con días nombrados:")
    print(df.head())

    # Run quantitative fractal analysis (automatically generates plot)
    print(f"\n=== RUNNING QUANTITATIVE FRACTAL ANALYSIS ({FRACTAL_METHOD.upper()} METHOD) ===")
    fractal_analysis(FRACTAL_METHOD, CHANGE_PCT, WINDOW_SIZE, CONFIRMATION_PERIODS)

    # Run master candle detection analysis
    print(f"\n=== RUNNING MASTER CANDLE DETECTION ANALYSIS ===")
    try:
        # Import master candle detection
        from find_mastercandle import analyze_mastercandles

        # Load pending creek data
        import os
        creek_file = f'fractals_2023_03_01_zigzag_{CHANGE_PCT}_pending_creeks.csv'
        creek_path = os.path.join('outputs', creek_file)

        # Load fractals data for Factor 3 calculation
        fractals_file = f'fractals_2023_03_01_zigzag_{CHANGE_PCT}.csv'
        fractals_path = os.path.join('outputs', fractals_file)
        fractals_data = None
        if os.path.exists(fractals_path):
            fractals_data = pd.read_csv(fractals_path)
            fractals_data['timestamp'] = pd.to_datetime(fractals_data['timestamp'])

        if os.path.exists(creek_path):
            creek_data = pd.read_csv(creek_path)
            creek_data['timestamp'] = pd.to_datetime(creek_data['timestamp'])

            # Prepare price data with proper datetime
            price_data = df.copy()
            if 'time' in price_data.columns:
                price_data['date'] = pd.to_datetime(price_data['date'].astype(str) + ' ' + price_data['time'])
            else:
                price_data['date'] = pd.to_datetime(price_data['date'])

            # Run master candle analysis with triple factor scoring
            master_candles, html_report_path = analyze_mastercandles(
                price_data=price_data,
                creek_data=creek_data,
                fractals_data=fractals_data,
                volume_window=60,
                volume_multiplier=2.0,
                passing_score=12  # Now using quadruple factor scoring (volume + cumulative + longterm + range/tail)
            )

            print(f"Master candle analysis completed. Found {len(master_candles)} master candles.")
            if html_report_path:
                print(f"HTML report saved: {html_report_path}")
        else:
            print(f"Pending creek data not found: {creek_path}")
            print("Master candle analysis skipped.")

    except Exception as e:
        print(f"Error in master candle analysis: {e}")
        print("Continuing with remaining analysis...")

    # Also generate direct plot from main.py
    print(f"\n=== GENERATING ADDITIONAL PLOT FROM MAIN.PY ===")
    # Para el gráfico, necesitamos datetime
    if 'time' in df.columns:
        df_for_plot = df.copy()
        df_for_plot['date'] = pd.to_datetime(df_for_plot['date'].astype(str) + ' ' + df_for_plot['time'])
    else:
        df_for_plot = df.copy()

    timeframe = 'main_2023-03-01'

    # Get master candles for plotting if available
    master_candles_for_plot = None
    try:
        # Try to get master candles from the previous analysis
        if 'master_candles' in locals():
            master_candles_for_plot = master_candles
    except:
        pass

    plot_minute_data(SYMBOL, timeframe, df_for_plot, fractal_method=FRACTAL_METHOD,
                     change_pct=CHANGE_PCT, window_size=WINDOW_SIZE, confirmation_periods=CONFIRMATION_PERIODS,
                     master_candles=master_candles_for_plot)

else:
    print(f"Error: No se encontró el archivo {file_path}")
    print(f"Verifique que el archivo existe en la carpeta: {DATA_DIR}")

    # Mostrar archivos disponibles en el directorio
    print(f"\nArchivos disponibles en {DATA_DIR}:")
    try:
        files = [f for f in os.listdir(str(DATA_DIR)) if f.endswith('.csv')]
        for file in files:
            print(f"  - {file}")
    except:
        print("  No se pudo acceder al directorio")