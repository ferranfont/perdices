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
filename = 'es_1min_clean_2023_03_01.csv'
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

    # Run quantitative fractal analysis
    print(f"\n=== RUNNING QUANTITATIVE FRACTAL ANALYSIS ({FRACTAL_METHOD.upper()} METHOD) ===")
    fractals, pending_creeks = fractal_analysis(FRACTAL_METHOD, CHANGE_PCT, WINDOW_SIZE, CONFIRMATION_PERIODS)

    # Generate plot with fractal and pending creek data
    print(f"\n=== GENERATING CHART FROM MAIN.PY ===")
    # Para el gráfico, necesitamos datetime
    if 'time' in df.columns:
        df_for_plot = df.copy()
        df_for_plot['date'] = pd.to_datetime(df_for_plot['date'].astype(str) + ' ' + df_for_plot['time'])
    else:
        df_for_plot = df.copy()

    timeframe = 'main_2023-03-01'
    plot_minute_data(SYMBOL, timeframe, df_for_plot, fractal_method=FRACTAL_METHOD,
                     change_pct=CHANGE_PCT, window_size=WINDOW_SIZE, confirmation_periods=CONFIRMATION_PERIODS)

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