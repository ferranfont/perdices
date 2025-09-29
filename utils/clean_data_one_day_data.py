import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, SYMBOL
from plot_minute_data import plot_minute_data

# ====================================================
# 📥 CONFIGURACIÓN
# ====================================================

symbol = SYMBOL
TARGET_DATE = '2023-03-02'  # Formato: YYYY-MM-DD

# ====================================================
# 📥 CARGA DE DATOS
# ====================================================
directorio = str(DATA_DIR)
nombre_fichero = 'es_1min_data_2015_2025.csv'
ruta_completa = os.path.join(directorio, nombre_fichero)

print(f"\n======================== Extrayendo datos del {TARGET_DATE} ===========================")
df = pd.read_csv(ruta_completa)
print('Fichero:', ruta_completa, 'importado')
print(f"Características del Fichero Base: {df.shape}")

# Normalizar columnas a minúsculas y renombrar 'volumen' a 'volume'
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={'volumen': 'volume'})

# Eliminar columnas no deseadas si existen
columns_to_drop = ['long_level', 'short_level', 'long_stop', 'short_stop']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_cols_to_drop:
    df = df.drop(columns=existing_cols_to_drop)
    print(f"Columnas eliminadas: {existing_cols_to_drop}")

# Asegurar formato datetime con zona UTC
df['date'] = pd.to_datetime(df['date'], utc=True)

# Filtrar datos solo para la fecha objetivo
target_date_start = pd.to_datetime(TARGET_DATE, utc=True)
target_date_end = target_date_start + pd.Timedelta(days=1)

df_filtered = df[(df['date'] >= target_date_start) & (df['date'] < target_date_end)].copy()

if len(df_filtered) > 0:
    # Separar fecha y hora
    df_filtered['time'] = df_filtered['date'].dt.strftime('%H:%M:%S')
    df_filtered['date'] = df_filtered['date'].dt.date
    df_filtered['dow'] = pd.to_datetime(df_filtered['date']).dt.dayofweek  # 0=Monday, 6=Sunday

    # Reordenar columnas: date, time, dow, open, high, low, close, volume
    column_order = ['date', 'time', 'dow', 'open', 'high', 'low', 'close', 'volume']
    # Solo incluir columnas que existen
    available_columns = [col for col in column_order if col in df_filtered.columns]
    df_filtered = df_filtered[available_columns]

    print(f"Columnas finales: {list(df_filtered.columns)}")
    print(f"Muestra de datos limpios:")
    print(df_filtered.head())

    print(f"Datos encontrados para {TARGET_DATE}: {len(df_filtered)} registros")

    # Crear nombre de archivo limpio para el día específico
    clean_filename = f'es_1min_clean_{TARGET_DATE.replace("-", "_")}.csv'
    clean_path = os.path.join(directorio, clean_filename)

    # Guardar CSV con datos limpios del día específico
    df_filtered.to_csv(clean_path, index=False)

    print(f"Archivo limpio creado: {clean_path}")
    print(f"Registros guardados: {len(df_filtered)}")
    print(f"Dia de la semana: {df_filtered['dow'].iloc[0]} (0=Lunes, 6=Domingo)")

    # Mostrar estadísticas básicas
    if 'open' in df_filtered.columns:
        print("\nEstadisticas del dia:")
        print(f"Open: {df_filtered['open'].iloc[0]:.2f}")
        print(f"High: {df_filtered['high'].max():.2f}")
        print(f"Low: {df_filtered['low'].min():.2f}")
        print(f"Close: {df_filtered['close'].iloc[-1]:.2f}")
        if 'volume' in df_filtered.columns:
            print(f"Volume total: {df_filtered['volume'].sum():,.0f}")

        # Para el gráfico, necesitamos reconvertir date y time a datetime
        df_for_plot = df_filtered.copy()
        df_for_plot['date'] = pd.to_datetime(df_for_plot['date'].astype(str) + ' ' + df_for_plot['time'])

        # Crear gráfico con los datos del día específico
        print(f"\nGenerando grafico para {TARGET_DATE}...")
        timeframe = f'1min_clean_{TARGET_DATE}'
        plot_minute_data(symbol, timeframe, df_for_plot)

else:
    print(f"No se encontraron datos para la fecha {TARGET_DATE}")
    print("Verifique que la fecha este disponible en el archivo de datos.")