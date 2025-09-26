# main.py
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from chart_volume import plot_close_and_volume


symbol = 'ES'

# ====================================================
# 📥 CARGA DE DATOS
# ====================================================
directorio = './data'
nombre_fichero = 'ES_near_tick_data_27_jul_2025.csv'

ruta_completa = os.path.join(directorio, nombre_fichero)

print("\n======================== 🔍 df  ===========================")
df = pd.read_csv(ruta_completa)
print('Fichero:', ruta_completa, 'importado')
print(f"Características del Fichero Base: {df.shape}")
print("Columnas disponibles:", df.columns.tolist())

# Normalizar columnas a minúsculas
df.columns = [col.strip().lower() for col in df.columns]

# Crear columnas OHLCV estándar desde los datos disponibles
# Asumiendo que los datos tick tienen una columna de precio
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
print("Columnas numéricas disponibles:", numeric_cols)

if len(numeric_cols) >= 1:
    price_col = numeric_cols[0]  # Primera columna numérica como precio
    print(f"Usando '{price_col}' como columna de precio")

    # Crear columnas OHLCV basadas en el precio
    df['open'] = df[price_col]
    df['high'] = df[price_col]
    df['low'] = df[price_col]
    df['close'] = df[price_col]

    # Si hay segunda columna numérica, usarla como volume
    if len(numeric_cols) >= 2:
        volume_col = numeric_cols[1]
        df['volume'] = df[volume_col]
        print(f"Usando '{volume_col}' como volumen")
    else:
        df['volume'] = 1  # Volumen por defecto si no hay datos
        print("Creando volumen por defecto = 1")
else:
    print("Error: No se encontraron columnas numéricas")

# Verificar si existe columna de fecha o usar índice
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')
elif 'datetime' in df.columns:
    df['date'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('date')
else:
    # Si no hay columna de fecha, usar el índice como timestamp
    print("No se encontró columna de fecha, usando índice como timestamp")
    df.index = pd.date_range(start='2025-07-27', periods=len(df), freq='1min')
    df.index.name = 'date'

# 🔁 Resample a velas diarias
print("Columnas después de normalización:", df.columns.tolist())

# Mapear columnas disponibles a formato OHLCV estándar
column_mapping = {}
available_cols = df.columns.tolist()

# Buscar columnas de precio
for col in available_cols:
    if 'open' in col.lower():
        column_mapping['open'] = col
    elif 'high' in col.lower():
        column_mapping['high'] = col
    elif 'low' in col.lower():
        column_mapping['low'] = col
    elif 'close' in col.lower() or 'price' in col.lower():
        column_mapping['close'] = col
    elif 'vol' in col.lower():
        column_mapping['volume'] = col

print("Mapeo de columnas:", column_mapping)

# Si solo hay una columna de precio, usar para todos los valores OHLC
if len([col for col in column_mapping.keys() if col in ['open', 'high', 'low', 'close']]) == 0:
    # Buscar cualquier columna numérica para usar como precio
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        price_col = numeric_cols[0]
        print(f"Usando columna '{price_col}' como precio único para OHLC")
        column_mapping = {
            'open': price_col,
            'high': price_col,
            'low': price_col,
            'close': price_col
        }
        if len(numeric_cols) > 1:
            column_mapping['volume'] = numeric_cols[1]

# Crear diccionario de agregación basado en columnas disponibles
agg_dict = {}
if 'open' in column_mapping:
    agg_dict[column_mapping['open']] = 'first'
if 'high' in column_mapping:
    agg_dict[column_mapping['high']] = 'max'
if 'low' in column_mapping:
    agg_dict[column_mapping['low']] = 'min'
if 'close' in column_mapping:
    agg_dict[column_mapping['close']] = 'last'
if 'volume' in column_mapping:
    agg_dict[column_mapping['volume']] = 'sum'

print("Diccionario de agregación:", agg_dict)

df_daily = df.resample('1D').agg(agg_dict).dropna()

# Renombrar columnas al formato estándar OHLCV
rename_dict = {v: k for k, v in column_mapping.items()}
df_daily = df_daily.rename(columns=rename_dict)

# Reset index para usar 'date' como columna
df_daily = df_daily.reset_index()

print(df_daily.head())

# Ejecutar gráfico
timeframe = '1D'
plot_close_and_volume(symbol, timeframe, df_daily)
