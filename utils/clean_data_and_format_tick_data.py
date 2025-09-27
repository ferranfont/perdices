# main.py
import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from plot_chart_volume import plot_close_and_volume


symbol = 'ES'

# ====================================================
# 📥 CARGA DE DATOS
# ====================================================
directorio = './data'
nombre_fichero = 'ES_near_tick_data_27_jul_2025.csv'

ruta_completa = os.path.join(directorio, nombre_fichero)

print("\n======================== 🔍 df  ===========================")
# Leer CSV con formato europeo (separador ; y decimal ,)
df = pd.read_csv(ruta_completa, sep=';', decimal=',',
                 names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
print('Fichero:', ruta_completa, 'importado')
print(f"Características del Fichero Base: {df.shape}")
print("Columnas disponibles:", df.columns.tolist())

# Convertir datetime a formato pandas y separar fecha y hora
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M', utc=True)
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time

# Establecer datetime como índice
df = df.set_index('datetime')

print("Primeras filas del DataFrame:")
print(df.tail(20))

# 🔁 Resample a velas diarias
print("Columnas finales:", df.columns.tolist())

df_daily = df.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Reset index para usar 'datetime' como columna 'date'
df_daily = df_daily.reset_index()
df_daily = df_daily.rename(columns={'datetime': 'date'})

print(df_daily.head())

# Ejecutar gráfico
timeframe = '1D'
plot_close_and_volume(symbol, timeframe, df_daily)
