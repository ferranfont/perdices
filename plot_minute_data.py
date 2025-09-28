import os
import webbrowser
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px1
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config import CHART_WIDTH, CHART_HEIGHT, CHART_TEMPLATE, get_chart_path, DATA_DIR, SYMBOL

# ⚙️ Configuration =========================================

CSV_FILE = 'es_1min_data_2015_2025.csv'  # CSV file to plot
FRACTAL_CSV_FILE = 'fractals_2023_03_01_zigzag_0.1.csv'  # Fractal detection results

#  ==========================================================


def load_fractals_data(method='zigzag', change_pct=0.05, window_size=7, confirmation_periods=3):
    """
    Load fractals data from the outputs folder using the configured FRACTAL_CSV_FILE
    """
    try:
        # Use the configured fractal file instead of generating dynamically
        fractal_file = FRACTAL_CSV_FILE
        fractals_path = os.path.join('outputs', fractal_file)

        if os.path.exists(fractals_path):
            df_fractals = pd.read_csv(fractals_path)
            df_fractals['timestamp'] = pd.to_datetime(df_fractals['timestamp'])
            print(f"Loaded {len(df_fractals)} fractals from {fractal_file}")
            return df_fractals, fractal_file
        else:
            print(f"Fractals file not found: {fractals_path}")
            return None, fractal_file
    except Exception as e:
        print(f"Error loading fractals: {e}")
        return None, fractal_file

def load_pending_creek_data():
    """
    Load pending creek data from the outputs folder
    """
    try:
        creek_file = FRACTAL_CSV_FILE.replace('.csv', '_pending_creeks.csv')
        creek_path = os.path.join('outputs', creek_file)

        if os.path.exists(creek_path):
            df_creeks = pd.read_csv(creek_path)
            df_creeks['timestamp'] = pd.to_datetime(df_creeks['timestamp'])
            print(f"Loaded {len(df_creeks)} pending creeks from {creek_file}")
            return df_creeks
        else:
            print(f"No pending creek file found: {creek_path}")
            return None
    except Exception as e:
        print(f"Error loading pending creeks: {e}")
        return None

def plot_minute_data(symbol, timeframe, df, fractal_method='zigzag', change_pct=0.05, window_size=7, confirmation_periods=3):
    """
    Función especializada para graficar datos de minutos con etiquetas de hora en el eje X
    Incluye fractales (picos y valles) como puntos azules
    """
    html_path = get_chart_path(symbol, timeframe)

    df = df.rename(columns=str.lower)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Load fractals data with dynamic filename
    df_fractals, fractal_file = load_fractals_data(fractal_method, change_pct, window_size, confirmation_periods)

    # Load pending creek data
    df_creeks = load_pending_creek_data()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.03,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Gráfico de velas (candlestick) con outline negro semi-transparente
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='rgba(0,0,0,0.5)',     # Outline negro con alpha 0.5 para velas alcistas
        decreasing_line_color='rgba(0,0,0,0.5)',     # Outline negro con alpha 0.5 para velas bajistas
        increasing_fillcolor='rgba(0,255,0,0.5)',    # Verde lima con alpha 0.5 para velas alcistas
        decreasing_fillcolor='rgba(255,0,0,0.5)',    # Rojo con alpha 0.5 para velas bajistas
        line=dict(width=1),
        name='OHLC'
    ), row=1, col=1)

    # Plot fractals (tops and bottoms) with color coding by swing size
    if df_fractals is not None and len(df_fractals) > 0:
        # Separate tops and bottoms for better visualization
        tops = df_fractals[df_fractals['type'] == 'TOP']
        bottoms = df_fractals[df_fractals['type'] == 'BOTTOM']

        # Add blue line connecting all fractals
        fig.add_trace(go.Scatter(
            x=df_fractals['timestamp'],
            y=df_fractals['price'],
            mode='lines',
            line=dict(
                color='blue',
                width=1
            ),
            name='Fractal Line',
            hovertemplate='Fractal Line<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)

        # Add tops with color coding based on swing size
        if len(tops) > 0:
            # Separate big swings from others for tops
            big_tops = tops[tops.get('swing_size', 'noise') == 'big'] if 'swing_size' in tops.columns else tops.iloc[0:0]
            other_tops = tops[tops.get('swing_size', 'noise') != 'big'] if 'swing_size' in tops.columns else tops

            # Add purple dots for big swing tops
            if len(big_tops) > 0:
                fig.add_trace(go.Scatter(
                    x=big_tops['timestamp'],
                    y=big_tops['price'],
                    mode='markers',
                    marker=dict(
                        color='purple',
                        size=10,
                        symbol='circle',
                        line=dict(width=1, color='darkmagenta')
                    ),
                    name='Big Tops',
                    hovertemplate='Big Top<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Add blue dots for small/noise swing tops
            if len(other_tops) > 0:
                fig.add_trace(go.Scatter(
                    x=other_tops['timestamp'],
                    y=other_tops['price'],
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=6,
                        symbol='circle',
                        line=dict(width=1, color='darkblue')
                    ),
                    name='Tops',
                    hovertemplate='Top<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

        # Add bottoms with color coding based on swing size
        if len(bottoms) > 0:
            # Separate big swings from others for bottoms
            big_bottoms = bottoms[bottoms.get('swing_size', 'noise') == 'big'] if 'swing_size' in bottoms.columns else bottoms.iloc[0:0]
            other_bottoms = bottoms[bottoms.get('swing_size', 'noise') != 'big'] if 'swing_size' in bottoms.columns else bottoms

            # Add purple dots for big swing bottoms
            if len(big_bottoms) > 0:
                fig.add_trace(go.Scatter(
                    x=big_bottoms['timestamp'],
                    y=big_bottoms['price'],
                    mode='markers',
                    marker=dict(
                        color='purple',
                        size=10,
                        symbol='circle',
                        line=dict(width=1, color='darkmagenta')
                    ),
                    name='Big Bottoms',
                    hovertemplate='Big Bottom<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

            # Add blue dots for small/noise swing bottoms
            if len(other_bottoms) > 0:
                fig.add_trace(go.Scatter(
                    x=other_bottoms['timestamp'],
                    y=other_bottoms['price'],
                    mode='markers',
                    marker=dict(
                        color='blue',
                        size=6,
                        symbol='circle',
                        line=dict(width=1, color='darkblue')
                    ),
                    name='Bottoms',
                    hovertemplate='Bottom<br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)

    # Plot pending creek lines (green horizontal lines after big downtrends)
    if df_creeks is not None and len(df_creeks) > 0:
        print(f"Adding {len(df_creeks)} pending creek lines to chart")

        for _, creek in df_creeks.iterrows():
            # Calculate end time for the 60-bar line extension
            creek_start_time = creek['timestamp']

            # Ensure timezone compatibility for comparison
            if hasattr(creek_start_time, 'tz_localize'):
                creek_start_time = creek_start_time.tz_localize(None) if creek_start_time.tz is not None else creek_start_time

            # Make sure df['date'] is also timezone-naive for comparison
            df_date_naive = df['date'].dt.tz_localize(None) if df['date'].dt.tz is not None else df['date']

            # Find the index in the main dataframe for time calculation
            creek_mask = df_date_naive >= creek_start_time
            if creek_mask.any():
                start_idx = df[creek_mask].index[0]

                # Check for crossover: find first close price above creek level
                crossover_idx = None
                max_extension = min(start_idx + 60, len(df) - 1)  # Maximum 60 bars

                for i in range(start_idx, max_extension + 1):
                    if df.iloc[i]['close'] > creek['price']:
                        crossover_idx = i
                        break

                # Determine end index based on crossover logic
                if crossover_idx is not None:
                    # Extend 2 bars after crossover (but not beyond data)
                    end_idx = min(crossover_idx + 2, len(df) - 1)
                else:
                    # No crossover found, extend full 60 bars
                    end_idx = max_extension

                creek_end_time = df.iloc[end_idx]['date']

                # Add horizontal line for pending creek
                fig.add_trace(go.Scatter(
                    x=[creek_start_time, creek_end_time],
                    y=[creek['price'], creek['price']],
                    mode='lines',
                    line=dict(
                        color='green',
                        width=2,
                        dash='solid'
                    ),
                    name=f'Pending Creek ${creek["price"]}',
                    hovertemplate=f'Pending Creek<br>Price: ${creek["price"]:.2f}<br>Strength: {creek["strength"]}<extra></extra>',
                    showlegend=False
                ), row=1, col=1)

                # Add green triangle-up marker at crossover point if crossover occurred
                if crossover_idx is not None:
                    crossover_time = df.iloc[crossover_idx]['date']
                    crossover_close = df.iloc[crossover_idx]['close']

                    fig.add_trace(go.Scatter(
                        x=[crossover_time],
                        y=[crossover_close],
                        mode='markers',
                        marker=dict(
                            color='lime',
                            size=15,
                            symbol='triangle-up',
                            line=dict(width=2, color='green')
                        ),
                        name=f'Creek Crossover',
                        hovertemplate=f'Crossover<br>Time: %{{x}}<br>Close: ${crossover_close:.2f}<br>Creek: ${creek["price"]:.2f}<extra></extra>',
                        showlegend=False
                    ), row=1, col=1)

                # Add a small marker at the creek point
                fig.add_trace(go.Scatter(
                    x=[creek_start_time],
                    y=[creek['price']],
                    mode='markers',
                    marker=dict(
                        color='green',
                        size=12,
                        symbol='diamond',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Creek Point',
                    hovertemplate=f'Pending Creek Start<br>Time: %{{x}}<br>Price: ${creek["price"]:.2f}<br>Strength: {creek["strength"]}<extra></extra>',
                    showlegend=False
                ), row=1, col=1)

    # Barras de volumen
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        marker_color='royalblue',
        marker_line_color='blue',
        marker_line_width=0.4,
        opacity=0.95,
        name='Volumen'
    ), row=2, col=1)

    fig.update_layout(
        dragmode='pan',
        title=f'{symbol}_{timeframe} - Perdices - {fractal_file}',
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12, color="black"),
        plot_bgcolor='white',  # Light grey with alpha 0.1
        paper_bgcolor='white',
        showlegend=False,
        template='plotly_white',
        xaxis=dict(
            type='date',
            tickformat="%H:%M",  # Solo mostrar hora:minuto
            tickangle=0,
            showgrid=False,  # Sin grid vertical
            linecolor='black',
            linewidth=1,
            range=[df['date'].min(), df['date'].max()],
            # Mostrar ticks cada hora
            dtick=3600000,  # 1 hora en milisegundos
            rangeslider=dict(visible=False)  # Ocultar el range slider
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',  # Gris muy pálido, casi transparente
            gridwidth=1,
            linecolor='black',
            linewidth=1
        ),
        xaxis2=dict(
            type='date',
            tickformat="%H:%M",  # Solo mostrar hora:minuto
            tickangle=45,
            showgrid=False,  # Sin grid vertical
            linecolor='black',
            linewidth=1,
            range=[df['date'].min(), df['date'].max()],
            # Mostrar ticks cada hora
            dtick=3600000  # 1 hora en milisegundos
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',  # Gris muy pálido, casi transparente
            gridwidth=1,
            linecolor='black',
            linewidth=1
        ),
    )


    fig.write_html(html_path, config={
        "scrollZoom": True,
        "displayModeBar": True,  # Mostrar barra de navegación
        "staticPlot": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "chart",
            "height": 500,
            "width": 700,
            "scale": 1
        }
    })
    print(f"Gráfico de minutos guardado como HTML: '{html_path}'")

    webbrowser.open('file://' + os.path.realpath(html_path))


if __name__ == "__main__":
    # Configuración para ejecución directa
    TARGET_DATE = '2023-03-01'  # Cambiar esta fecha según necesidad
    symbol = SYMBOL

    # ====================================================
    # CARGA DE DATOS
    # ====================================================
    directorio = str(DATA_DIR)
    ruta_completa = os.path.join(directorio, CSV_FILE)

    print(f"\n======================== Extrayendo datos del {TARGET_DATE} ===========================")
    print(f"CSV File: {CSV_FILE}")
    df = pd.read_csv(ruta_completa)
    print('Fichero:', ruta_completa, 'importado')
    print(f"Características del Fichero Base: {df.shape}")

    # Normalizar columnas a minúsculas y renombrar 'volumen' a 'volume'
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns={'volumen': 'volume'})

    # Asegurar formato datetime (timezone-naive, como en main.py)
    df['date'] = pd.to_datetime(df['date'])
    # Convert to timezone-naive if timezone-aware
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)

    # Filtrar datos solo para la fecha objetivo
    target_date_start = pd.to_datetime(TARGET_DATE)
    target_date_end = target_date_start + pd.Timedelta(days=1)

    df_filtered = df[(df['date'] >= target_date_start) & (df['date'] < target_date_end)].copy()

    print(f"Datos encontrados para {TARGET_DATE}: {len(df_filtered)} registros")

    if len(df_filtered) > 0:
        # Mostrar estadísticas básicas
        print("\nEstadísticas del día:")
        print(f"Open: {df_filtered['open'].iloc[0]:.2f}")
        print(f"High: {df_filtered['high'].max():.2f}")
        print(f"Low: {df_filtered['low'].min():.2f}")
        print(f"Close: {df_filtered['close'].iloc[-1]:.2f}")
        print(f"Volume total: {df_filtered['volume'].sum():,.0f}")

        # Crear gráfico con los datos del día específico
        print(f"\nGenerando gráfico para {TARGET_DATE}...")
        timeframe = f'1min_{TARGET_DATE}'
        plot_minute_data(symbol, timeframe, df_filtered, fractal_method='zigzag', change_pct=0.1)

    else:
        print(f"No se encontraron datos para la fecha {TARGET_DATE}")
        print("Verifique que la fecha esté disponible en el archivo de datos.")