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
FRACTAL_CSV_FILE = 'fractals_2023_03_02_zigzag_0.1.csv'  # Fractal detection results

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

def load_master_candle_data():
    """
    Load master candle detection results from outputs folder
    """
    try:
        import glob
        import json

        # Look for the most recent master candle report
        pattern = os.path.join('outputs', 'master_candle_report_*.html')
        report_files = glob.glob(pattern)

        if not report_files:
            print("No master candle reports found")
            return None

        # Get the most recent report file
        latest_report = max(report_files, key=os.path.getctime)
        print(f"Found master candle report: {latest_report}")

        # For now, we'll return an empty list since we need to implement
        # a way to extract data from the HTML or store it separately
        # This is a placeholder - we'll need to modify the master candle detector
        # to also save data in a machine-readable format
        return []

    except Exception as e:
        print(f"Error loading master candle data: {e}")
        return None

def plot_minute_data(symbol, timeframe, df, fractal_method='zigzag', change_pct=0.05, window_size=7, confirmation_periods=3, master_candles=None):
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

        # Add yellow stars at last bottom before each creek crossover
        if len(bottoms) > 0 and master_candles is not None and len(master_candles) > 0:
            # For each master candle (creek crossover), find the last bottom before it
            for candle in master_candles:
                crossover_time = pd.to_datetime(candle.get('crossover_time'))

                # Find bottoms that occurred before this crossover
                bottoms_before_crossover = bottoms[pd.to_datetime(bottoms['timestamp']) < crossover_time]

                if len(bottoms_before_crossover) > 0:
                    # Only show yellow star if cumulative_volume_score (Factor 2) > 5
                    cumulative_volume_score = candle.get('cumulative_volume_score', 0)
                    if cumulative_volume_score > 5:
                        # Get the last (most recent) bottom before this crossover
                        last_bottom_before = bottoms_before_crossover.iloc[-1]

                        # Position star very close to the bottom price
                        star_y_position = last_bottom_before['price'] - 0.5  # 0.5 points below the low

                        # Add yellow star marker (same size as Factor 1)
                        fig.add_trace(go.Scatter(
                            x=[last_bottom_before['timestamp']],
                            y=[star_y_position],
                            mode='markers',
                            marker=dict(
                                color='yellow',
                                size=12,  # Same size as Factor 1 stars
                                symbol='star',
                                line=dict(width=1, color='orange')
                            ),
                            name='Last Bottom Before Crossover',
                            hovertemplate=f'Last Bottom Before Crossover<br>Time: %{{x}}<br>Price: ${last_bottom_before["price"]:.2f}<br>Creek: ${candle.get("creek_price", "N/A"):.2f}<extra></extra>',
                            showlegend=False
                        ), row=1, col=1)

        # Add red/orange stars at big bottom (Factor 3) when longterm_volume_score > 5
        if len(bottoms) > 0 and master_candles is not None and len(master_candles) > 0:
            # For each master candle, find the big bottom and check Factor 3 score
            for candle in master_candles:
                longterm_volume_score = candle.get('longterm_volume_score', 0)
                if longterm_volume_score > 5:
                    crossover_time = pd.to_datetime(candle.get('crossover_time'))

                    # Find big bottoms before this crossover
                    big_bottoms_before = bottoms[
                        (bottoms['swing_size'] == 'big') &
                        (pd.to_datetime(bottoms['timestamp']) < crossover_time)
                    ].sort_values('timestamp')

                    if len(big_bottoms_before) > 0:
                        # Get the last big bottom before crossover
                        last_big_bottom = big_bottoms_before.iloc[-1]

                        # Position star very close to the big bottom price
                        star_y_position = last_big_bottom['price'] - 0.5  # 0.5 points below the low

                        # Add gold star marker (Factor 3) - same color as Factor 1
                        fig.add_trace(go.Scatter(
                            x=[last_big_bottom['timestamp']],
                            y=[star_y_position],
                            mode='markers',
                            marker=dict(
                                color='gold',
                                size=12,  # Same size as other factor stars
                                symbol='star',
                                line=dict(width=1, color='orange')
                            ),
                            name='Factor 3 - Big Bottom',
                            hovertemplate=f'Factor 3 - Long-term Volume<br>Time: %{{x}}<br>Big Bottom: ${last_big_bottom["price"]:.2f}<br>LT Vol Score: {longterm_volume_score}/10<extra></extra>',
                            showlegend=False
                        ), row=1, col=1)

    # Add markers for Factor 4 and Factor 5 qualifying candles
    if master_candles is not None and len(master_candles) > 0:
        for candle in master_candles:
            range_score = candle.get('range_score', 0)
            qualifying_candles = candle.get('qualifying_candles', [])
            crossover_range_score = candle.get('crossover_range_score', 0)

            # Factor 4: Add yellow rectangle markers on qualifying swing candles
            if len(qualifying_candles) > 0:
                for qual_candle in qualifying_candles:
                    qual_time = pd.to_datetime(qual_candle['time'])
                    qual_high = qual_candle['high']

                    # Position rectangle just a tick above the high of the qualifying candle
                    rect_y_position = qual_high + 0.5  # Add small tick margin above the high

                    # Add yellow rectangle marker (Factor 4) at qualifying candle high
                    fig.add_trace(go.Scatter(
                        x=[qual_time],
                        y=[rect_y_position],
                        mode='markers',
                        marker=dict(
                            color='yellow',
                            size=12,  # Same size as stars
                            symbol='square'
                        ),
                        name='Factor 4 - Swing Range & Tail',
                        hovertemplate=f'Factor 4 - Swing Range & Tail<br>Time: %{{x}}<br>High: ${qual_high:.2f}<br>Factor 4 Score: {range_score}/10<br>Range Ratio: {qual_candle["range_ratio"]:.2f}x<br>Tail: {qual_candle["tail_percentage"]:.1%}<extra></extra>',
                        showlegend=False
                    ), row=1, col=1)

            # Factor 5: Add marker on crossover candle if it meets criteria (score > 5)
            if crossover_range_score > 5:
                crossover_time = pd.to_datetime(candle.get('crossover_time'))
                crossover_high = candle.get('crossover_high', candle.get('crossover_close', 0))

                # Position marker above crossover candle high
                cross_y_position = crossover_high + 1.0  # Slightly higher than Factor 4 markers

                # Add yellow square marker (Factor 5) at crossover candle high
                fig.add_trace(go.Scatter(
                    x=[crossover_time],
                    y=[cross_y_position],
                    mode='markers',
                    marker=dict(
                        color='yellow',
                        size=12,  # Same size as stars and Factor 4
                        symbol='square'
                    ),
                    name='Factor 5 - Crossover Candle',
                    hovertemplate=f'Factor 5 - Crossover Candle<br>Time: %{{x}}<br>High: ${crossover_high:.2f}<br>Factor 5 Score: {crossover_range_score}/10<br>Range Ratio: {candle.get("crossover_range_ratio", 0):.2f}x<br>Tail: {candle.get("crossover_tail_percentage", 0):.1%}<extra></extra>',
                    showlegend=False
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

    # Plot master candle stars (gold stars below qualifying candles)
    if master_candles is not None and len(master_candles) > 0:
        print(f"Adding {len(master_candles)} master candle stars to chart")

        for candle in master_candles:
            # Convert timestamp to datetime if it's a string
            if isinstance(candle.get('crossover_time'), str):
                candle_time = pd.to_datetime(candle['crossover_time'])
            else:
                candle_time = candle.get('crossover_time')

            # Get the low price for this candle to position star below it
            candle_low = candle.get('crossover_low', candle.get('crossover_close', 0))

            # Only show Factor 1 gold star if volume_score > 5
            volume_score = candle.get('volume_score', 0)
            if volume_score > 5:
                # Position star very close to the low price (same as yellow stars)
                star_y_position = candle_low - 0.5  # 0.5 points below the low

                # Add gold star marker
                fig.add_trace(go.Scatter(
                x=[candle_time],
                y=[star_y_position],
                mode='markers',
                marker=dict(
                    color='gold',
                    size=12,
                    symbol='star',
                    line=dict(width=1, color='orange')
                ),
                name='Master Candle',
                    hovertemplate=f'Master Candle<br>Time: %{{x}}<br>Score: {candle.get("total_score", "N/A")}<br>Volume Ratio: {candle.get("volume_ratio", "N/A"):.2f}x<extra></extra>',
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
    TARGET_DATE = '2023-03-02'  # Cambiar esta fecha según necesidad
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
        plot_minute_data(symbol, timeframe, df_filtered, fractal_method='zigzag', change_pct=0.1, master_candles=None)

    else:
        print(f"No se encontraron datos para la fecha {TARGET_DATE}")
        print("Verifique que la fecha esté disponible en el archivo de datos.")