import os
import webbrowser
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from config import CHART_WIDTH, CHART_HEIGHT, CHART_TEMPLATE, get_chart_path

def plot_minute_data(symbol, timeframe, df):
    """
    Función especializada para graficar datos de minutos con etiquetas de hora en el eje X
    """
    html_path = get_chart_path(symbol, timeframe)

    df = df.rename(columns=str.lower)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.03,
    )

    # Gráfico de velas (candlestick) con perfil dark grey
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',  # Velas verdes (alcistas)
        decreasing_line_color='red',    # Velas rojas (bajistas)
        increasing_fillcolor='darkgreen',
        decreasing_fillcolor='darkred',
        line=dict(width=1),
        name='OHLC'
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
        title=f'{symbol}_{timeframe} - Candlestick por Minuto',
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12, color="black"),
        plot_bgcolor='white',
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
            gridcolor='rgba(128,128,128,0.1)',  # Gris muy pálido, casi transparente
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
        "displayModeBar": False,  # Eliminar barra de navegación
        "staticPlot": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "chart",
            "height": 500,
            "width": 700,
            "scale": 1
        }
    })
    print(f"✅ Gráfico de minutos guardado como HTML: '{html_path}'")

    webbrowser.open('file://' + os.path.realpath(html_path))