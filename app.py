import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta


# Configiracion de la pagina
st.set_page_config(page_title="Analizador de Portafolios", page_icon="📈", layout="wide")
st.sidebar.title("Analizador de Portafolios de Inversion")

# Creamos pestañas para la aplicacion
tab1, tab2 = st.tabs(["Analisis individual del Activo", "Analisis de Portafolio"])

# Entrada de simbolos y pesos 
simbolos = st.sidebar.text_input("Ingrese los simbolos de las acciones (separados por comas)", "AAPL, MSFT, GOOG, AMZN, NVDA")
pesos = st.sidebar.text_input("Ingrese los pesos de las acciones (separados por comas)", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip().upper() for s in simbolos.split(",")]
pesos = [float(p) for p in pesos.split(",")]    

# Seleccion de benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}

selected_benchmark = st.sidebar.selectbox("Seleccione un benchmark", list(benchmark_options.keys()))

#Periodo de tiempo
end_date = datetime.today().date()
start_date_options = { 

    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
    "2 años": end_date - timedelta(days=365*2),
    "5 años": end_date - timedelta(days=365*5),
    "10 años": end_date - timedelta(days=365*10) }

selected_timeframe = st.sidebar.selectbox("Seleccione el periodo de tiempo", list(start_date_options.keys()))
start_date = start_date_options[selected_timeframe]

# FUNCIONES AUXILIARES

def obtener_datos(simbolos, start_date, end_date):
    """Obtiene los datos de precios ajustados de los simbolos especificados entre las fechas dadas."""
    data = yf.download(simbolos, start=start_date, end=end_date)["Close"]
    return data.ffill().dropna()

def calcular_metricas(data):
    """Calcula los rendimientos diarios y acumulados de los precios ajustados."""
    returns = data.pct_change().dropna()
    returns_acumulados = (1 + returns).cumprod() - 1
    normalized_prices = data / data.iloc[0] * 100
    return returns, returns_acumulados, normalized_prices

def calcular_rendimiento_portafolio(returns, pesos):
    
    portafolio_returns = (returns * pesos).sum(axis=1)
    return portafolio_returns

def Calcular_Var(returns, confidence_level=0.95):
    """Calcula el VaR del portafolio."""
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var
def Calcular_CVaR(returns, var):
    """Calcula el CVaR del portafolio."""
    cvar = returns[returns <= var].mean()
    return cvar

def calcular_sharpe_dinamico(rendimientos, selected_timeframe, rf_anual=0.0449):
    """
    Calcula el Sharpe Ratio ajustado al horizonte temporal seleccionado.
    
    Parámetros:
    - rendimientos: pd.Series con retornos diarios del activo.
    - selected_timeframe: str, clave del periodo seleccionado por el usuario (ej. '3 meses').
    - rf_anual: float, tasa libre de riesgo anualizada (por defecto 4.49%).

    Retorna:
    - sharpe_ratio ajustado al periodo seleccionado.
    """

    # Diccionario de días hábiles estimados por periodo
    period_days = {
        "1 mes": 21,
        "3 meses": 63,
        "6 meses": 126,
        "1 año": 252,
        "2 años": 504,
        "5 años": 1260,
        "10 años": 2520
    }

    dias_periodo = period_days.get(selected_timeframe, 252)  # por defecto 1 año

    # Tasa libre de riesgo ajustada al periodo (compuesta)
    rf_periodo = (1 + rf_anual) ** (dias_periodo / 252) - 1

    # Retorno esperado y volatilidad ajustados al periodo
    retorno_esperado = rendimientos.mean() * dias_periodo
    volatilidad_ajustada = rendimientos.std() * np.sqrt(dias_periodo)

    sharpe_ratio = (retorno_esperado - rf_periodo) / volatilidad_ajustada
    return sharpe_ratio



if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    # Mensaje de error si los simbolos y pesos no coinciden
    st.sidebar.error("El número de símbolos y pesos no coincide. Por favor, verifique los datos ingresados.")
else:

    # Descarga de datos

    all_symbols = simbolos + [benchmark_options[selected_benchmark]]
    data_stocks = obtener_datos(all_symbols, start_date, end_date)
    returns, returns_acumulados, precios_norm = calcular_metricas(data_stocks)


    # TAB 1: ANALISIS INDIVIDUAL DEL ACTIVO 
    with tab1:
        st.header("📊 Análisis Individual del Activo")

        selected_asset = st.selectbox("Seleccione un activo para analizar", simbolos)

        # Extraemos series de tiempo específicas del activo
        precios = data_stocks[selected_asset]
        rendimientos = returns[selected_asset]
        rend_acumulado = returns_acumulados[selected_asset]

        # ================================
        # 1️⃣ RESUMEN GENERAL DE RENDIMIENTO
        # ================================
        st.subheader("🔹 Resumen de Rendimiento")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendimiento Acumulado (%)", f"{rend_acumulado.iloc[-1] * 100:.2f}%")
        col2.metric("Media de Retornos Diarios (%)", f"{rendimientos.mean() * 100:.4f}%")
        col3.metric("Volatilidad Anualizada (%)", f"{rendimientos.std() * np.sqrt(252) * 100:.2f}%")

        # ================================
        # 2️⃣ INDICADORES DE RIESGO
        # ================================
        st.subheader("🔸 Indicadores de Riesgo")
        sharpe = calcular_sharpe_dinamico(rendimientos, selected_timeframe)
        sortino = rendimientos.mean() / rendimientos[rendimientos < 0].std()
        var_95 = Calcular_Var(rendimientos)
        cvar_95 = Calcular_CVaR(rendimientos, var_95)
        max_drawdown = (rend_acumulado.cummax() - rend_acumulado).max()

        col4, col5, col6 = st.columns(3)
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col5.metric("Sortino Ratio", f"{sortino:.2f}")
        col6.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}%")

        col7, col8 = st.columns(2)
        col7.metric("VaR 95% (%)", f"{var_95 * 100:.2f}%")
        col8.metric("CVaR 95% (%)", f"{cvar_95 * 100:.2f}%")

        # ================================
        # 3️⃣ ESTADÍSTICAS AVANZADAS
        # ================================
        st.subheader("📐 Estadísticas de Retornos")
        skewness = rendimientos.skew()
        kurtosis = rendimientos.kurtosis()

        col9, col10 = st.columns(2)
        col9.metric("Skewness", f"{skewness:.3f}")
        col10.metric("Curtosis", f"{kurtosis:.3f}")

        # ================================
        # 4️⃣ GRÁFICOS INTERACTIVOS
        # ================================

        st.subheader("📊 Comparaciones Visuales: Activo vs Benchmark")
        benchmark_symbol = benchmark_options[selected_benchmark]
        benchmark_norm = precios_norm[benchmark_symbol]
        benchmark_returns = returns[benchmark_symbol]

        col_fig1, col_fig2 = st.columns(2)

        # === Gráfico 1: Evolución de precios normalizados
        with col_fig1:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=precios_norm.index, y=precios_norm[selected_asset], 
                                        name=selected_asset, line=dict(color='royalblue')))
            fig_price.add_trace(go.Scatter(x=benchmark_norm.index, y=benchmark_norm, 
                                        name=selected_benchmark, line=dict(color='firebrick')))
            fig_price.update_layout(title=f"{selected_asset} vs {selected_benchmark} - Precio Normalizado",
                                    xaxis_title="Fecha", yaxis_title="Precio Indexado (100)",
                                    height=400)
            st.plotly_chart(fig_price, use_container_width=True)

        # === Gráfico 2: Histograma de retornos
        with col_fig2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=rendimientos, name=selected_asset, opacity=0.6, color='royalblue'))
            fig_hist.add_trace(go.Histogram(x=benchmark_returns, name=selected_benchmark, opacity=0.6, color='firebrick'))
            fig_hist.update_layout(barmode='overlay',
                                title=f"Distribución de Retornos Diarios: {selected_asset} vs {selected_benchmark}",
                                xaxis_title="Retorno Diario", yaxis_title="Frecuencia",
                                height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
