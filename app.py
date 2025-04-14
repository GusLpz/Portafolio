import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt


# Configiracion de la pagina
st.set_page_config(page_title="holis holis", page_icon="📈", layout="wide")
st.sidebar.title("Analizador de Portafolios de Inversion")

# Creamos pestañas para la aplicacion
tab1, tab2, tab3 = st.tabs(["Analisis individual del Activo", "Analisis de Portafolio", "Simulación Monte Carlo"])

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
        beta = np.cov(rendimientos, returns[benchmark_options[selected_benchmark]])[0][1] / np.var(returns[benchmark_options[selected_benchmark]])

        max_drawdown = (rend_acumulado.cummax() - rend_acumulado).max()

        col4, col5, col6 = st.columns(3)
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col5.metric("Sortino Ratio", f"{sortino:.2f}")
        col6.metric("Max Drawdown (%)", f"{max_drawdown * 100:.2f}%")
        



        col7, col8, colbeta = st.columns(3)
        colbeta.metric("Beta", f"{beta:.2f}")
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

        # === Gráfico principal de precios normalizados (100 base)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=precios_norm.index, y=precios_norm[selected_asset], 
                                    name=selected_asset, line=dict(color='royalblue')))
        fig_price.add_trace(go.Scatter(x=benchmark_norm.index, y=benchmark_norm, 
                                    name=selected_benchmark, line=dict(color='firebrick')))
        fig_price.update_layout(title=f"Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)",
                                xaxis_title="Fecha", yaxis_title="Precio Normalizado")
        st.plotly_chart(fig_price, use_container_width=True)



        # === Histogramas por separado
        st.subheader(f"Distribución de Retornos: {selected_asset} vs {selected_benchmark}")
        col_hist1, col_hist2 = st.columns(2)

        # Histograma del activo seleccionado
        with col_hist1:
            fig_hist_asset = px.histogram(rendimientos, nbins=50, title=f"Distribución de Retornos - {selected_asset}",
                                        labels={"value": "Retornos"}, color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig_hist_asset, use_container_width=True)

        # Histograma del benchmark
        with col_hist2:
            fig_hist_benchmark = px.histogram(benchmark_returns, nbins=50, title=f"Distribución de Retornos - {selected_benchmark}",
                                            labels={"value": "Retornos"}, color_discrete_sequence=["#ff7f0e"])
            st.plotly_chart(fig_hist_benchmark, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: ANALISIS DEL PORTAFOLIO
    # ---------------------------------------------------------
    with tab2:
        st.header("📈 Análisis del Portafolio")

        # Calculamos los retornos del benchmark y del portafolio
        benchmark_symbol = benchmark_options[selected_benchmark]
        benchmark_returns = returns[benchmark_symbol]
        portfolio_returns = calcular_rendimiento_portafolio(returns[simbolos], pesos)

        # Rendimientos acumulados para portafolio y benchmark
        portfolio_cumreturns = (1 + portfolio_returns).cumprod() - 1
        benchmark_cumreturns = (1 + benchmark_returns).cumprod() - 1

        # Calculamos las principales métricas del portafolio
        total_return_portfolio = portfolio_cumreturns.iloc[-1] * 100  # en porcentaje
        sharpe_portfolio = calcular_sharpe_dinamico(portfolio_returns, selected_timeframe)
        sortino_portfolio = portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std()
        var_95_portfolio = Calcular_Var(portfolio_returns)
        cvar_95_portfolio = Calcular_CVaR(portfolio_returns, var_95_portfolio)
        max_dd_portfolio = (portfolio_cumreturns.cummax() - portfolio_cumreturns).max() * 100

        # NUEVO: Cálculo del beta del portafolio
        beta_portfolio = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)

        # Mostramos las métricas utilizando columnas (se agregan 6 columnas para incluir el beta)
        colp1, colp2, colp3 = st.columns(3)
        colp1.metric("Rendimiento Total", f"{total_return_portfolio:.2f}%")
        colp2.metric("Sharpe Ratio", f"{sharpe_portfolio:.2f}")
        colp3.metric("Sortino Ratio", f"{sortino_portfolio:.2f}")

        colp4, colp5, colp6 = st.columns(3)
        colp4.metric("VaR 95%", f"{var_95_portfolio * 100:.2f}%")
        colp5.metric("CVaR 95%", f"{cvar_95_portfolio * 100:.2f}%")
        colp6.metric("Beta del Portafolio", f"{beta_portfolio:.2f}")

        # Gráfico comparativo: Rendimientos Acumulados del Portafolio vs Benchmark
        st.subheader(f"Rendimientos Acumulados: Portafolio vs {selected_benchmark}")
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=portfolio_cumreturns.index,
            y=portfolio_cumreturns,
            name='Portafolio',
            line=dict(color='blue')
        ))
        fig_port.add_trace(go.Scatter(
            x=benchmark_cumreturns.index,
            y=benchmark_cumreturns,
            name=selected_benchmark,
            line=dict(color='orange')
        ))
        fig_port.update_layout(
            title=f"Rendimientos Acumulados: Portafolio vs {selected_benchmark}",
            xaxis_title="Fecha",
            yaxis_title="Rendimiento Acumulado"
        )
        st.plotly_chart(fig_port, use_container_width=True)



        
#Contenido Tab Analis de Portafolio 

    









        
with tab3: 
        st.header("Parámetros de la Simulación")

        # Entrada de parámetros
        S0 = st.number_input("Precio actual del activo (S0)", value=100.0, min_value=0.0, step=1.0)
        K = st.number_input("Precio de ejercicio (K)", value=105.0, min_value=0.0, step=1.0)
        T = st.number_input("Tiempo hasta vencimiento (T, años)", value=1.0, min_value=0.1, step=0.1)
        r = st.number_input("Tasa libre de riesgo (r)", value=0.05, step=0.01)
        sigma = st.number_input("Volatilidad (σ)", value=0.2, step=0.01)
        N = st.number_input("Número de simulaciones (N)", value=100000, step=1000)

        # Opción para mostrar trayectorias de precios
        mostrar_paths = st.checkbox("Mostrar sample paths simulados")
        if mostrar_paths:
            n_paths = st.number_input("Número de sample paths", value=10, step=1)
            n_steps = st.number_input("Número de pasos en el tiempo", value=100, step=10)

        if st.button("Ejecutar Simulación"):
            # Semilla para reproducibilidad
            np.random.seed(42)
            
            ## SIMULACIÓN DE VALOR FINAL DEL ACTIVO (ST)
            # Generación de variables aleatorias para la simulación (valor final)
            Z = np.random.standard_normal(int(N))
            ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
            
            # Cálculo del payoff para cada simulación
            payoffs = np.maximum(ST - K, 0)
            call_price_mc = np.exp(-r * T) * np.mean(payoffs)
            
            # Cálculo del precio según la fórmula de Black-Scholes
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call_price_bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            
            # Mostrar resultados
            st.write(f"**Precio de la opción (Monte Carlo):** {call_price_mc:.2f}")
            st.write(f"**Precio de la opción (Black-Scholes):** {call_price_bs:.2f}")
            
            ## GRÁFICO 1: Histograma de ST con línea en K
            fig1, ax1 = plt.subplots()
            ax1.hist(ST, bins=50, density=True, alpha=0.7)
            ax1.axvline(K, color='red', linestyle='dashed', linewidth=2, label=f'Precio de ejercicio (K={K})')
            ax1.set_title("Distribución de precios al vencimiento (ST)")
            ax1.set_xlabel("Precio del activo")
            ax1.set_ylabel("Densidad")
            ax1.legend()
            st.pyplot(fig1)
            
            ## GRÁFICO 2: Convergencia de la estimación de la opción
            # Se calcula la media acumulada de los payoffs descontados
            running_avg = np.cumsum(payoffs) / np.arange(1, int(N)+1)
            running_price = np.exp(-r * T) * running_avg

            fig2, ax2 = plt.subplots()
            ax2.plot(running_price, lw=1)
            ax2.axhline(call_price_mc, color='red', linestyle='dashed', linewidth=2, label=f'Valor final (MC = {call_price_mc:.2f})')
            ax2.set_title("Convergencia del precio de la opción (Monte Carlo)")
            ax2.set_xlabel("Número de simulaciones")
            ax2.set_ylabel("Precio estimado")
            ax2.legend()
            st.pyplot(fig2)
            
            ## GRÁFICO 3: Simulación de sample paths (si se selecciona)
            if mostrar_paths:
                dt = T / n_steps
                time_grid = np.linspace(0, T, n_steps+1)
                paths = np.zeros((n_paths, n_steps+1))
                paths[:, 0] = S0
                
                for i in range(n_paths):
                    # Simulación de una trayectoria
                    z = np.random.standard_normal(n_steps)
                    for j in range(1, n_steps+1):
                        paths[i, j] = paths[i, j-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z[j-1])
                
                fig3, ax3 = plt.subplots()
                for i in range(n_paths):
                    ax3.plot(time_grid, paths[i, :], lw=1, label=f'Trayectoria {i+1}' if n_paths<=10 else None)
                ax3.set_title("Sample paths simulados")
                ax3.set_xlabel("Tiempo")
                ax3.set_ylabel("Precio del activo")
                if n_paths <= 10:
                    ax3.legend()
                st.pyplot(fig3)
