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
simbolos = st.sidebar.text_input("Ingrese los simbolos de las acciones (separados por comas)", "AAPL, MSFT, GOOG, AMZN, NIKE")
pesos = st.sidebar.text_input("Ingrese los pesos de las acciones (separados por comas)", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip().upper() for s in simbolos.split(",")]
pesos = [float(p) for p in pesos.split(",")]    

# Seleccion de benchmark
benchmark_options = { "S&P 500": "GSPC", "NASDAQ": "IXIC", "Dow Jones": "DJI", "Russell 2K": "RUT", "FTSE 100": "FTSE", "DAX": "GDAXI", "Nikkei 225": "N225", "Hang Seng": "HSI"}

selected_benchmark = st.sidebar.selectbox("Seleccione un benchmark", list(benchmark_options.keys()))

#Periodo de tiempo
end_date = datetime.now()
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
    #normalized_prices = data / data.iloc[0] * 100
    return returns, returns_acumulados

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

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    # Mensaje de error si los simbolos y pesos no coinciden
    st.sidebar.error("El número de símbolos y pesos no coincide. Por favor, verifique los datos ingresados.")
else:

    # Descarga de datos

    all_symbols = simbolos + [benchmark_options[selected_benchmark]]
    data_stocks = obtener_datos(all_symbols, start_date, end_date)
    returns, returns_acumulados= calcular_metricas(data_stocks)


# TAB 1: ANALISIS INDIVIDUAL DEL ACTIVO 

with tab1:

    st.header("Analisis individual del Activo")
    selected_asset = st.selectbox("Seleccione un activo", simbolos)
    col1 = st.columns(1)
    
    col1.metric("Rendimiento Acumulado (%)", f"{returns_acumulados[selected_asset].iloc[-1] * 100:.2f}%")
