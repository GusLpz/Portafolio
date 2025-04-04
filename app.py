import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Agregamos CSS personalizado para mejorar el diseño
st.markdown("""
    <style>
    /* Fondo y tipografía general */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }
    /* Fondo de la barra lateral */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    /* Títulos y subtítulos */
    h1, h2, h3, h4, h5, h6 {
        color: #1F3A93;
    }
    /* Estilos para botones de descarga */
    .stButton>button {
        background-color: #1F3A93;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    /* Estilos para los checkbox */
    .stCheckbox>div {
        background-color: #e8ebf0;
        padding: 8px;
        border-radius: 5px;
    }
    /* Líneas divisorias */
    hr {
        border: 0;
        height: 1px;
        background: #cccccc;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Configuración inicial de la página
st.set_page_config(
    page_title="Portafolio de Finanzas Cuantitativas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función principal para la navegación
def main():
    st.sidebar.title("Navegación")
    menu = st.sidebar.radio("Ir a", ["Inicio", "Proyectos", "Visualizaciones", "CV y Contacto"])
    
    if menu == "Inicio":
        mostrar_inicio()
    elif menu == "Proyectos":
        mostrar_proyectos()
    elif menu == "Visualizaciones":
        mostrar_visualizaciones()
    elif menu == "CV y Contacto":
        mostrar_cv_contacto()

# Sección de Introducción y Perfil Profesional
def mostrar_inicio():
    st.title("Bienvenido a mi Portafolio en Finanzas Cuantitativas")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("""
    **Resumen Personal:**  
    Soy un profesional en finanzas cuantitativas con experiencia en análisis de datos, optimización de portafolios, 
    backtesting de estrategias y modelado predictivo. Mi objetivo es aplicar métodos cuantitativos para resolver 
    problemas financieros complejos y generar valor en entornos de alta competitividad.
    """)
    st.subheader("Habilidades y Tecnologías")
    st.write("""
    - Python (Pandas, NumPy, Scikit-learn)
    - Análisis de Series Temporales
    - Machine Learning y Deep Learning
    - Optimización de Cartera
    - Simulaciones Monte Carlo
    - Visualización interactiva con Plotly y Matplotlib
    """)

# Sección de Proyectos Destacados
def mostrar_proyectos():
    st.title("Proyectos Destacados")
    
    # Proyecto 1: Análisis de Datos Financieros
    st.subheader("Análisis de Datos Financieros")
    st.write("""
    En este proyecto se analizan series temporales y se implementan modelos predictivos para estimar precios de activos.
    """)
    if st.checkbox("Ver ejemplo de análisis"):
        df = pd.DataFrame({
            'Fecha': pd.date_range(start="2022-01-01", periods=100, freq='D'),
            'Precio': np.random.randn(100).cumsum() + 100
        })
        st.line_chart(df.set_index('Fecha'))
    
    # Proyecto 2: Optimización de Portafolios
    st.subheader("Optimización de Portafolios")
    st.write("""
    Proyecto donde se aplican técnicas de optimización para la asignación de activos utilizando modelos matemáticos.
    """)
    if st.checkbox("Ver ejemplo de optimización"):
        st.write("Aquí se mostrarían los resultados y visualizaciones del proceso de optimización.")
    
    # Proyecto 3: Backtesting de Estrategias de Trading
    st.subheader("Backtesting de Estrategias")
    st.write("""
    Implementación y evaluación de estrategias de trading cuantitativo mediante backtesting histórico.
    """)
    if st.checkbox("Ver ejemplo de backtesting"):
        st.write("Código y visualizaciones del proceso de backtesting.")
    
    # Proyecto 4: Integración con APIs
    st.subheader("Integración con APIs")
    st.write("""
    Ejemplos de cómo se pueden integrar datos en tiempo real utilizando APIs como Alpha Vantage o Yahoo Finance.
    """)
    st.write("Puedes incluir enlaces a repositorios o demostraciones en vivo.")

# Sección de Visualizaciones Interactivas
def mostrar_visualizaciones():
    st.title("Visualizaciones Interactivas")
    
    st.write("Ejemplo de visualización interactiva con Plotly:")
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    fig = px.scatter(df, x='x', y='y', title="Visualización con Plotly")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Ejemplo de gráfico con Matplotlib:")
    fig2, ax = plt.subplots()
    ax.plot(np.random.randn(100))
    ax.set_title("Gráfico con Matplotlib")
    st.pyplot(fig2)

# Sección de CV y Datos de Contacto
def mostrar_cv_contacto():
    st.title("CV y Contacto")
    
    st.subheader("Contacto")
    st.write("""
    Puedes contactarme a través de:
    - **Correo:** tuemail@dominio.com
    - **LinkedIn:** [Perfil LinkedIn](https://www.linkedin.com/)
    - **GitHub:** [Repositorio GitHub](https://github.com/)
    """)
    
    st.subheader("Descargar CV")
    try:
        with open("CV.pdf", "rb") as cv_file:
            st.download_button(
                label="Descargar CV",
                data=cv_file,
                file_name="CV.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.error("CV.pdf no encontrado. Asegúrate de tener el archivo en el directorio del proyecto.")

if __name__ == "__main__":
    main()
