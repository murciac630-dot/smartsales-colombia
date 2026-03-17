import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="SmartSales Colombia", page_icon="🚀")

st.title("🚀 SmartSales: IA para el Agro")
st.write("Predicción de precios mayoristas en el Valle del Cauca.")

# --- Carga de Datos ---
@st.cache_data
def load_data():
    # Cargamos el archivo que subiste a GitHub
    df = pd.read_csv('ventas_valle_sipsa.csv')
    return df

try:
    df_smart = load_data()

    # --- Barra Lateral ---
    st.sidebar.header("Opciones de Predicción")
    producto = st.sidebar.selectbox("Selecciona un producto:", df_smart['Producto'].unique())

    # --- Procesamiento de IA ---
    df_p = df_smart[df_smart['Producto'] == producto].copy()
    df_p['Dia_Num'] = np.arange(len(df_p))

    X = df_p[['Dia_Num']]
    y = df_p['Precio_Promedio']

    modelo = LinearRegression().fit(X, y)
    
    # Predicción para el día siguiente
    proximo_dia = len(df_p)
    prediccion = modelo.predict([[proximo_dia]])

    # --- Resultados en Pantalla ---
    st.metric(label=f"Precio estimado para mañana ({producto})", value=f"${prediccion[0]:,.2f}")

    # Gráfica de Tendencia
    st.subheader("Gráfica de Tendencia Predictiva")
    fig, ax = plt.subplots()
    ax.scatter(df_p['Dia_Num'], y, color='skyblue', label='Historial', alpha=0.6)
    ax.plot(df_p['Dia_Num'], modelo.predict(X), color='red', label='IA SmartSales')
    ax.set_xlabel("Días del Año")
    ax.set_ylabel("Precio por Kilo ($)")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error("⚠️ Error: No se encontró el archivo de datos o el formato es incorrecto.")
    st.write("Asegúrate de haber subido 'ventas_valle_sipsa.csv' a tu repositorio.")
