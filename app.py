import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuración básica
st.set_page_config(page_title="SmartSales Valle", page_icon="🚀")
st.title("🚀 SmartSales: IA Predictiva")

# --- FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data
def obtener_datos():
    nombre_archivo = 'ventas_valle_sipsa.csv'
    try:
        df = pd.read_csv(nombre_archivo)
        return df
    except:
        # Respaldo por si el CSV falla
        productos = ['Tomate Larga Vida', 'Cebolla Cabezona', 'Papa Capira']
        fechas = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(365)]
        data = []
        for producto in productos:
            precio_base = np.random.randint(2000, 4000)
            for i, fecha in enumerate(fechas):
                precio = precio_base + (i * 1.5) + np.random.normal(0, 100)
                data.append([fecha.strftime('%Y-%m-%d'), producto, round(precio, 2)])
        return pd.DataFrame(data, columns=['Fecha', 'Producto', 'Precio_Promedio'])

# --- EJECUCIÓN ---
df = obtener_datos()
producto = st.selectbox("Selecciona un producto:", df['Producto'].unique())

df_p = df[df['Producto'] == producto].copy()
df_p['Dia_Num'] = np.arange(len(df_p))

modelo = LinearRegression().fit(df_p[['Dia_Num']], df_p['Precio_Promedio'])
prediccion = modelo.predict([[len(df_p)]])

st.metric(label=f"Precio estimado mañana ({producto})", value=f"${prediccion[0]:,.2f}")

fig, ax = plt.subplots()
ax.scatter(df_p['Dia_Num'], df_p['Precio_Promedio'], color='skyblue', alpha=0.5, label="Historial")
ax.plot(df_p['Dia_Num'], modelo.predict(df_p[['Dia_Num']]), color='red', label="IA SmartSales")
ax.legend()
st.pyplot(fig)

st.info("MVP SmartSales - Valle del Cauca")
