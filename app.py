import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="SmartSales Valle", page_icon="🚀")
st.title("🚀 SmartSales: IA Predictiva")

# --- BUSCADOR DE ARCHIVOS (Para evitar errores) ---
archivos_en_github = os.listdir('.')
st.write(f"Archivos detectados: {archivos_en_github}") # Esto nos dirá si ve el CSV

if 'ventas_valle_sipsa.csv' in archivos_en_github:
    df = pd.read_csv('ventas_valle_sipsa.csv')
    
    # --- Tu lógica de IA ---
    producto = st.selectbox("Selecciona Producto", df['Producto'].unique())
    df_p = df[df['Producto'] == producto].copy()
    df_p['Dia_Num'] = np.arange(len(df_p))
    
    modelo = LinearRegression().fit(df_p[['Dia_Num']], df_p['Precio_Promedio'])
    pred = modelo.predict([[len(df_p)]])
    
    st.metric(f"Predicción {producto}", f"${pred[0]:,.2f}")
    
    fig, ax = plt.subplots()
    ax.plot(df_p['Dia_Num'], df_p['Precio_Promedio'], label="Real")
    ax.plot(df_p['Dia_Num'], modelo.predict(df_p[['Dia_Num']]), color="red", label="IA")
    st.pyplot(fig)
else:
    st.error("❌ ¡ALERTA! El archivo 'ventas_valle_sipsa.csv' no se encuentra en la carpeta principal de GitHub.")


