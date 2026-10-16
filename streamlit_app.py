import streamlit as st
import joblib
import pandas as pd

st.title("CREDITSCORE-AGENT 💳")
st.write("Agente Inteligente de Scoring Crediticio")

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load('creditscore_model.pkl')

model = cargar_modelo()

# Formulario de entrada de datos en la interfaz
loan_amnt = st.number_input("Monto del Préstamo ($)", value=10000.0)
annual_inc = st.number_input("Ingresos Anuales ($)", value=50000.0)
fico_range_low = st.number_input("Puntaje FICO", value=700.0)
dti = st.number_input("Ratio de Endeudamiento (DTI)", value=15.0)
historial_delitos = st.checkbox("¿Tiene antecedentes/historial delictivo (AML)?")

if st.button("Evaluar Crédito"):
    if historial_delitos:
        accion = "DERIVAR_ANALISTA"
        prob_mora = 1.00
    else:
        features = [[loan_amnt, annual_inc, fico_range_low, dti]]
        prob_mora = float(model.predict_proba(features)[0][1])
        
        if prob_mora <= 0.30 and fico_range_low >= 680:
            accion = "APROBAR_CREDITO"
        elif prob_mora > 0.60:
            accion = "RECHAZAR_CREDITO"
        else:
            accion = "DERIVAR_ANALISTA"
            
    st.subheader(f"Resultado: {accion}")
    st.write(f"**Probabilidad de Mora Calculada:** {prob_mora:.4f}")
