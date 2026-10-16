import streamlit as st
import joblib
import pandas as pd
import sklearn

st.title("CREDITSCORE-AGENT 💳")
st.write("Agente Inteligente de Scoring Crediticio (con DTI Automatizado)")

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load('creditscore_model.pkl')

try:
    model = cargar_modelo()
    modelo_cargado = True
except Exception as e:
    modelo_cargado = False
    st.error(f"No se pudo cargar el archivo del modelo (.pkl): {e}")

if modelo_cargado:
    st.subheader("📋 Datos Financieros del Solicitante")
    
    # Entradas del formulario
    loan_amnt = st.number_input("Monto del Préstamo Solicitado ($)", value=10000.0, min_value=0.0)
    annual_inc = st.number_input("Ingresos Anuales ($)", value=60000.0, min_value=0.0)
    fico_range_low = st.number_input("Puntaje FICO", value=700.0, min_value=300.0, max_value=850.0)
    
    # Nuevos campos para calcular el DTI automáticamente
    st.markdown("---")
    st.write("### Ratio de Endeudamiento (DTI)")
    pagos_mensuales_deudas = st.number_input("Pagos mensuales actuales de otras deudas ($)", value=500.0, min_value=0.0)
    
    # Cálculo automático del DTI (Debt-to-Income Ratio) mensual
    ingreso_mensual = annual_inc / 12.0 if annual_inc > 0 else 1.0
    dti = (pagos_mensuales_deudas / ingreso_mensual) * 100.0
    
    st.info(f"💡 **DTI Calculado Automáticamente:** {dti:.2f}% (Basado en tus deudas mensuales frente a tus ingresos mensuales)")

    historial_delitos = st.checkbox("¿Tiene antecedentes / historial delictivo (AML)?")

    if st.button("Evaluar Crédito"):
        if historial_delitos:
            accion = "DERIVAR_ANALISTA"
            prob_mora = 1.00
        else:
            # El modelo espera las características originales, incluyendo el DTI calculado
            features = [[loan_amnt, annual_inc, fico_range_low, dti]]
            prob_mora = float(model.predict_proba(features)[0][1])
            
            if prob_mora <= 0.30 and fico_range_low >= 680 and dti <= 35.0:
                accion = "APROBAR_CREDITO"
            elif prob_mora > 0.60 or dti > 50.0:
                accion = "RECHAZAR_CREDITO"
            else:
                accion = "DERIVAR_ANALISTA"
                
        st.subheader(f"Resultado: {accion}")
        st.write(f"**Probabilidad de Mora Calculada:** {prob_mora:.4f}")
        st.write(f"**DTI Utilizado para la Evaluación:** {dti:.2f}%")
