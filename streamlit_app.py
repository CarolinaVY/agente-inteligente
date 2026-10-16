import streamlit as st
import joblib
import pandas as pd
import sklearn

st.title("CREDITSCORE-AGENT 💳")
st.write("Agente Inteligente con DTI y Puntaje FICO Automatizados")

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
    
    # Entradas principales
    loan_amnt = st.number_input("Monto del Préstamo Solicitado ($)", value=10000.0, min_value=0.0)
    annual_inc = st.number_input("Ingresos Anuales ($)", value=60000.0, min_value=0.0)
    
    st.markdown("---")
    st.write("###  DTI y FICO")
    
    pagos_mensuales_deudas = st.number_input("Pagos mensuales actuales de otras deudas ($)", value=500.0, min_value=0.0)
    retrasos_previos = st.selectbox("¿Tiene historial de pagos atrasados?", ["No", "Sí, ocasionales", "Sí, graves/recurrentes"])
    
    # 1. Cálculo automático del DTI
    ingreso_mensual = annual_inc / 12.0 if annual_inc > 0 else 1.0
    dti = (pagos_mensuales_deudas / ingreso_mensual) * 100.0
    
    # 2. Cálculo automático del Puntaje FICO estimado (Rango: 300 - 850)
    # Base inicial ideal: 750 puntos
    fico_base = 750
    
    # Penalización por DTI alto
    if dti > 40:
        fico_base -= 150
    elif dti > 25:
        fico_base -= 80
    elif dti > 15:
        fico_base -= 30
        
    # Penalización por historial de retrasos
    if retrasos_previos == "Sí, ocasionales":
        fico_base -= 70
    elif retrasos_previos == "Sí, graves/recurrentes":
        fico_base -= 180
        
    # Asegurar que esté dentro de los límites reales de FICO (300 a 850)
    fico_range_low = float(max(300, min(850, fico_base)))
    
    # Mostrar los resultados calculados en tiempo real
    st.info(f"💡 **DTI Calculado:** {dti:.2f}%  |  💡 **Puntaje FICO Estimado:** {int(fico_range_low)}")

    historial_delitos = st.checkbox("¿Tiene antecedentes / historial delictivo (AML)?")

    if st.button("Evaluar Crédito"):
        if historial_delitos:
            accion = "DERIVAR_ANALISTA"
            prob_mora = 1.00
        else:
            # El modelo recibe las 4 variables oficiales
            features = [[loan_amnt, annual_inc, fico_range_low, dti]]
            prob_mora = float(model.predict_proba(features)[0][1])
            
            if prob_mora <= 0.30 and fico_range_low >= 680 and dti <= 35.0:
                accion = "APROBAR_CREDITO"
            elif prob_mora > 0.60 or dti > 50.0 or fico_range_low < 600:
                accion = "RECHAZAR_CREDITO"
            else:
                accion = "DERIVAR_ANALISTA"
                
        st.subheader(f"Resultado: {accion}")
        st.write(f"**Probabilidad de Mora Calculada:** {prob_mora:.4f}")
        st.write(f"**Métricas Automáticas Utilizadas:** DTI: {dti:.2f}% | FICO: {int(fico_range_low)}")
