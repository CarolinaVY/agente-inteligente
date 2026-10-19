import streamlit as st
import joblib
import pandas as pd
import time
import os
from datetime import datetime

st.title("CREDITSCORE-AGENT 💳")
st.write("Agente Inteligente para Scoring Crediticio y Gestión de Riesgo (Fase 2 & 3)")

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
    st.subheader("📋 1. Percepción (Entradas / Sensores)")
    
    loan_amnt = st.number_input("Monto del Préstamo Solicitado ($)", value=10000.0, min_value=0.0)
    annual_inc = st.number_input("Ingresos Anuales ($)", value=60000.0, min_value=0.0)
    pagos_mensuales_deudas = st.number_input("Pagos mensuales de otras deudas ($)", value=500.0, min_value=0.0)
    retrasos_previos = st.selectbox("Historial de pagos atrasados", ["No", "Sí, ocasionales", "Sí, graves/recurrentes"])
    historial_delitos = st.checkbox("¿Tiene antecedentes / historial delictivo (AML)?")

    # CICLO DE PERCEPCIÓN AUTOMÁTICA (Cálculo DTI y FICO)
    ingreso_mensual = annual_inc / 12.0 if annual_inc > 0 else 1.0
    dti = (pagos_mensuales_deudas / ingreso_mensual) * 100.0
    
    fico_base = 750
    if dti > 40: fico_base -= 150
    elif dti > 25: fico_base -= 80
    elif dti > 15: fico_base -= 30
        
    if retrasos_previos == "Sí, ocasionales": fico_base -= 70
    elif retrasos_previos == "Sí, graves/recurrentes": fico_base -= 180
        
    fico_estimado = float(max(300, min(850, fico_base)))
    
    st.info(f"💡 **DTI Calculado:** {dti:.2f}%  |  💡 **Puntaje FICO Estimado:** {int(fico_estimado)}")

    if st.button("Evaluar Crédito (Ejecutar Ciclo)"):
        start_time = time.time()
        
        # 2. DECISIÓN (Razonamiento / Modelo + Reglas de Gobernanza)
        if historial_delitos:
            accion = "DERIVAR_ANALISTA"
            prob_mora = 1.00
            motivo = "Alerta de Cumplimiento Normativo (AML)"
        else:
            features = [[loan_amnt, annual_inc, fico_estimado, dti]]
            prob_mora = float(model.predict_proba(features)[0][1])
            
            if prob_mora <= 0.30 and fico_estimado >= 680 and dti <= 35.0:
                accion = "APROBAR_CREDITO"
                motivo = "Riesgo Bajo y Solvencia Adecuada"
            elif prob_mora > 0.60 or dti > 50.0 or fico_estimado < 600:
                accion = "RECHAZAR_CREDITO"
                motivo = "Riesgo Alto de Mora o Sobreendeudamiento"
            else:
                accion = "DERIVAR_ANALISTA"
                motivo = "Riesgo Intermedio (Zona Gris)"
                
        execution_time = round(time.time() - start_time, 4)
        
        # 3. ACCIÓN (Renderizado UI)
        st.subheader(f"3. Acción / Resultado: {accion}")
        st.write(f"**Probabilidad de Mora Calculada:** {prob_mora:.4f}")
        st.write(f"**Motivo de la Decisión:** {motivo}")
        st.write(f"**Tiempo de Inferencia:** {execution_time} segundos")

        # 4. REGISTRO (LOG DE EJECUCIÓN - OBLIGATORIO FASE 2)
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "dti": round(dti, 2),
            "fico": int(fico_estimado),
            "aml_alert": historial_delitos,
            "prob_mora": round(prob_mora, 4),
            "accion": accion,
            "tiempo_seg": execution_time
        }
        
        log_file = "log_ejecucion.csv"
        df_log = pd.DataFrame([log_data])
        
        if not os.path.exists(log_file):
            df_log.to_csv(log_file, index=False)
        else:
            df_log.to_csv(log_file, mode='a', header=False, index=False)
            
        st.success("✅ Ejecución registrada en el Log operativo (log_ejecucion.csv)")
        
    # Mostrar el log acumulado en pantalla
    if os.path.exists("log_ejecucion.csv"):
        st.markdown("---")
        st.subheader("📜 Bitácora de Registros (Log de Ejecución)")
        df_historico = pd.read_csv("log_ejecucion.csv")
        st.dataframe(df_historico.tail(10))
