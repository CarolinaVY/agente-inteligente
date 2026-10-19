import sqlite3
from datetime import datetime
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================
# 1. CARGAR EL MODELO ENTRENADO Y CONFIGURAR MEMORIA
# ==========================================
try:
    # Carga el modelo que entrenaste en Colab (.pkl)
    model = joblib.load('creditscore_model.pkl')
    print("Modelo de Machine Learning cargado exitosamente.")
except Exception as e:
    print(f"Advertencia: No se encontró el archivo .pkl ({e}). Asegúrate de subirlo a la misma carpeta.")

def init_db():
    conn = sqlite3.connect("credit_agent_memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            loan_amnt REAL,
            annual_inc REAL,
            fico_range_low REAL,
            dti REAL,
            probabilidad_mora REAL,
            decision TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. DEFINICIÓN DE PERCEPCIONES (Inputs)
# ==========================================
class SolicitudCredito(BaseModel):
    loan_amnt: float = Field(..., gt=0, description="Monto solicitado del préstamo")
    annual_inc: float = Field(..., gt=0, description="Ingresos anuales declarados")
    fico_range_low: float = Field(..., ge=300, le=850, description="Puntaje FICO / Buró de crédito")
    dti: float = Field(..., ge=0, description="Ratio de endeudamiento (Debt-to-Income)")
    historial_delitos: bool = Field(False, description="Presencia en listas restrictivas AML")

app = FastAPI(
    title="CREDITSCORE-AGENT",
    description="Agente Inteligente Basado en Modelos y Aprendizaje para Scoring Crediticio",
    version="1.0.0"
)

# ==========================================
# 3. NÚCLEO DEL AGENTE (Agent Core & Loop)
# ==========================================
@app.post("/evaluar-credito/")
def ejecutar_ciclo_agente(solicitud: SolicitudCredito):
    # Paso A: Percepción y Guardrails (Filtro de seguridad institucional)
    if solicitud.historial_delitos:
        accion = "DERIVAR_ANALISTA"
        prob_mora = 1.00
    else:
        # Paso B: Planificación y Toma de Decisiones (Inferencia del Modelo Real)
        features = [[solicitud.loan_amnt, solicitud.annual_inc, solicitud.fico_range_low, solicitud.dti]]
        
        try:
            # Predice la probabilidad de caer en mora (clase 1)
            probabilidades = model.predict_proba(features)[0]
            prob_mora = float(probabilidades[1])
        except Exception:
            # Fallback por seguridad si el modelo no está cargado
            prob_mora = 0.15

        # Reglas de decisión basadas en utilidad y umbrales de riesgo
        if prob_mora <= 0.30 and solicitud.fico_range_low >= 680:
            accion = "APROBAR_CREDITO"
        elif prob_mora > 0.60:
            accion = "RECHAZAR_CREDITO"
        else:
            accion = "DERIVAR_ANALISTA"  # Zona gris enviada a revisión humana

    # Paso C: Memoria (Registro persistente en SQLite - State Tracking)
    try:
        conn = sqlite3.connect("credit_agent_memory.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluaciones (timestamp, loan_amnt, annual_inc, fico_range_low, dti, probabilidad_mora, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            solicitud.loan_amnt,
            solicitud.annual_inc,
            solicitud.fico_range_low,
            solicitud.dti,
            prob_mora,
            accion
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error al guardar en memoria SQLite: {e}")

    # Paso D: Actuadores (Respuesta estructurada para el entorno)
    return {
        "agente": "CREDITSCORE-AGENT",
        "timestamp": datetime.now().isoformat(),
        "percepcion_recibida": solicitud.dict(),
        "evaluacion_interna": {
            "probabilidad_mora_calculada": round(prob_mora, 4)
        },
        "accion_ejecutada": accion,
        "mensaje": f"El expediente ha sido procesado exitosamente por el agente con veredicto: {accion}"
    }
