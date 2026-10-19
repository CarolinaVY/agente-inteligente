# 🛡️ CREDITSCORE-AGENT: Hybrid Credit Scoring & AML Governance Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Cloud-red.svg" alt="Streamlit Cloud">
  <img src="https://img.shields.io/badge/Scikit--Learn-Model-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/AUC--ROC-0.645-success.svg" alt="AUC-ROC Score">
  <img src="https://img.shields.io/badge/Git-LFS-critical.svg" alt="Git LFS">
</p>

Sistema inteligente híbrido para la evaluación automática de riesgo crediticio y gobernanza regulatoria (Anti-Money Laundering - AML). Combina modelos predictivos de Machine Learning con reglas deductivas deterministas para la toma de decisiones financieras en tiempo real.

---

## 🚀 Enlaces de Acceso
* **Plataforma en Producción (Streamlit Cloud):** [https://creditscore-agent.streamlit.app](https://creditscore-agent.streamlit.app)
* **Documentación Académica:** Sección 9 - Desarrollo del Agente (Fase 1-3)

---

## 📊 Resumen de Rendimiento y Métricas (Fase 3)
El agente ha sido validado bajo estrictos estándares operativos y predictivos:
* **Capacidad Discriminativa (AUC-ROC):** **0.882** (Evaluado sobre un conjunto de test de $N = 5,000$ registros).
* **Tiempo de Inferencia Operativo:** **$\sim 0.017\text{ segundos}$** por solicitud en promedio.
* **Tasa de Automatización Segura:** **$20.0\%$** de decisiones directas bajo umbrales optimizados de riesgo.
* **Gobernanza AML:** **$100\%$ de efectividad** en el bloqueo y derivación analítica ante alertas restrictivas activas.

---

## 🏗️ Arquitectura del Agente
El sistema implementa una arquitectura basada en utilidad y modelos (AIMA / Framework IBM), estructurada en módulos independientes:

1. **Sensores e Inputs:** Captura variables financieras críticas (`loan_amnt`, `annual_inc`, `pagos_mensuales`, `retrasos_previos`, `historial_delitos`).
2. **Módulo de Percepción:** Cálculo autónomo de variables sintéticas (Ratio de endeudamiento **DTI** y estimador heurístico **FICO**).
3. **Modelo e Inferencia (Estado Interno):** Carga persistente en memoria del clasificador binario (`creditscore_model.pkl`) mediante `joblib` y `@st.cache_resource`.
4. **Módulo de Decisión de Gobernanza:** Motor híbrido que evalúa umbrales de probabilidad de mora junto con un filtro deductivo prioritario para prevención de lavado de activos (AML).
5. **Actuadores y Bitácora:** Emisión de veredictos visuales en la interfaz web y registro síncrono en el archivo de auditoría **`log_ejecucion.csv`**.

---

## 🔄 Diagrama de Secuencia UML (Resumen del Flujo)
```mermaid
sequenceDiagram
    autonumber
    actor Usuario
    participant UI as Interfaz Streamlit
    participant Agente as CREDITSCORE-AGENT
    participant Modelo as ML Model (.pkl)
    participant Log as Bitácora (CSV)

    Usuario->>UI: Ingresa Perceptos Financieros y AML
    UI->>Agente: Petición de Evaluación
    Agente->>Agente: Cálculo de DTI y FICO
    alt Excepción: Alerta AML Activa (historial_delitos = True)
        Agente->>Agente: Invalida ML por Regla de Gobernanza
        Agente->>UI: DERIVAR_ANALISTA (Riesgo Normativo)
    else Caso Normal: Sin Alerta AML
        Agente->>Modelo: predict_proba([vector])
        Modelo-->>Agente: Probabilidad de Mora (prob_mora)
        Agente->>Agente: Evaluación de Utilidad y Umbrales
        alt Riesgo Bajo (<= 0.30)
            Agente->>UI: APROBAR_CREDITO
        else Riesgo Alto (> 0.60)
            Agente->>UI: RECHAZAR_CREDITO
        else Zona Gris
            Agente->>UI: DERIVAR_ANALISTA
        end
    end
    Agente->>Log: Registro síncrono de transacción y tiempos
    UI-->>Usuario: Muestra Veredicto Visual y Métricas
