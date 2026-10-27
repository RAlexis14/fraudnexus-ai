# FraudNexus AI

**Detección de fraude en transacciones financieras mediante minería de datos, LightGBM y ONNX Runtime.**

Proyecto académico desarrollado para el componente práctico del examen complexivo de Ingeniería en Sistemas de Información de la **Universidad Central del Ecuador (2026)**.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

**Repositorio:** <https://github.com/RAlexis14/fraudnexus-ai>  
**Rama de entrega:** `main`  
**Autor:** Rommel Alexis Pachacama Guala

> **Alcance:** prototipo académico para análisis, clasificación e inferencia de riesgo. Las respuestas `ALLOW` y `REVIEW` son decisiones simuladas; el sistema no está conectado a un switch bancario real ni autoriza cargos.

## Contenido

1. [Descripción y objetivos](#descripción-y-objetivos)
2. [Resultados y metodología](#resultados-y-metodología)
3. [Tecnologías y arquitectura](#tecnologías-y-arquitectura)
4. [Instalación rápida con Docker](#instalación-rápida-con-docker)
5. [Uso del dashboard y API](#uso-del-dashboard-y-api)
6. [Pruebas automatizadas y validación ONNX](#pruebas-automatizadas-y-validación-onnx)
7. [Entorno Python para análisis y desarrollo](#entorno-python-para-análisis-y-desarrollo)
8. [Ejecución local sin Docker](#ejecución-local-sin-docker)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Reproducibilidad y datos](#reproducibilidad-y-datos)
11. [Solución de problemas](#solución-de-problemas)
12. [Versionamiento y actualización](#versionamiento-y-actualización)

## Descripción y objetivos

FraudNexus AI procesa transacciones con **nueve campos de entrada** para identificar operaciones que ameriten revisión. Su flujo utiliza un preprocesador persistido que transforma la entrada en **15 características `float32`** y un modelo LightGBM exportado a ONNX, ejecutado con ONNX Runtime desde una API REST.

Funciones implementadas:

- **Inteligencia transaccional:** comparación de tasas de fraude por canal, actividad comercial, horario y señales de comportamiento.
- **Laboratorio de modelos:** consulta de reportes de comparación y validación del entrenamiento.
- **Evaluación individual:** formulario de nueve variables y decisión `ALLOW` / `REVIEW`.
- **Análisis CSV:** validación de columnas, predicción por lotes y descarga de resultados.
- **Switch simulado:** secuencia de peticiones, bitácora de decisiones y tiempos.
- **Arquitectura y auditoría:** visualización del entorno y evidencias técnicas.

El caso académico prioriza **Recall de fraude superior al 90 %** y una latencia de respuesta **inferior a 100 ms**, controlando falsas alertas en el protocolo experimental.

## Resultados y metodología

Se siguieron las seis fases **CRISP-DM**: comprensión del negocio, comprensión de los datos, preparación, modelado, evaluación y despliegue.

### Conjunto de datos del ejercicio

| Característica | Valor |
|---|---:|
| Transacciones | 50.000 |
| Operaciones legítimas | 49.800 |
| Fraudes | 200 (0,4 %) |
| Predictores originales | 9 |
| Características después del preprocesamiento | 15 |
| Desarrollo / evaluación reservada | 40.000 / 10.000 |

### Entrenamiento y comparación

Se consideraron Random Forest, XGBoost y LightGBM, con **pesos de clase, SMOTE y ADASYN**; Isolation Forest se estudió como detector exploratorio de anomalías. La comparación de nueve configuraciones supervisadas empleó **validación cruzada estratificada K=5**.

| Resultado documentado | Valor |
|---|---:|
| LightGBM + pesos, desarrollo K=5 | Recall 98,125 %; F1 99,054 % |
| Fraudes identificados en desarrollo | 157/160; FP=0 |
| Holdout independiente | TP=40; TN=9.960; FP=0; FN=0 |
| Benchmark de servicio HTTP local (200 solicitudes) | p95 = 62,071 ms |
| Inferencia ONNX instrumentada tras calentamiento (100 medidas) | p95 = 0,2187 ms |
| Concordancia de decisiones Python/ONNX sobre desarrollo | 40.000/40.000 |

**Interpretación:** los resultados corresponden al conjunto académico y a los protocolos indicados. El tiempo HTTP mide un alcance diferente del tiempo instrumentado de ONNX; no deben equipararse ni extrapolarse directamente a operación bancaria real.

Los archivos que respaldan estas cifras están en [`reports/`](reports/).

## Tecnologías y arquitectura

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Procesamiento | pandas, NumPy, scikit-learn |
| Modelado | LightGBM, XGBoost, Random Forest, Isolation Forest |
| Balanceo | imbalanced-learn (SMOTE / ADASYN) |
| Motor de inferencia | ONNX Runtime |
| API y validación | FastAPI, Pydantic, Uvicorn |
| Interfaz | Streamlit, Plotly |
| Contenedores | Docker Compose |
| Pruebas | pytest |

```text
Navegador
   |
   v
Streamlit (http://localhost:8501)
   |
   | API_BASE_URL=http://api:8000  [red Docker interna]
   v
FastAPI (http://localhost:8000)
   |
   +--> Validación Pydantic (9 campos)
   +--> preprocessor.joblib (9 -> 15 características float32)
   +--> ONNX Runtime + model_margin_final.onnx
   +--> Comparación de margen contra umbral congelado
   '--> JSON: score, margin, is_fraud, decision, tiempos
```

Los dos servicios se definen en [`compose.yaml`](compose.yaml). El selector del modelo de la API es `FRAUDNEXUS_MODEL=lightgbm_candidate`.

## Instalación rápida con Docker

### 1. Requisitos

- Git.
- Docker Desktop con Docker Compose, iniciado y operativo.
- Puertos **8000** y **8501** disponibles.
- Los tres artefactos finales de LightGBM en las rutas indicadas abajo.

> **Importante:** para iniciar el prototipo **no es necesario reentrenar** LightGBM. Se utilizan los artefactos finales persistidos. Si tu clonación no los contiene, solicita al autor los artefactos de la versión evaluada; no los reemplaces por archivos inventados ni reentrenes para suplirlos.

### 2. Clonar la versión principal

En **PowerShell**:

```powershell
git clone https://github.com/RAlexis14/fraudnexus-ai.git
cd fraudnexus-ai
git switch main
```

### 3. Verificar los artefactos de inferencia

El Dockerfile de la API requiere estos archivos:

```text
models/candidates/lightgbm_weights/
├── metadata.json
├── model_margin_final.onnx
└── preprocessor.joblib
```

Comprobarlos en PowerShell:

```powershell
Test-Path .\models\candidates\lightgbm_weights\metadata.json
Test-Path .\models\candidates\lightgbm_weights\model_margin_final.onnx
Test-Path .\models\candidates\lightgbm_weights\preprocessor.joblib
```

Los tres comandos deben devolver `True`. Si alguno es `False`, **no continúes con el build**: falta una dependencia necesaria del modelo seleccionado. La variante evaluada usa un ONNX de **44.236 bytes** y un preprocesador de **4.174 bytes**; verifica la versión mediante los hashes documentados en [`reports/lightgbm_final_holdout.json`](reports/lightgbm_final_holdout.json).

### 4. Construir y levantar los servicios

```powershell
docker compose config
docker compose up -d --build
docker compose ps
```

El servicio `api` tiene un *healthcheck* HTTP; el `dashboard` depende de que la API esté saludable.

### 5. Acceder a la aplicación

| Servicio | Dirección |
|---|---|
| Dashboard Streamlit | <http://localhost:8501> |
| Swagger UI | <http://localhost:8000/docs> |
| Estado del servicio | <http://localhost:8000/health> |

Para revisar el modelo realmente cargado:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

La respuesta debe incluir `"status": "ok"` y `"model": "lightgbm_candidate"`.

### 6. Apagar / reconstruir

```powershell
# Detener los servicios
docker compose down

# Iniciar nuevamente
docker compose up -d

# Reconstruir imágenes después de cambios en código o Dockerfiles
docker compose up -d --build

# Revisar logs si hay errores
docker compose logs --tail=100 api
docker compose logs --tail=100 dashboard
```

## Uso del dashboard y API

### Dashboard

Visita <http://localhost:8501>. En el menú lateral están disponibles: **Centro de control**, **Inteligencia transaccional**, **Evaluación individual**, **Análisis de CSV**, **Switch simulado**, **Laboratorio de modelos** y **Arquitectura y auditoría**.

### Endpoints REST

| Método | Endpoint | Finalidad |
|---|---|---|
| `GET` | `/health` | Estado del servicio, variante de modelo y umbral |
| `POST` | `/predict` | Clasificación de una transacción |
| `POST` | `/predict/batch` | Clasificación de 1 a 1.000 transacciones |

#### Ejemplo de predicción individual

**Cuerpo JSON:**

```json
{
  "monto_transaccion": 65.2,
  "canal_transaccion": "POS",
  "categoria_comercio": "Retail",
  "velocidad_tx_10m": 0,
  "velocidad_desplazamiento_kmh": 10.0,
  "ratio_desviacion_monto": 1.1,
  "pais_emisor_coincide": 1,
  "distancia_residencia_km": 3.0,
  "hora_dia": 14
}
```

**Enviar desde PowerShell:**

```powershell
$transaction = @{
    monto_transaccion = 65.2
    canal_transaccion = 'POS'
    categoria_comercio = 'Retail'
    velocidad_tx_10m = 0
    velocidad_desplazamiento_kmh = 10.0
    ratio_desviacion_monto = 1.1
    pais_emisor_coincide = 1
    distancia_residencia_km = 3.0
    hora_dia = 14
}

Invoke-RestMethod -Uri 'http://localhost:8000/predict' `
    -Method Post `
    -ContentType 'application/json' `
    -Body ($transaction | ConvertTo-Json)
```

**Resultado de referencia documentado:** `is_fraud=false`, `decision="ALLOW"`, margen aproximado `-12.72158`, umbral de margen aproximado `16.27694`. Los tiempos se miden en cada ejecución y pueden variar. El campo `score` es una transformación sigmoidal de presentación, **no una probabilidad calibrada**.

Para el endpoint por lotes, el cuerpo se estructura como:

```json
{
  "transactions": [
    {
      "monto_transaccion": 65.2,
      "canal_transaccion": "POS",
      "categoria_comercio": "Retail",
      "velocidad_tx_10m": 0,
      "velocidad_desplazamiento_kmh": 10.0,
      "ratio_desviacion_monto": 1.1,
      "pais_emisor_coincide": 1,
      "distancia_residencia_km": 3.0,
      "hora_dia": 14
    }
  ]
}
```

### Archivos CSV

El módulo **Análisis de CSV** acepta archivos UTF-8 con los nueve predictores, valida el esquema y permite descargar las predicciones. Si el archivo incluye `es_fraude` puede calcular métricas respecto de esa etiqueta; si no la incluye, entrega inferencias sin atribuirles exactitud verificada.

## Pruebas automatizadas y validación ONNX

Las pruebas se ejecutan con un entorno Python local configurado según el siguiente apartado. **No es obligatorio instalarlas para utilizar la distribución Docker**.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m scripts.verify_onnx_input_contract
```

La auditoría del prototipo registró **22 pruebas aprobadas** y cinco comprobaciones satisfactorias del contrato ONNX, pero una clonación nueva debe producir sus propios resultados. La verificación ONNX requiere los artefactos candidatos.

Reportes útiles:

- [`reports/onnx_input_contract.json`](reports/onnx_input_contract.json): nueve variables originales, quince transformadas y tipo `tensor(float)`.
- [`reports/lightgbm_final_holdout.json`](reports/lightgbm_final_holdout.json): resultados independientes y hashes.
- [`reports/resampling_comparison.json`](reports/resampling_comparison.json): modelos y configuraciones K=5.
- [`reports/lightgbm_latency_benchmark.json`](reports/lightgbm_latency_benchmark.json): protocolo HTTP.
- [`reports/onnx_submillisecond_benchmark.json`](reports/onnx_submillisecond_benchmark.json): tiempos de ejecución ONNX instrumentados.

## Entorno Python para análisis y desarrollo

Para trabajar con scripts y pruebas en Windows (independientemente del despliegue Docker):

```powershell
# Desde la raíz del repositorio, con Python 3.12 instalado
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/training.txt
python -m pip install -r requirements/api.txt
python -m pip install -r requirements/dashboard.txt
python -m pip install -e .

# Verificación
python -m pytest -q
python -m scripts.verify_onnx_input_contract
```

> Los scripts que reconstruyen el entrenamiento o recalculan análisis sobre el CSV académico requieren el archivo original proporcionado en el ejercicio. No es parte de la instalación mínima para inferencia.


## Ejecución local sin Docker

FraudNexus AI también puede ejecutarse directamente en Windows mediante
Python 3.12 y un entorno virtual, sin utilizar contenedores.

### 1. Preparar el entorno

Desde PowerShell, en la carpeta raíz del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements/api.txt
python -m pip install -r requirements/dashboard.txt
python -m pip install -r requirements/training.txt
python -m pip install -e .
```

Antes de iniciar el sistema, verificar que existan:

- `models/candidates/lightgbm_weights/metadata.json`
- `models/candidates/lightgbm_weights/model_margin_final.onnx`
- `models/candidates/lightgbm_weights/preprocessor.joblib`

### 2. Terminal 1 — Iniciar la API

Abrir la primera terminal de PowerShell en la raíz del proyecto:

```powershell
.\.venv\Scripts\Activate.ps1

$env:FRAUDNEXUS_MODEL = "lightgbm_candidate"

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Comprobar el servicio en:

- API: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

La respuesta de `/health` debe indicar `status: ok`
y `model: lightgbm_candidate`.

### 3. Terminal 2 — Iniciar el dashboard

Mantener abierta la terminal de la API.

Abrir una segunda terminal de PowerShell, también en la raíz
del proyecto, y ejecutar:

```powershell
.\.venv\Scripts\Activate.ps1

$env:API_BASE_URL = "http://127.0.0.1:8000"

python -m streamlit run dashboard/app.py --server.port 8501
```

Abrir el dashboard en:

http://localhost:8501

La interfaz se comunica con la API local mediante HTTP.

### 4. Comprobar el funcionamiento

Desde el dashboard se pueden ejecutar predicciones individuales,
procesar archivos CSV y utilizar el switch transaccional simulado.

Para ejecutar las pruebas automatizadas en otra terminal:

```powershell
.\.venv\Scripts\Activate.ps1

python -m pytest -q
python -m scripts.verify_onnx_input_contract
```

### 5. Detener el proyecto

En cada terminal, presionar `Ctrl + C`.

### Diferencia entre los dos métodos

- **Docker Compose:** ejecuta API y dashboard mediante contenedores.
- **Entorno virtual:** ejecuta los dos servicios directamente con Python.

Ambos utilizan los mismos artefactos de inferencia y el mismo
modelo LightGBM seleccionado.

**Nota:** no ejecutar simultáneamente Docker y la versión local
sobre los mismos puertos 8000 y 8501.



## Estructura del repositorio

```text
fraudnexus-ai/
├── api/                         # Endpoints y motor de inferencia
├── dashboard/                   # Interfaz Streamlit
├── docker/                      # Dockerfiles de API y dashboard
├── models/                      # Modelos y preprocesadores
│   └── candidates/
│       └── lightgbm_weights/    # Artefactos finales requeridos por la API
├── reports/                     # Reportes JSON, CSV y gráficas experimentales
│   ├── fraud_patterns/
│   └── feature_reduction/
├── requirements/                # Dependencias por componente
├── scripts/                     # Verificación, comparación y benchmarks
├── src/fraudnexus/              # Procesamiento y entrenamiento
├── tests/                       # Pruebas automatizadas
├── compose.yaml                 # Orquestación local
├── pyproject.toml
└── README.md
```

## Reproducibilidad y datos

- **Código:** rama de entrega `main`. La rama `test` fue utilizada para integración y verificación durante el desarrollo.
- **Modelo desplegado:** `lightgbm_candidate`, con preprocesamiento persistido y umbral fijo de margen.
- **Trazabilidad:** hashes SHA-256 en el reporte final permiten identificar datos y artefactos de la evaluación.
- **Conjunto oficial:** se obtuvo del ejercicio académico; no se sustituye por datos sintéticos generados con un script del repositorio.
- **Privacidad:** no publicar `.env`, claves, información bancaria real ni archivos privados.
- **Licencias y distribución:** consultar las condiciones institucionales antes de redistribuir el CSV del ejercicio.

**Comprobar si los artefactos están versionados:**

```powershell
git branch --show-current
git log -1 --oneline
git ls-files models/candidates/lightgbm_weights/
```

Si `git ls-files` no enumera los tres archivos, **no se puede afirmar que una clonación del repositorio reproduzca automáticamente la inferencia LightGBM**. Quien mantenga el proyecto debe incorporarlos conforme a la política de distribución correspondiente o proporcionar un mecanismo documentado y verificable para obtenerlos.

## Solución de problemas

| Problema | Acción recomendada |
|---|---|
| `COPY failed` al construir la API | Verifica `models/candidates/lightgbm_weights/` y los tres artefactos; no basta tenerlos mencionados en el README. |
| Docker Desktop no inicia por WSL desactualizado | En PowerShell con privilegios adecuados, ejecuta `wsl --update`, reinicia Docker Desktop y comprueba `docker info`. |
| No carga `localhost:8501` | Ejecuta `docker compose ps` y `docker compose logs --tail=100 dashboard`. |
| `/health` responde error | Revisa `docker compose logs --tail=100 api`; comprueba el modelo y sus metadatos. |
| Puerto 8000 o 8501 ocupado | Libera el puerto o modifica el mapeo de `compose.yaml` de forma coherente. |
| Tras un `git pull` siguen viéndose cambios antiguos | Reconstruye con `docker compose up -d --build` si hubo cambios en código, dependencias o Dockerfiles. |
| Falla el contrato 9→15 | Comprueba que `metadata.json`, `preprocessor.joblib` y `model_margin_final.onnx` pertenezcan a la misma versión. |
| PowerShell no permite activar `.venv` | Para la sesión actual, ajusta la política de ejecución solo si la administración del equipo lo permite. |

## Versionamiento y actualización

Flujo de desarrollo utilizado:

```text
test  --> validación y revisión --> main (entrega)
```

Después de integrar cambios en `main`, los archivos de GitHub **no modifican automáticamente los contenedores que ya están ejecutándose en tu equipo**. Para desplegar cambios de código localmente:

```powershell
git switch main
git pull origin main
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

Un cambio exclusivamente documental en `README.md` no requiere reconstruir imágenes Docker.

---

**FraudNexus AI · Universidad Central del Ecuador · 2026**  
**Autor:** Rommel Alexis Pachacama Guala
