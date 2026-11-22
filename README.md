# Proyecto Final MLOps — Predicción de Precios Inmobiliarios

Plataforma **MLOps End-to-End** desplegada sobre **Kubernetes (K3s)** bajo un enfoque **GitOps**, diseñada para la ingesta incremental por lotes, validación de datos, detección de *drift*, entrenamiento continuo condicionado por reglas, registro y promoción automática de modelos (`@champion`), inferencia con recarga en caliente (*hot reload*), interfaz de usuario, pruebas de carga y observabilidad en tiempo real.

---

## Tabla de Contenidos

1. [Arquitectura General](#1-arquitectura-general)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Funcionamiento Detallado de Cada Servicio](#3-funcionamiento-detallado-de-cada-servicio)
4. [Ciclo de Vida: Despliegue de Código vs. Despliegue de Modelos](#4-ciclo-de-vida-despliegue-de-código-vs-despliegue-de-modelos)
5. [Procedimiento de Creación y Configuración del Proyecto](#5-procedimiento-de-creación-y-configuración-del-proyecto)
6. [Proceso de Ejecución Paso a Paso](#6-proceso-de-ejecución-paso-a-paso)
7. [Comandos Importantes y Referencia Operativa](#7-comandos-importantes-y-referencia-operativa)
8. [Inconvenientes Presentados y Soluciones (Troubleshooting)](#8-inconvenientes-presentados-y-soluciones-troubleshooting)

---

## 1. Arquitectura General

El sistema corre en un clúster **K3s** (nodo `NBAM20`), dentro del namespace **`mlops-pf`**, separando de manera limpia dos flujos automatizados:

* **Flujo CI/CD + GitOps (Infraestructura y Código):** Al realizar un `push` a la rama `main` sobre `services/**`, **GitHub Actions** construye las imágenes Docker, las publica en **Docker Hub** etiquetadas con el SHA del commit y actualiza automáticamente los manifiestos en `k8s/`. **Argo CD** monitorea la carpeta `k8s/` en `main` y sincroniza el estado deseado en Kubernetes (`prune: true`, `selfHeal: true`).
* **Flujo CT/CD de Modelos (Entrenamiento y Promoción Continua):** **Airflow** orquesta la ingesta incremental desde la API de datos, separa los datos en `RAW DATA` y `CLEAN DATA`, evalúa reglas de *drift*/calidad/volumen y, si corresponde, entrena un modelo candidato, lo registra en **MLflow** (artefactos en **MinIO** y metadatos en **PostgreSQL**), lo compara frente al modelo productivo (`@champion`) y, si lo supera, actualiza el alias `@champion` y notifica a **FastAPI** mediante `POST /reload-model` para actualizar el modelo en memoria sin reiniciar ni redesplegar contenedores.

---

## 2. Estructura del Proyecto

```text
Proyecto_Final_MLOPS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml                  # Pipeline CI/CD: build Docker Hub + bump GitOps en k8s/
├── k8s/
│   ├── base/                          # Namespace (mlops-pf), ConfigMaps y plantillas de Secrets
│   ├── infrastructure/                # Servicios base y plataforma de datos/ML
│   │   ├── 04-mlflow.yaml             # Deployment y Service de MLflow (puerto 5000)
│   │   ├── 05-airflow.yaml            # Deployment y Service de Airflow (puerto 8080)
│   │   └── ...                        # PostgreSQL, MinIO (9000/9001) y Data API
│   ├── apps/                          # Microservicios de aplicación y pruebas
│   │   ├── 06-fastapi.yaml            # API de inferencia y métricas Prometheus (puerto 8000)
│   │   ├── 07-streamlit.yaml          # Interfaz web de inferencia e historial (puerto 8501)
│   │   └── 08-locust.yaml             # Generador de pruebas de carga (puerto 8089)
│   └── observability/                 # Stack de monitoreo
│       ├── ...                        # Prometheus (puerto 9090)
│       └── ...                        # Grafana (puerto 3000) con dashboards aprovisionados
├── services/
│   ├── airflow/
│   │   ├── dags/
│   │   │   ├── real_estate_mlops_dag.py   # DAG principal MLOps (ingesta, drift, MLflow, promoción)
│   │   │   └── shap_explainer.py          # Generación de explicabilidad SHAP registrada en MLflow
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── fastapi/
│   │   ├── app/
│   │   │   └── main.py                # Endpoints /health, /predict, /reload-model y /metrics
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── locust/                        # Definición de usuarios virtuales y tráfico hacia FastAPI
│   ├── mlflow/                        # Imagen personalizada del servidor de tracking MLflow
│   └── streamlit/                     # Aplicación web para inferencia y consulta de trazabilidad
├── argocd-application.yaml            # Definición declarativa de la aplicación en Argo CD
├── .gitignore
└── README.md
```

---

## 3. Funcionamiento Detallado de Cada Servicio

### 3.1. Data API (`data-api`)
* **Imagen:** `cristiandiaz13/mlops-puj:data-api-pf-v2`
* **Función:** Simula la fuente externa de datos inmobiliarios entregando lotes incrementales bajo demanda al pipeline de Airflow para evaluar la evolución temporal de los datos y la aparición de *data drift*.

### 3.2. PostgreSQL
* **Función:** Base de datos relacional central que soporta tres responsabilidades principales:
  1. **Almacenamiento analítico y trazabilidad:** mantiene la separación entre `RAW DATA` (datos crudos ingeridos por lote) y `CLEAN DATA` (datos validados y preprocesados), además del historial de lotes, decisiones de entrenamiento y la tabla de auditoría `raw_data.inference_logs` donde FastAPI registra cada predicción.
  2. **Backend Store de MLflow:** guarda metadatos de experimentos, ejecuciones (*runs*), parámetros, métricas y versiones de modelos.
  3. **Metadata DB de Airflow:** almacena el estado de los DAGs, tareas y ejecuciones.

### 3.3. MinIO
* **Puertos:** `9000` (API compatible con S3) y `9001` (Consola web).
* **Función:** Almacenamiento de objetos (*Artifact Store*) utilizado por MLflow para persistir los binarios de los modelos entrenados (`sklearn`), gráficos de evaluación y artefactos de explicabilidad **SHAP**. Las credenciales (`MINIO_ROOT_USER` y `MINIO_ROOT_PASSWORD`) están externalizadas mediante `Secrets` de Kubernetes.

### 3.4. MLflow (`mlflow-svc:5000`)
* **Imagen:** `joshin290/mlops-pf-mlflow:<commit-sha>`
* **Función:** Actúa como **Tracking Server** y **Model Registry** bajo el experimento `real_estate_continuous_training`. Registra hiperparámetros, parámetros de preprocesamiento, métricas de desempeño, gráficos, artefactos SHAP y el versionado del modelo. Utiliza **aliases** (`@champion` para el modelo productivo vigente y `challenger_rejected` para candidatos descartados).

### 3.5. Apache Airflow (`airflow-svc:8080`)
* **Imagen:** `joshin290/mlops-pf-airflow:<commit-sha>`
* **DAG principal:** `real_estate_mlops_dag` (`schedule_interval=None`, `catchup=False`, `max_active_runs=1`).
* **Funcionamiento interno:** Aunque el disparo del DAG es bajo demanda (manual), **la decisión de entrenar y promover es completamente automática** mediante bifurcaciones y manejo de errores:
  1. **Ingesta y separación:** consulta un nuevo lote desde `data-api` y almacena `RAW DATA` y `CLEAN DATA` con trazabilidad por `batch_id`.
  2. **Validación y reglas de entrenamiento:** evalúa calidad de datos, cambios de distribución (*drift*), volumen, nuevas categorías o degradación de desempeño.
  3. **Entrenamiento y registro (`register_candidate_in_mlflow`):** entrena el modelo candidato y registra parámetros, métricas, gráficos y modelo en MLflow.
  4. **Comparación y decisión (`compare_with_production` -> `decide_promotion`):** carga el modelo actual `models:/<MODEL_NAME>@champion` y compara sus métricas frente al candidato.
  5. **Promoción y recarga en caliente:** si el candidato gana, mueve el alias `@champion` en MLflow a la nueva versión y envía una petición `POST` a `http://fastapi-svc:8000/reload-model` con el encabezado `X-Admin-Token`. Si no mejora, etiqueta la versión como `challenger_rejected`.
  6. **Explicabilidad (`shap_explainer.py`):** genera análisis SHAP del modelo `@champion` y los registra en MLflow bajo `shap_analysis`.

### 3.6. FastAPI (`fastapi-svc:8000`)
* **Imagen:** `joshin290/mlops-pf-fastapi:<commit-sha>`
* **Función:** Microservicio de inferencia en producción.
  * Al iniciar, consulta a MLflow (`MlflowClient.get_model_version_by_alias(MODEL_NAME, "champion")`) y carga `models:/<MODEL_NAME>@champion` en memoria.
  * **`GET /health`:** retorna el estado del servicio y la versión activa del modelo (`{"status": "healthy", "model_version": "..."}`).
  * **`POST /predict`:** realiza la predicción del precio inmobiliario, guarda el registro en `raw_data.inference_logs` e incrementa las métricas de Prometheus.
  * **`POST /reload-model`:** protegido por `ADMIN_RELOAD_TOKEN` (vía header `X-Admin-Token`), permite actualizar el modelo productivo en caliente (`force=True`) sin reiniciar el pod ni redesplegar la aplicación.
  * **`GET /metrics`:** expone métricas para Prometheus, incluyendo `inference_requests_total` (con etiquetas `status` y `model_version`) y `loaded_model_version`.

### 3.7. Streamlit (`streamlit-svc:8501`)
* **Imagen:** `joshin290/mlops-pf-streamlit:<commit-sha>`
* **Función:** Interfaz gráfica orientada al usuario final y a la auditoría del sistema. Permite enviar solicitudes de predicción a FastAPI y visualizar el historial claro de lotes procesados, decisiones de entrenamiento, métricas y registros de inferencia.

### 3.8. Locust (`locust-svc:8089`)
* **Imagen:** `joshin290/mlops-pf-locust:<commit-sha>`
* **Función:** Herramienta de pruebas de carga y estrés. Simula usuarios concurrentes enviando peticiones al servicio interno `http://fastapi-svc:8000` para validar la estabilidad, latencia, tasa de errores y rendimiento (RPS) de la API bajo carga.

### 3.9. Prometheus (`9090`) y Grafana (`3000`)
* **Imágenes:** `prom/prometheus:v2.54.1` y `grafana/grafana:11.2.0`.
* **Función:**
  * **Prometheus** realiza *scraping* periódico del endpoint `/metrics` de `fastapi-svc:8000` y almacena las series de tiempo.
  * **Grafana** consume Prometheus como fuente de datos y presenta el dashboard **`MLOps FastAPI & Locust Observability`**, con paneles de:
    * Peticiones por Segundo (RPS)
    * Latencia P95 de Inferencia (segundos)
    * Versión Activa del Modelo (`@champion`)
    * Tasa de Errores de Inferencia
    * Consumo de RAM y CPU de FastAPI

### 3.10. GitHub Actions (CI/CD) y Argo CD (GitOps)
* **GitHub Actions (`.github/workflows/ci-cd.yml`):**
  * Se dispara en `push` a `main` cuando cambian rutas en `services/**` o `.github/workflows/ci-cd.yml` (y manualmente vía `workflow_dispatch`).
  * Ejecuta una matriz paralela (`build-and-push`) para los 5 microservicios (`mlflow`, `airflow`, `fastapi`, `streamlit`, `locust`), publicando en Docker Hub las etiquetas `:<github.sha>` y `:latest`.
  * En el job `update-k8s-manifests`, actualiza los tags de imagen y la variable `# git-commit-sha` en los YAML de `k8s/` y realiza un commit automático a `main` como `github-actions[bot]` con `[skip ci]` para evitar bucles.
* **Argo CD (`argocd-application.yaml`):**
  * Vigila `https://github.com/Joshin290/Proyecto_Final_MLOPS.git` en `targetRevision: main` y `path: k8s`.
  * Con `automated` (`prune: true`, `selfHeal: true`), aplica cualquier cambio en `k8s/` sobre el namespace `mlops-pf` y revierte desviaciones manuales en el clúster.

---

## 4. Ciclo de Vida: Despliegue de Código vs. Despliegue de Modelos

Es fundamental distinguir cuándo actúa cada automatización:

| Escenario | Quién actúa | ¿Pasa por GitHub Actions / Argo CD? |
| :--- | :--- | :--- |
| **Cambio de código en microservicios** (`services/**`) | GitHub Actions construye imágenes y actualiza `k8s/` -> Argo CD sincroniza en K3s. | **Sí** (CI/CD + GitOps completo). |
| **Cambio directo en manifiestos** (`k8s/**`) | Argo CD detecta el commit en `main` y aplica el estado en `mlops-pf`. | **Solo Argo CD** (no dispara build de imágenes). |
| **Entrenamiento y promoción de un nuevo modelo** | Airflow ejecuta `real_estate_mlops_dag` -> MLflow actualiza `@champion` -> FastAPI ejecuta `POST /reload-model`. | **No** (recarga en caliente sin redespliegue). |

---

## 5. Procedimiento de Creación y Configuración del Proyecto

1. **Diseño y contenerización de microservicios:** se crearon los directorios independientes bajo `services/` (`airflow`, `fastapi`, `mlflow`, `streamlit`, `locust`), cada uno con su `Dockerfile` y `requirements.txt` fijando versiones compatibles (por ejemplo `mlflow==2.16.2`).
2. **Externalización de credenciales y secretos:** se retiraron contraseñas y tokens *hardcodeados* del código y manifiestos (como `ADMIN_RELOAD_TOKEN`, credenciales de PostgreSQL y `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`), gestionándolos mediante `Secrets` en Kubernetes y variables de entorno.
3. **Definición de manifiestos Kubernetes (`k8s/`):** se estructuraron por capas (`base`, `infrastructure`, `apps`, `observability`) especificando `Deployments`, `Services`, *probes* de salud, límites/solicitudes de recursos (`requests`/`limits`) y almacenamiento persistente.
4. **Configuración de CI/CD en GitHub Actions:** se registraron los secretos `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN` en el repositorio de GitHub y se configuró `.github/workflows/ci-cd.yml` con permisos `contents: write` para actualizar los manifiestos de `k8s/` con cada SHA.
5. **Despliegue de Argo CD y vinculación GitOps:** se aplicó `argocd-application.yaml` en el clúster K3s para vincular el repositorio público de GitHub con el namespace `mlops-pf` en modo sincronización automática (`Healthy` / `Synced`).

---

## 6. Proceso de Ejecución Paso a Paso

### Paso 1: Verificar el clúster y la sincronización GitOps
Comprobar que todos los pods del namespace `mlops-pf` estén en estado `Running` y que las imágenes correspondan al último commit sincronizado por Argo CD:

```bash
kubectl get pods -n mlops-pf
kubectl get deploy -n mlops-pf -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'
```

*(Si es el despliegue inicial en un clúster limpio, aplicar primero los secretos locales y luego `kubectl apply -f argocd-application.yaml`.)*

### Paso 2: Exponer los servicios localmente (`port-forward`)
Dado que los servicios corren internos en K3s, se levantan túneles locales desde el nodo `NBAM20` para acceder a las interfaces web desde el navegador:

```bash
kubectl port-forward svc/airflow-svc -n mlops-pf --address 0.0.0.0 8080:8080 > /tmp/pf-airflow.log 2>&1 &
kubectl port-forward svc/fastapi-svc -n mlops-pf --address 0.0.0.0 8000:8000 > /tmp/pf-fastapi.log 2>&1 &
kubectl port-forward svc/mlflow-svc -n mlops-pf --address 0.0.0.0 5000:5000 > /tmp/pf-mlflow.log 2>&1 &
kubectl port-forward svc/minio-svc -n mlops-pf --address 0.0.0.0 9001:9001 > /tmp/pf-minio.log 2>&1 &
kubectl port-forward svc/streamlit-svc -n mlops-pf --address 0.0.0.0 8501:8501 > /tmp/pf-streamlit.log 2>&1 &
kubectl port-forward svc/locust-svc -n mlops-pf --address 0.0.0.0 8089:8089 > /tmp/pf-locust.log 2>&1 &
kubectl port-forward svc/prometheus-svc -n mlops-pf --address 0.0.0.0 9090:9090 > /tmp/pf-prometheus.log 2>&1 &
kubectl port-forward svc/grafana-svc -n mlops-pf --address 0.0.0.0 3000:3000 > /tmp/pf-grafana.log 2>&1 &
```

> **Nota:** Si algún nombre de servicio difiere en el clúster (por ejemplo `minio` o `grafana` sin `-svc`), validar previamente con `kubectl get svc -n mlops-pf`.

**Accesos web:**
* **Airflow:** `http://localhost:8080`
* **FastAPI (Swagger / Health):** `http://localhost:8000/docs` y `http://localhost:8000/health`
* **MLflow:** `http://localhost:5000`
* **MinIO Console:** `http://localhost:9001`
* **Streamlit:** `http://localhost:8501`
* **Locust:** `http://localhost:8089`
* **Prometheus:** `http://localhost:9090`
* **Grafana:** `http://localhost:3000`

### Paso 3: Ejecutar el pipeline de entrenamiento continuo en Airflow
1. Ingresar a **Airflow** (`http://localhost:8080`) y activar/disparar manualmente (`Trigger DAG`) el DAG **`real_estate_mlops_dag`**.
2. Observar en la vista *Graph* la ejecución de ingesta, separación `RAW`/`CLEAN`, validación, análisis de *drift*, entrenamiento, registro en **MLflow** y decisión de promoción.
3. Verificar en **MLflow** (`http://localhost:5000`) el experimento `real_estate_continuous_training`, los artefactos en **MinIO** y el alias `@champion` asignado a la versión ganadora.
4. Confirmar que **FastAPI** actualizó el modelo en caliente:
   ```bash
   curl -s http://localhost:8000/health && echo ""
   ```

### Paso 4: Validar inferencia e historial en Streamlit
1. Abrir **Streamlit** (`http://localhost:8501`).
2. Realizar predicciones de prueba y revisar las pestañas de historial de lotes, decisiones de entrenamiento y logs de inferencia.

### Paso 5: Ejecutar pruebas de carga con Locust y monitorear en Grafana / Prometheus
1. Abrir **Locust** (`http://localhost:8089`).
2. Configurar el campo **Host** apuntando al servicio interno del clúster: **`http://fastapi-svc:8000`** (no `localhost:8000`), definir número de usuarios y *spawn rate*, e iniciar la prueba.
3. Abrir **Grafana** (`http://localhost:3000`) en el dashboard **`MLOps FastAPI & Locust Observability`**.
4. Ajustar la ventana de tiempo superior derecha a **`Last 15 minutes`** (o **`Last 5 minutes`**) con refresco automático de **`5s`** para observar en tiempo real las curvas de RPS, Latencia P95, versión del modelo (`@champion`), tasa de errores y consumo de CPU/RAM.
5. Opcionalmente, verificar en **Prometheus** (`http://localhost:9090/targets`) que `fastapi-svc:8000` figure `UP` y consultar `rate(inference_requests_total[1m])`.

---

## 7. Comandos Importantes y Referencia Operativa

### Estado del clúster, servicios e imágenes
```bash
# Ver pods, IPs y estado general
kubectl get pods -n mlops-pf -o wide

# Ver servicios y puertos expuestos
kubectl get svc -n mlops-pf

# Confirmar que las imágenes desplegadas coinciden con el último commit SHA
kubectl get deploy -n mlops-pf -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'
```

### Verificación de salud, métricas y secretos
```bash
# Comprobar salud y versión @champion activa en FastAPI
curl -s http://localhost:8000/health && echo ""

# Verificar métricas Prometheus expuestas directamente por FastAPI
curl -s http://localhost:8000/metrics | grep -E "^inference_requests_total|^loaded_model_version"

# Consultar credenciales activas de MinIO desde su pod
MINIO_POD=$(kubectl get pods -n mlops-pf -o name | grep -i minio | head -n 1)
kubectl exec -n mlops-pf "$MINIO_POD" -- env | grep -iE "MINIO_ROOT_USER|MINIO_ROOT_PASSWORD"

# Decodificar un Secret específico de Kubernetes
kubectl get secret <nombre-del-secret> -n mlops-pf -o jsonpath='{.data}' | jq 'with_entries(.value |= @base64d)'
```

### Gestión de *port-forwards*
```bash
# Verificar respuesta HTTP de los puertos locales
for p in 8000 8080 8089 5000 9001 3000 9090; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 4 http://localhost:$p)
  echo "Puerto $p -> HTTP $code"
done

# Reiniciar un port-forward colgado (ejemplo con puerto 5000 de MLflow)
pkill -f "port-forward.*5000:5000" 2>/dev/null || true
kubectl port-forward svc/mlflow-svc -n mlops-pf --address 0.0.0.0 5000:5000 > /tmp/pf-mlflow.log 2>&1 &
```

### Mantenimiento de disco en nodo K3s (`NBAM20`)
```bash
# Revisar espacio disponible en /var
df -h /var

# Podar imágenes y capas sin uso en containerd (seguro para pods en ejecución)
sudo /usr/local/bin/k3s crictl rmi --prune
```

### Consultas PromQL útiles (Prometheus)
```promql
# Peticiones por segundo (RPS)
rate(inference_requests_total[1m])

# Total acumulado de inferencias por estado y versión de modelo
inference_requests_total

# Versión numérica del modelo @champion cargado en memoria
loaded_model_version
```

---

## 8. Inconvenientes Presentados y Soluciones (Troubleshooting)

### 8.1. `DiskPressure` y error `no space left on device` en `/var` (`NBAM20`)
* **Síntoma:** Durante despliegues sucesivos (*rolling updates*), varios pods fueron desalojados (*Evicted*) por presión de `ephemeral-storage` (`DiskPressure`) o fallaban al extraer capas de imágenes con el error `no space left on device`.
* **Causa raíz:** El nodo `NBAM20` cuenta con una partición `/var` de **15 GB**, donde K3s/containerd almacena imágenes y *snapshots* de `overlayfs` (`/var/lib/rancher/k3s/...`). Debido a que el flujo CI/CD etiqueta cada nueva compilación con el SHA completo del commit (`joshin290/mlops-pf-*:<sha>`), la acumulación de versiones anteriores saturaba el umbral de almacenamiento efímero del nodo.
* **Solución aplicada:**
  1. Monitoreo preventivo del uso de disco con `df -h /var` antes y después de despliegues (actuando cuando supera el 75–80 %).
  2. Poda controlada de imágenes y capas huérfanas mediante `sudo /usr/local/bin/k3s crictl rmi --prune`, liberando espacio sin afectar los pods activos (`Running`) ni los volúmenes persistentes de PostgreSQL/MinIO.
  3. **Decisión sobre caché:** Se mantuvo el pipeline de GitHub Actions (`ci-cd.yml`) sin caché Docker para garantizar compilaciones limpias y reproducibles, concentrando la limpieza únicamente en el nodo `NBAM20`.

### 8.2. Caída o bloqueo de túneles `kubectl port-forward` tras despliegues
* **Síntoma:** Interfaz web inaccesible o cargando indefinidamente (por ejemplo en puertos `8080`, `8089` o `5000`) después de una sincronización de Argo CD o reinicio de un pod.
* **Causa raíz:** Los túneles `kubectl port-forward` son procesos locales en la sesión del nodo y no forman parte del estado gestionado por Argo CD; cuando un pod se reemplaza durante un despliegue, el túnel anterior puede quedar colgado apuntando a un socket cerrado.
* **Solución aplicada:** Diagnóstico mediante `curl` y revisión de logs (`/tmp/pf-*.log`), finalización del túnel antiguo (`pkill -f "port-forward..."`) y reactivación limpia del reenvío de puertos.

### 8.3. Gráficas aparentemente estáticas en Grafana durante pruebas de carga con Locust
* **Síntoma:** Al iniciar Locust, el dashboard `MLOps FastAPI & Locust Observability` en Grafana parecía permanecer plano o sin movimiento.
* **Causa raíz:** La ventana temporal de Grafana estaba configurada en **`Last 6 hours`**, comprimiendo los picos recientes de 1–5 minutos contra el borde derecho de la gráfica. Adicionalmente, en Locust es necesario apuntar al DNS interno del clúster y no a `localhost`.
* **Solución aplicada:**
  1. Confirmar en Locust que el **Host** sea `http://fastapi-svc:8000` y que las peticiones sumen con `0%` de fallos.
  2. Validar en Prometheus (`Status -> Targets`) que `fastapi-svc:8000` esté `UP`.
  3. Cambiar el selector temporal de Grafana a **`Last 15 minutes`** (o **`Last 5 minutes`**) con refresco de **`5s`**, visualizando inmediatamente las curvas de RPS, latencia P95 y consumo de recursos.

### 8.4. Consideración de escalado horizontal con `/reload-model`
* **Observación técnica:** En la configuración actual (1 réplica de FastAPI), el `POST /reload-model` disparado por Airflow actualiza inmediatamente el modelo en memoria. Como mejora futura, si se escala `fastapi` a múltiples réplicas, se recomienda propagar la recarga a todas las réplicas (vía *polling* periódico en segundo plano, evento compartido o `kubectl rollout restart`) para asegurar consistencia en todos los pods detrás de `fastapi-svc`.