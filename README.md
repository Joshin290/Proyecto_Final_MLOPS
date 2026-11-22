# Proyecto Final MLOps - Prediccion de Precios Inmobiliarios

Plataforma MLOps end-to-end sobre Kubernetes (K3s) para ingesta incremental por lotes, validacion, deteccion de drift, entrenamiento con MLflow, promocion del modelo @champion, inferencia con FastAPI, interfaz en Streamlit, pruebas de carga con Locust y observabilidad con Prometheus y Grafana.

## Estructura
- `argocd-application.yaml`: GitOps con Argo CD.
- `.github/workflows/`: CI/CD (build Docker y actualizacion de tags).
- `k8s/`: manifiestos (`base/`, `infrastructure/`, `apps/`, `observability/`).
- `services/`: microservicios (`airflow/`, `fastapi/`, `mlflow/`, `streamlit/`, `locust/`).

## Despliegue paso a paso

1. Crear namespace y Secret (fuera de Git):
```bash
kubectl apply -f k8s/base/00-namespace.yaml
cp k8s/base/00-secrets.yaml.example k8s/base/00-secrets.yaml
# Editar k8s/base/00-secrets.yaml con credenciales reales
kubectl apply -f k8s/base/00-secrets.yaml
kubectl apply -f k8s/infrastructure/
kubectl apply -f k8s/apps/
kubectl apply -f k8s/observability/
kubectl exec -n mlops-pf deploy/airflow -- airflow dags trigger real_estate_mlops_dag

## Acceso a interfaces (port-forward)

| Servicio | Comando | URL local |
| :--- | :--- | :--- |
| Airflow | `kubectl port-forward svc/airflow-webserver -n mlops-pf 8080:8080` | `http://localhost:8080` |
| MLflow | `kubectl port-forward svc/mlflow-svc -n mlops-pf 5000:5000` | `http://localhost:5000` |
| FastAPI | `kubectl port-forward svc/fastapi-svc -n mlops-pf 8000:8000` | `http://localhost:8000/docs` |
| Streamlit | `kubectl port-forward svc/streamlit-svc -n mlops-pf 8501:8501` | `http://localhost:8501` |
| Locust | `kubectl port-forward svc/locust-svc -n mlops-pf 8089:8089` | `http://localhost:8089` |
| Prometheus | `kubectl port-forward svc/prometheus-svc -n mlops-pf 9090:9090` | `http://localhost:9090` |
| Grafana | `kubectl port-forward svc/grafana-svc -n mlops-pf 3000:3000` | `http://localhost:3000` |
