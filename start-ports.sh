#!/usr/bin/env bash
set -euo pipefail

NS="mlops-pf"

stop_all() {
  echo "Deteniendo port-forwards activos..."
  pkill -f "kubectl port-forward.*-n ${NS}" 2>/dev/null || true
  pkill -f "kubectl port-forward.*-n argocd" 2>/dev/null || true
  echo "Túneles detenidos."
}

if [[ "${1:-}" == "stop" ]]; then
  stop_all
  exit 0
fi

resolve_svc() {
  local ns="$1"
  local pattern="$2"
  kubectl get svc -n "$ns" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep -iE "$pattern" | head -n 1 || true
}

start_pf() {
  local label="$1"
  local ns="$2"
  local pattern="$3"
  local local_port="$4"
  local target_port="$5"

  local svc
  svc=$(resolve_svc "$ns" "$pattern")

  if [[ -z "$svc" ]]; then
    printf "  [WARN] %-12s -> No se encontró servicio (%s) en %s\n" "$label" "$pattern" "$ns"
    return
  fi

  nohup kubectl port-forward "svc/${svc}" -n "$ns" --address 0.0.0.0 "${local_port}:${target_port}" \
    > "/tmp/pf-${label}.log" 2>&1 &
  printf "  [OK]   %-12s -> svc/%-20s (localhost:%s -> %s)\n" "$label" "$svc" "$local_port" "$target_port"
}

stop_all
echo ""
echo "Levantando port-forwards para ${NS} y argocd..."

start_pf "airflow"    "$NS"    "airflow"         8080 8080
start_pf "mlflow"     "$NS"    "mlflow"          5000 5000
start_pf "minio"      "$NS"    "minio"           9001 9001
start_pf "fastapi"    "$NS"    "fastapi"         8000 8000
start_pf "data-api"   "$NS"    "data-api"        8002 80
start_pf "streamlit"  "$NS"    "streamlit"       8501 8501
start_pf "locust"     "$NS"    "locust"          8089 8089
start_pf "prometheus" "$NS"    "prometheus"      9090 9090
start_pf "grafana"    "$NS"    "grafana"         3000 3000
start_pf "argocd"     "argocd" "^argocd-server$" 8081 443

echo ""
echo "Esperando 3 segundos para verificar puertos..."
sleep 3
echo ""

declare -A URLS=(
  ["Airflow"]="http://localhost:8080"
  ["MLflow"]="http://localhost:5000"
  ["MinIO"]="http://localhost:9001"
  ["FastAPI"]="http://localhost:8000/health"
  ["DataAPI"]="http://localhost:8002/docs"
  ["Streamlit"]="http://localhost:8501"
  ["Locust"]="http://localhost:8089"
  ["Prometheus"]="http://localhost:9090"
  ["Grafana"]="http://localhost:3000"
  ["ArgoCD"]="https://localhost:8081"
)

for name in Airflow MLflow MinIO FastAPI DataAPI Streamlit Locust Prometheus Grafana ArgoCD; do
  url="${URLS[$name]}"
  code=$(curl -sk -o /dev/null -w "%{http_code}" --max-time 4 "$url" || echo "000")
  printf "  %-12s %-30s -> HTTP %s\n" "$name" "$url" "$code"
done

echo ""
echo "Listo. Para detener todos los túneles ejecuta: ./start-ports.sh stop"
echo "Logs disponibles en: /tmp/pf-*.log"
