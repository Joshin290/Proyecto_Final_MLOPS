#!/usr/bin/env bash
set -e

echo "Iniciando Airflow Scheduler en segundo plano..."
airflow scheduler &

echo "Iniciando Airflow Webserver en el puerto 8080..."
exec airflow webserver --port 8080