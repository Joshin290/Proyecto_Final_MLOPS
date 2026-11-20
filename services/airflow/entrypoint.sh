#!/usr/bin/env bash
set -euo pipefail

airflow scheduler &
exec airflow webserver --port 8080
