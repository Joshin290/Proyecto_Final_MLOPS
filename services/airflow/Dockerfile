FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY --chown=airflow:0 dags/ /opt/airflow/dags/

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]