import json, boto3
import os
import requests
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-svc:8000")
DB_URL = os.environ["DATABASE_URL"]

st.set_page_config(page_title="Real Estate MLOps Platform", layout="wide")
st.title("Sistema MLOps Nivel 4: Inferencia Inmobiliaria y Auditoría de Lotes")

tab1, tab2, tab3 = st.tabs(["1. Inferencia de Propiedades", "2. Historial de Entrenamiento y Despliegue", "3. Explicabilidad del Campeón (SHAP)"])

with tab1:
    st.subheader("Estimación de Precio Consumiendo FastAPI + MLflow (@champion)")
    col1, col2, col3 = st.columns(3)
    with col1:
        status = st.selectbox("Estado de la propiedad (status)", ["for_sale", "ready_to_build"])
        bed = st.number_input("Número de habitaciones (bed)", min_value=1.0, max_value=20.0, value=3.0)
        bath = st.number_input("Número de baños (bath)", min_value=1.0, max_value=20.0, value=2.0)
        house_size = st.number_input("Área habitable en sq.ft (house_size)", min_value=200.0, value=1650.0)
    with col2:
        acre_lot = st.number_input("Tamaño del terreno en acres (acre_lot)", min_value=0.01, value=0.20)
        city = st.text_input("Ciudad (city)", value="San Juan")
        state = st.text_input("Estado (state)", value="Puerto Rico")
        zip_code = st.text_input("Código Postal (zip_code)", value="00901")
    with col3:
        brokered_by = st.text_input("Agencia / Corredor (brokered_by)", value="103378")
        street = st.text_input("Dirección codificada (street)", value="1962661")
        prev_sold_date = st.text_input("Fecha venta anterior (YYYY-MM-DD)", value="2021-05-10")

    if st.button("Calcular Predicción", type="primary"):
        payload = {
            "brokered_by": brokered_by, "status": status, "bed": bed, "bath": bath,
            "acre_lot": acre_lot, "street": street, "city": city, "state": state,
            "zip_code": zip_code, "house_size": house_size, "prev_sold_date": prev_sold_date
        }
        try:
            r = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                data = r.json()
                m1, m2 = st.columns(2)
                m1.metric("Precio Estimado (USD)", f"${data['predicted_price']:,.2f}")
                m2.metric("Versión del Modelo Productivo Usada", f"v{data['model_version']}")
            else:
                st.error(f"Error en la API ({r.status_code}): {r.text}")
        except Exception as e:
            st.error(f"Error de conexión con FastAPI: {e}")

with tab2:
    st.subheader("Historial de Decisiones por Lote Procesado")
    try:
        engine = create_engine(DB_URL)
        audit_df = pd.read_sql("SELECT * FROM clean_data.training_audit_log ORDER BY batch_id ASC", engine)
        if audit_df.empty:
            st.info("Aún no se han procesado lotes desde el DAG de Airflow.")
        else:
            for _, row in audit_df.iterrows():
                accion = "entrenó" if row["trained"] else "no entrenó"
                resultado = (
                    f"Se promovió (v{row['candidate_version']})"
                    if row["promoted"]
                    else ("No se promovió" if row["trained"] else "Mantiene modelo actual")
                )
                train_r = str(row["training_reason"] or "N/A").replace("$", r"\$")
                promo_r = str(row["promotion_reason"] or "N/A").replace("$", r"\$")
                st.markdown(
                    f"- **Batch {row['batch_id']}**: **{accion}** ({train_r}). "
                    f"**{resultado}** ({promo_r}). "
                    f"MLflow Run ID: `{row['mlflow_run_id'] or 'N/A'}`"
                )
            st.divider()
            st.dataframe(audit_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error consultando el historial en PostgreSQL: {e}")
with tab3:
    st.subheader("Explicabilidad Global y Local del Modelo Ganador (@champion)")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
    s3_ep = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio-svc:9000")
    try:
        r_mv = requests.get(f"{mlflow_uri}/api/2.0/mlflow/registered-models/alias",
                            params={"name": "real_estate_pricer", "alias": "champion"}, timeout=10)
        r_mv.raise_for_status()
        mv = r_mv.json()["model_version"]
        r_run = requests.get(f"{mlflow_uri}/api/2.0/mlflow/runs/get", params={"run_id": mv["run_id"]}, timeout=10)
        r_run.raise_for_status()
        art_uri = r_run.json()["run"]["info"]["artifact_uri"].replace("s3://", "")
        bucket, prefix = art_uri.split("/", 1)
        s3 = boto3.client("s3", endpoint_url=s3_ep)
        meta = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/shap_analysis/shap_global.json")["Body"].read())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelo Campeón", f"v{meta['model_version']}")
        c2.metric("Precio Base E[f(X)]", f"${meta['base_value_usd']:,.0f}")
        c3.metric("Muestra SHAP", f"{meta['sample_size']:,}")
        c4.metric("Variable #1", meta["global_importance"][0]["feature"])

        st.dataframe(pd.DataFrame(meta["global_importance"]), use_container_width=True)
        imgs = [
            ("01_shap_global_grouped.png", "Importancia Global Agrupada (USD)"),
            ("02_shap_beeswarm.png", "Distribución de Impacto (Beeswarm)"),
            ("05_shap_dependence_top.png", f"Dependencia SHAP ({meta['top_numeric_feature']})"),
            ("03_shap_waterfall_high.png", "Explicación Local - Propiedad Alta (P95)"),
            ("04_shap_waterfall_low.png", "Explicación Local - Propiedad Baja (P10)"),
        ]
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for col, (fname, cap) in zip(cols, imgs[i:i+2]):
                data = s3.get_object(Bucket=bucket, Key=f"{prefix}/shap_analysis/{fname}")["Body"].read()
                col.image(data, caption=cap, use_column_width=True)
    except Exception as e:
        st.info(f"Aún no hay artefactos SHAP para el modelo @champion o error al cargarlos: {e}")
