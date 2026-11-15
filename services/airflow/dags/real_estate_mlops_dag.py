import os
import json
import hashlib
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:mlops_password_2026@postgres-svc:5432/mlops_data")
DATA_API_URL = os.getenv("DATA_API_URL", "http://data-api-svc:80")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-svc:8000")
RELOAD_TOKEN = os.getenv("ADMIN_RELOAD_TOKEN", "supersecret-reload-token")
MODEL_NAME = "real_estate_pricer"
GROUP_ID = int(os.getenv("DATA_API_GROUP", "1"))

EXPECTED_COLS = {
    "brokered_by", "status", "price", "bed", "bath",
    "acre_lot", "street", "city", "state", "zip_code",
    "house_size", "prev_sold_date"
}

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(seconds=20),
}


def get_engine():
    return create_engine(DB_URL)


def _fetch_batch_from_api(**context):
    """Tarea 2: Consume un lote desde la API externa con manejo de reintentos y errores."""
    candidate_endpoints = [
        f"{DATA_API_URL}/data?group={GROUP_ID}",
        f"{DATA_API_URL}/batch?group={GROUP_ID}",
        f"{DATA_API_URL}/properties?group={GROUP_ID}",
        f"{DATA_API_URL}/data",
    ]
    records = None
    for url in candidate_endpoints:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                payload = resp.json()
                records = payload.get("data", payload) if isinstance(payload, dict) else payload
                if isinstance(records, list) and len(records) > 0:
                    break
        except requests.RequestException:
            continue

    if not records or not isinstance(records, list):
        raise ValueError("La API de datos no retornó registros válidos o se agotaron los lotes.")

    raw_bytes = json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    payload_hash = hashlib.sha256(raw_bytes).hexdigest()

    tmp_path = f"/tmp/batch_{payload_hash[:12]}.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f)

    context["ti"].xcom_push(key="tmp_path", value=tmp_path)
    context["ti"].xcom_push(key="payload_hash", value=payload_hash)
    context["ti"].xcom_push(key="row_count", value=len(records))


def _store_raw_batch(**context):
    """Tarea 3: Almacena la respuesta original en RAW DATA registrando metadatos del lote."""
    ti = context["ti"]
    tmp_path = ti.xcom_pull(key="tmp_path", task_ids="fetch_batch_from_api")
    payload_hash = ti.xcom_pull(key="payload_hash", task_ids="fetch_batch_from_api")

    with open(tmp_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    engine = get_engine()
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT batch_id FROM raw_data.batch_metadata WHERE payload_hash = :h"),
            {"h": payload_hash}
        ).fetchone()
        if existing:
            batch_id = existing[0]
        else:
            cols = list(records[0].keys()) if records else []
            res = conn.execute(
                text("""
                    INSERT INTO raw_data.batch_metadata (row_count, payload_hash, schema_columns, status)
                    VALUES (:rc, :ph, :sc, 'RAW_STORED') RETURNING batch_id
                """),
                {"rc": len(records), "ph": payload_hash, "sc": json.dumps(cols)}
            )
            batch_id = res.fetchone()[0]

            rows_to_insert = []
            for r in records:
                rows_to_insert.append({
                    "batch_id": batch_id,
                    "brokered_by": str(r.get("brokered_by")) if r.get("brokered_by") is not None else None,
                    "status": str(r.get("status")) if r.get("status") is not None else None,
                    "price": float(r["price"]) if r.get("price") not in (None, "") else None,
                    "bed": float(r["bed"]) if r.get("bed") not in (None, "") else None,
                    "bath": float(r["bath"]) if r.get("bath") not in (None, "") else None,
                    "acre_lot": float(r["acre_lot"]) if r.get("acre_lot") not in (None, "") else None,
                    "street": str(r.get("street")) if r.get("street") is not None else None,
                    "city": str(r.get("city")) if r.get("city") is not None else None,
                    "state": str(r.get("state")) if r.get("state") is not None else None,
                    "zip_code": str(r.get("zip_code")) if r.get("zip_code") is not None else None,
                    "house_size": float(r["house_size"]) if r.get("house_size") not in (None, "") else None,
                    "prev_sold_date": str(r.get("prev_sold_date")) if r.get("prev_sold_date") is not None else None,
                    "raw_payload": json.dumps(r, default=str),
                })
            pd.DataFrame(rows_to_insert).to_sql(
                "properties_raw", conn, schema="raw_data", if_exists="append", index=False
            )

    ti.xcom_push(key="batch_id", value=batch_id)


def _validate_schema(**context):
    """Tarea 4: Valida columnas, tipos y estructura del lote."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM raw_data.properties_raw WHERE batch_id = {batch_id} LIMIT 5", engine)
    missing = EXPECTED_COLS - set(df.columns)
    schema_ok = len(missing) == 0
    ti.xcom_push(key="schema_ok", value=schema_ok)


def _validate_data_quality(**context):
    """Tarea 5: Evalúa nulos, duplicados, rangos y consistencia del target."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM raw_data.properties_raw WHERE batch_id = {batch_id}", engine)
    valid_rows = df[df["price"].notna() & (df["price"] > 1000) & (df["price"] < 1e8)]
    valid_ratio = len(valid_rows) / max(len(df), 1)
    ti.xcom_push(key="valid_rows_count", value=int(len(valid_rows)))
    ti.xcom_push(key="valid_ratio", value=float(valid_ratio))


def _detect_new_categories(**context):
    """Tarea 6: Identifica categorías nuevas en variables geográficas."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()

    batch_df = pd.read_sql(f"SELECT city, state, status FROM raw_data.properties_raw WHERE batch_id = {batch_id}", engine)
    hist_df = pd.read_sql("SELECT DISTINCT city, state FROM clean_data.properties_clean", engine)

    if hist_df.empty:
        new_cat_ratio = 1.0
        new_cities = list(batch_df["city"].dropna().unique()[:10])
    else:
        known_cities = set(hist_df["city"].dropna().unique())
        is_new_city = ~batch_df["city"].fillna("UNKNOWN").isin(known_cities)
        new_cat_ratio = float(is_new_city.mean())
        new_cities = list(batch_df.loc[is_new_city, "city"].dropna().unique()[:10])

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE raw_data.batch_metadata SET new_categories = :nc WHERE batch_id = :bid"),
            {"nc": json.dumps({"new_cities_sample": new_cities, "ratio": new_cat_ratio}), "bid": batch_id}
        )
    ti.xcom_push(key="new_cat_ratio", value=new_cat_ratio)


def _detect_data_drift(**context):
    """Tarea 7: Compara la distribución del lote con la referencia histórica usando KS-test."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()

    batch_df = pd.read_sql(
        f"SELECT house_size, price FROM raw_data.properties_raw WHERE batch_id = {batch_id} AND price IS NOT NULL",
        engine
    )
    hist_df = pd.read_sql(
        f"SELECT house_size, price FROM clean_data.properties_clean WHERE raw_batch_id < {batch_id}",
        engine
    )

    if len(hist_df) < 50 or len(batch_df) < 20:
        drift_detected = False
        ks_stat = 0.0
    else:
        ks_hs = ks_2samp(batch_df["house_size"].dropna(), hist_df["house_size"].dropna())
        ks_pr = ks_2samp(batch_df["price"].dropna(), hist_df["price"].dropna())
        ks_stat = float(max(ks_hs.statistic, ks_pr.statistic))
        p_val = float(min(ks_hs.pvalue, ks_pr.pvalue))
        drift_detected = bool(p_val < 0.05 and ks_stat >= 0.08)

    ti.xcom_push(key="drift_detected", value=drift_detected)
    ti.xcom_push(key="ks_stat", value=ks_stat)


def _preprocess_data(**context):
    """Tarea 8: Transforma el lote y actualiza CLEAN DATA manteniendo trazabilidad."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()

    df = pd.read_sql(f"SELECT * FROM raw_data.properties_raw WHERE batch_id = {batch_id}", engine)
    df = df.dropna(subset=["price"]).copy()
    df = df[(df["price"] > 5000) & (df["price"] < 50_000_000)]

    for col, default_val in [("bed", 3.0), ("bath", 2.0), ("acre_lot", 0.2), ("house_size", 1600.0)]:
        med = df[col].median()
        df[col] = df[col].fillna(med if pd.notna(med) else default_val).clip(lower=0)

    for cat_col in ["brokered_by", "status", "city", "state", "zip_code"]:
        df[cat_col] = df[cat_col].fillna("UNKNOWN").astype(str)

    parsed_dates = pd.to_datetime(df["prev_sold_date"], errors="coerce")
    df["has_prev_sold"] = parsed_dates.notna().astype(int)
    df["years_since_sold"] = ((pd.Timestamp.now() - parsed_dates).dt.days / 365.25).fillna(-1.0)

    clean_df = pd.DataFrame({
        "raw_batch_id": batch_id,
        "brokered_by": df["brokered_by"],
        "status": df["status"],
        "price": df["price"],
        "bed": df["bed"],
        "bath": df["bath"],
        "acre_lot": df["acre_lot"],
        "city": df["city"],
        "state": df["state"],
        "zip_code": df["zip_code"],
        "house_size": df["house_size"],
        "years_since_sold": df["years_since_sold"],
        "has_prev_sold": df["has_prev_sold"],
    })

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM clean_data.properties_clean WHERE raw_batch_id = :bid"), {"bid": batch_id})
        clean_df.to_sql("properties_clean", conn, schema="clean_data", if_exists="append", index=False)
        total_clean = conn.execute(text("SELECT COUNT(*) FROM clean_data.properties_clean")).scalar()

    ti.xcom_push(key="clean_batch_rows", value=int(len(clean_df)))
    ti.xcom_push(key="total_clean_records", value=int(total_clean))


def _decide_training(**context):
    """Tarea 9: Bifurca entre entrenar o no entrenar según criterios técnicos explícitos."""
    ti = context["ti"]
    schema_ok = ti.xcom_pull(key="schema_ok", task_ids="validate_schema")
    clean_rows = ti.xcom_pull(key="clean_batch_rows", task_ids="preprocess_data")
    total_clean = ti.xcom_pull(key="total_clean_records", task_ids="preprocess_data")
    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="detect_data_drift")
    ks_stat = ti.xcom_pull(key="ks_stat", task_ids="detect_data_drift")
    new_cat_ratio = ti.xcom_pull(key="new_cat_ratio", task_ids="detect_new_categories")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    has_prod_model = False
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
        has_prod_model = True
    except Exception:
        has_prod_model = False

    if not schema_ok or clean_rows < 30:
        reason = f"No se entrena: lote pequeño o inválido ({clean_rows} registros válidos)."
        ti.xcom_push(key="trained", value=False)
        ti.xcom_push(key="training_reason", value=reason)
        return "skip_training"

    if not has_prod_model:
        reason = "Se entrena: Línea base inicial (no existe modelo productivo @champion en MLflow)."
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=reason)
        return "train_candidate_model"

    if drift_detected:
        reason = f"Se entrena: Cambio significativo de distribución detectado (KS={ks_stat:.3f})."
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=reason)
        return "train_candidate_model"

    if new_cat_ratio >= 0.12:
        reason = f"Se entrena: Aparición de nuevas categorías geográficas ({new_cat_ratio*100:.1f}% del lote)."
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=reason)
        return "train_candidate_model"

    if clean_rows / max(total_clean - clean_rows, 1) >= 0.25:
        reason = f"Se entrena: El nuevo lote incrementa el volumen acumulado en >= 25% ({clean_rows} registros)."
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=reason)
        return "train_candidate_model"

    reason = f"No se entrena: Sin cambio de distribución relevante (KS={ks_stat:.3f}) ni nuevas categorías significativas."
    ti.xcom_push(key="trained", value=False)
    ti.xcom_push(key="training_reason", value=reason)
    return "skip_training"


def _skip_training(**context):
    """Tarea 10: Registra la omisión justificada del entrenamiento."""
    ti = context["ti"]
    ti.xcom_push(key="promoted", value=False)
    ti.xcom_push(key="promotion_reason", value="No se entrenó modelo candidato en este lote.")


def _train_candidate_model(**context):
    """Tarea 11: Entrena un modelo candidato usando todos los datos limpios acumulados."""
    import joblib
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM clean_data.properties_clean", engine)

    feature_cols = [
        "brokered_by", "status", "bed", "bath", "acre_lot",
        "city", "state", "zip_code", "house_size",
        "years_since_sold", "has_prev_sold"
    ]
    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = ["bed", "bath", "acre_lot", "house_size", "years_since_sold", "has_prev_sold"]
    low_card_cols = ["status", "state"]
    high_card_cols = ["brokered_by", "city", "zip_code"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("low_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cols),
            ("high_cat", TargetEncoder(smooth="auto"), high_card_cols),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_depth=8, random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    model_path = f"/tmp/candidate_model_b{batch_id}.joblib"
    test_path = f"/tmp/test_data_b{batch_id}.parquet"
    joblib.dump(pipeline, model_path)
    X_test.assign(price=y_test).to_parquet(test_path)

    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="test_path", value=test_path)


def _evaluate_candidate_model(**context):
    """Tarea 12: Calcula métricas (MAE, RMSE, R2) y genera gráfico de desempeño."""
    import joblib
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    model_path = ti.xcom_pull(key="model_path", task_ids="train_candidate_model")
    test_path = ti.xcom_pull(key="test_path", task_ids="train_candidate_model")

    pipeline = joblib.load(model_path)
    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    preds = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(root_mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    fig_path = f"/tmp/residuals_b{batch_id}.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, preds, alpha=0.3)
    plt.xlabel("Precio Real (USD)")
    plt.ylabel("Precio Predicho (USD)")
    plt.title(f"Batch {batch_id} - Predicción vs Real (MAE={mae:,.0f})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    ti.xcom_push(key="candidate_mae", value=mae)
    ti.xcom_push(key="candidate_rmse", value=rmse)
    ti.xcom_push(key="candidate_r2", value=r2)
    ti.xcom_push(key="fig_path", value=fig_path)


def _register_candidate_in_mlflow(**context):
    """Tarea 13: Registra modelo candidato, métricas, parámetros y artefactos en MLflow."""
    import joblib
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    model_path = ti.xcom_pull(key="model_path", task_ids="train_candidate_model")
    fig_path = ti.xcom_pull(key="fig_path", task_ids="evaluate_candidate_model")
    mae = ti.xcom_pull(key="candidate_mae", task_ids="evaluate_candidate_model")
    rmse = ti.xcom_pull(key="candidate_rmse", task_ids="evaluate_candidate_model")
    r2 = ti.xcom_pull(key="candidate_r2", task_ids="evaluate_candidate_model")
    reason = ti.xcom_pull(key="training_reason", task_ids="decide_training")
    total_clean = ti.xcom_pull(key="total_clean_records", task_ids="preprocess_data")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("real_estate_continuous_training")

    pipeline = joblib.load(model_path)

    with mlflow.start_run(run_name=f"batch_{batch_id}_training") as run:
        mlflow.log_params({
            "batch_id": batch_id,
            "total_clean_records": total_clean,
            "model_type": "HistGradientBoostingRegressor",
            "max_iter": 200,
            "learning_rate": 0.08,
            "git_commit": os.getenv("GIT_COMMIT_SHA", "main"),
        })
        mlflow.set_tag("training_reason", reason)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.log_artifact(fig_path, artifact_path="plots")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id
        version = model_info.registered_model_version

    ti.xcom_push(key="mlflow_run_id", value=run_id)
    ti.xcom_push(key="candidate_version", value=str(version))


def _compare_with_production(**context):
    """Tarea 14: Compara el modelo candidato contra el modelo productivo @champion actual."""
    ti = context["ti"]
    test_path = ti.xcom_pull(key="test_path", task_ids="train_candidate_model")
    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    try:
        prod_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        prod_version = str(prod_mv.version)
        prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
        prod_preds = prod_model.predict(X_test)
        prod_mae = float(mean_absolute_error(y_test, prod_preds))
        prod_rmse = float(root_mean_squared_error(y_test, prod_preds))
    except Exception:
        prod_version = None
        prod_mae = None
        prod_rmse = None

    ti.xcom_push(key="prod_version", value=prod_version)
    ti.xcom_push(key="prod_mae", value=prod_mae)
    ti.xcom_push(key="prod_rmse", value=prod_rmse)


def _decide_promotion(**context):
    """Tarea 15: Bifurca entre promover o rechazar el modelo según mejora de MAE >= 3%."""
    ti = context["ti"]
    cand_mae = ti.xcom_pull(key="candidate_mae", task_ids="evaluate_candidate_model")
    cand_rmse = ti.xcom_pull(key="candidate_rmse", task_ids="evaluate_candidate_model")
    prod_version = ti.xcom_pull(key="prod_version", task_ids="compare_with_production")
    prod_mae = ti.xcom_pull(key="prod_mae", task_ids="compare_with_production")
    prod_rmse = ti.xcom_pull(key="prod_rmse", task_ids="compare_with_production")

    if prod_version is None or prod_mae is None:
        ti.xcom_push(key="promotion_reason", value="Se promovió porque no existía modelo productivo previo.")
        return "promote_model"

    mae_improvement = (prod_mae - cand_mae) / prod_mae
    rmse_change = (cand_rmse - prod_rmse) / prod_rmse

    if mae_improvement >= 0.03 and rmse_change <= 0.01:
        reason = (
            f"Se promovió porque redujo el MAE en {mae_improvement*100:.2f}% "
            f"({cand_mae:,.0f} vs {prod_mae:,.0f}) y mantuvo controlado el RMSE ({rmse_change*100:.2f}%)."
        )
        ti.xcom_push(key="promotion_reason", value=reason)
        return "promote_model"

    reason = (
        f"No se promovió porque la variación de MAE ({mae_improvement*100:.2f}%) "
        f"no superó el umbral de mejora del 3% frente al modelo productivo."
    )
    ti.xcom_push(key="promotion_reason", value=reason)
    return "reject_model"


def _promote_model(**context):
    """Tarea 16: Actualiza el alias @champion en MLflow y recarga FastAPI sin redespliegue."""
    ti = context["ti"]
    candidate_version = ti.xcom_pull(key="candidate_version", task_ids="register_candidate_in_mlflow")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", candidate_version)
    ti.xcom_push(key="promoted", value=True)

    try:
        requests.post(
            f"{FASTAPI_URL}/reload-model",
            headers={"X-Admin-Token": RELOAD_TOKEN},
            timeout=10
        )
    except requests.RequestException:
        pass


def _reject_model(**context):
    """Tarea 17: Registra el rechazo manteniendo intacto el modelo productivo actual."""
    ti = context["ti"]
    candidate_version = ti.xcom_pull(key="candidate_version", task_ids="register_candidate_in_mlflow")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "challenger_rejected", candidate_version)
    ti.xcom_push(key="promoted", value=False)


def _notify_or_log_result(**context):
    """Tarea 18: Actualiza la tabla de historial consultada por Streamlit."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    clean_rows = ti.xcom_pull(key="clean_batch_rows", task_ids="preprocess_data")
    total_clean = ti.xcom_pull(key="total_clean_records", task_ids="preprocess_data")
    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="detect_data_drift")
    ks_stat = ti.xcom_pull(key="ks_stat", task_ids="detect_data_drift")
    new_cat_ratio = ti.xcom_pull(key="new_cat_ratio", task_ids="detect_new_categories")
    trained = ti.xcom_pull(key="trained", task_ids="decide_training")
    training_reason = ti.xcom_pull(key="training_reason", task_ids="decide_training")

    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="register_candidate_in_mlflow")
    cand_ver = ti.xcom_pull(key="candidate_version", task_ids="register_candidate_in_mlflow")
    cand_mae = ti.xcom_pull(key="candidate_mae", task_ids="evaluate_candidate_model")
    cand_rmse = ti.xcom_pull(key="candidate_rmse", task_ids="evaluate_candidate_model")
    cand_r2 = ti.xcom_pull(key="candidate_r2", task_ids="evaluate_candidate_model")
    prod_ver = ti.xcom_pull(key="prod_version", task_ids="compare_with_production")
    prod_mae = ti.xcom_pull(key="prod_mae", task_ids="compare_with_production")
    prod_rmse = ti.xcom_pull(key="prod_rmse", task_ids="compare_with_production")

    promoted = ti.xcom_pull(key="promoted", task_ids=["promote_model", "reject_model", "skip_training"])
    promoted_flag = bool(promoted[0]) if isinstance(promoted, list) and promoted else False
    promotion_reason = (
        ti.xcom_pull(key="promotion_reason", task_ids="decide_promotion")
        or ti.xcom_pull(key="promotion_reason", task_ids="skip_training")
    )

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM clean_data.training_audit_log WHERE batch_id = :bid"), {"bid": batch_id})
        conn.execute(
            text("""
                INSERT INTO clean_data.training_audit_log (
                    batch_id, records_in_batch, total_clean_records,
                    drift_detected, drift_ks_stat, new_categories_ratio,
                    trained, training_reason, mlflow_run_id,
                    candidate_version, candidate_mae, candidate_rmse, candidate_r2,
                    prod_version, prod_mae, prod_rmse, promoted, promotion_reason
                ) VALUES (
                    :batch_id, :records_in_batch, :total_clean_records,
                    :drift_detected, :drift_ks_stat, :new_categories_ratio,
                    :trained, :training_reason, :mlflow_run_id,
                    :candidate_version, :candidate_mae, :candidate_rmse, :candidate_r2,
                    :prod_version, :prod_mae, :prod_rmse, :promoted, :promotion_reason
                )
            """),
            {
                "batch_id": batch_id,
                "records_in_batch": clean_rows,
                "total_clean_records": total_clean,
                "drift_detected": drift_detected,
                "drift_ks_stat": ks_stat,
                "new_categories_ratio": new_cat_ratio,
                "trained": trained,
                "training_reason": training_reason,
                "mlflow_run_id": run_id,
                "candidate_version": cand_ver,
                "candidate_mae": cand_mae,
                "candidate_rmse": cand_rmse,
                "candidate_r2": cand_r2,
                "prod_version": prod_ver,
                "prod_mae": prod_mae,
                "prod_rmse": prod_rmse,
                "promoted": promoted_flag,
                "promotion_reason": promotion_reason,
            }
        )


with DAG(
    dag_id="real_estate_mlops_dag",
    default_args=default_args,
    description="Flujo MLOps Nivel 4 con 19 tareas y bifurcaciones técnicas",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")
    fetch_batch_from_api = PythonOperator(task_id="fetch_batch_from_api", python_callable=_fetch_batch_from_api)
    store_raw_batch = PythonOperator(task_id="store_raw_batch", python_callable=_store_raw_batch)
    validate_schema = PythonOperator(task_id="validate_schema", python_callable=_validate_schema)
    validate_data_quality = PythonOperator(task_id="validate_data_quality", python_callable=_validate_data_quality)
    detect_new_categories = PythonOperator(task_id="detect_new_categories", python_callable=_detect_new_categories)
    detect_data_drift = PythonOperator(task_id="detect_data_drift", python_callable=_detect_data_drift)
    preprocess_data = PythonOperator(task_id="preprocess_data", python_callable=_preprocess_data)

    decide_training = BranchPythonOperator(task_id="decide_training", python_callable=_decide_training)
    skip_training = PythonOperator(task_id="skip_training", python_callable=_skip_training)

    train_candidate_model = PythonOperator(task_id="train_candidate_model", python_callable=_train_candidate_model)
    evaluate_candidate_model = PythonOperator(task_id="evaluate_candidate_model", python_callable=_evaluate_candidate_model)
    register_candidate_in_mlflow = PythonOperator(task_id="register_candidate_in_mlflow", python_callable=_register_candidate_in_mlflow)
    compare_with_production = PythonOperator(task_id="compare_with_production", python_callable=_compare_with_production)

    decide_promotion = BranchPythonOperator(task_id="decide_promotion", python_callable=_decide_promotion)
    promote_model = PythonOperator(task_id="promote_model", python_callable=_promote_model)
    reject_model = PythonOperator(task_id="reject_model", python_callable=_reject_model)

    notify_or_log_result = PythonOperator(
        task_id="notify_or_log_result",
        python_callable=_notify_or_log_result,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    (
        start
        >> fetch_batch_from_api
        >> store_raw_batch
        >> validate_schema
        >> validate_data_quality
        >> detect_new_categories
        >> detect_data_drift
        >> preprocess_data
        >> decide_training
    )
    decide_training >> skip_training >> notify_or_log_result >> end
    (
        decide_training
        >> train_candidate_model
        >> evaluate_candidate_model
        >> register_candidate_in_mlflow
        >> compare_with_production
        >> decide_promotion
    )
    decide_promotion >> [promote_model, reject_model] >> notify_or_log_result >> end
