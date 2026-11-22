import os
import json
import hashlib
from datetime import datetime, timedelta
import joblib
import requests
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
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

DB_URL = os.environ["DATABASE_URL"]
DATA_API_URL = os.getenv("DATA_API_URL", "http://data-api-svc:80")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-svc:8000")
RELOAD_TOKEN = os.environ["ADMIN_RELOAD_TOKEN"]
MODEL_NAME = "real_estate_pricer"
GROUP_NUMBER = int(os.getenv("DATA_API_GROUP", "1"))

EXPECTED_COLS = {
    "brokered_by", "status", "price", "bed", "bath",
    "acre_lot", "street", "city", "state", "zip_code",
    "house_size", "prev_sold_date"
}

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(seconds=15),
}


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def safe_float(val):
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return None if np.isnan(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return None


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _fetch_batch_from_api(**context):
    """Tarea 2: Consume el siguiente lote desde GET /data?group_number=1."""
    url = f"{DATA_API_URL}/data"
    resp = requests.get(url, params={"group_number": GROUP_NUMBER}, timeout=30)

    if resp.status_code in (204, 404):
        raise ValueError("La API de datos indicó que no hay más lotes disponibles para este grupo.")
    resp.raise_for_status()

    payload = resp.json()
    records = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("data", "records", "items", "results", "batch"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break

    if not records or not isinstance(records, list):
        raise ValueError(f"Respuesta vacía o inesperada de la API de datos: {type(payload)}")

    raw_bytes = json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    payload_hash = hashlib.sha256(raw_bytes).hexdigest()

    tmp_path = f"/tmp/batch_{payload_hash[:12]}.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, default=str)

    context["ti"].xcom_push(key="tmp_path", value=tmp_path)
    context["ti"].xcom_push(key="payload_hash", value=payload_hash)
    context["ti"].xcom_push(key="row_count", value=len(records))


def _store_raw_batch(**context):
    """Tarea 3: Persiste el lote original en RAW DATA sin reprocesar duplicados."""
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
            duplicate_batch = True
        else:
            duplicate_batch = False
            cols = sorted(list(records[0].keys())) if records else []
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
                    "price": safe_float(r.get("price")),
                    "bed": safe_float(r.get("bed")),
                    "bath": safe_float(r.get("bath")),
                    "acre_lot": safe_float(r.get("acre_lot")),
                    "street": str(r.get("street")) if r.get("street") is not None else None,
                    "city": str(r.get("city")) if r.get("city") is not None else None,
                    "state": str(r.get("state")) if r.get("state") is not None else None,
                    "zip_code": str(r.get("zip_code")) if r.get("zip_code") is not None else None,
                    "house_size": safe_float(r.get("house_size")),
                    "prev_sold_date": str(r.get("prev_sold_date")) if r.get("prev_sold_date") is not None else None,
                    "raw_payload": json.dumps(r, default=str),
                })

            pd.DataFrame(rows_to_insert).to_sql(
                "properties_raw", conn, schema="raw_data", if_exists="append", index=False, chunksize=1000
            )

    ti.xcom_push(key="batch_id", value=int(batch_id))
    ti.xcom_push(key="duplicate_batch", value=duplicate_batch)


def _validate_schema(**context):
    """Tarea 4: Valida columnas críticas del lote recibido."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()
    with engine.connect() as conn:
        raw_cols = conn.execute(
            text("SELECT schema_columns FROM raw_data.batch_metadata WHERE batch_id = :bid"),
            {"bid": batch_id}
        ).scalar()

    received = set(raw_cols) if isinstance(raw_cols, list) else set(json.loads(raw_cols or "[]"))
    critical_missing = {"price", "house_size", "city", "state"} - received
    schema_ok = len(critical_missing) == 0

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE raw_data.batch_metadata SET validation_passed = :ok WHERE batch_id = :bid"),
            {"ok": schema_ok, "bid": batch_id}
        )
    ti.xcom_push(key="schema_ok", value=schema_ok)


def _validate_data_quality(**context):
    """Tarea 5: Evalúa nulos, rangos válidos y consistencia del target."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM raw_data.properties_raw WHERE batch_id = {batch_id}", engine)
    valid = df[df["price"].notna() & (df["price"] > 5000) & (df["price"] < 50_000_000)]
    ti.xcom_push(key="valid_rows_count", value=int(len(valid)))
    ti.xcom_push(key="valid_ratio", value=float(len(valid) / max(len(df), 1)))


def _detect_new_categories(**context):
    """Tarea 6: Detecta nuevas ciudades frente al histórico en CLEAN DATA."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()

    batch_df = pd.read_sql(
        f"SELECT city, state, status FROM raw_data.properties_raw WHERE batch_id = {batch_id}",
        engine
    )
    hist_df = pd.read_sql(
        f"SELECT DISTINCT city, state FROM clean_data.properties_clean WHERE raw_batch_id < {batch_id}",
        engine
    )

    if hist_df.empty:
        new_cat_ratio = 1.0
        new_cities = list(batch_df["city"].dropna().unique()[:15])
    else:
        known_cities = set(hist_df["city"].dropna().unique())
        is_new = ~batch_df["city"].fillna("UNKNOWN").isin(known_cities)
        new_cat_ratio = float(is_new.mean())
        new_cities = list(batch_df.loc[is_new, "city"].dropna().unique()[:15])

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE raw_data.batch_metadata SET new_categories = :nc WHERE batch_id = :bid"),
            {"nc": json.dumps({"new_cities_sample": new_cities, "ratio": round(new_cat_ratio, 4)}), "bid": batch_id}
        )
    ti.xcom_push(key="new_cat_ratio", value=new_cat_ratio)


def _detect_data_drift(**context):
    """Tarea 7: Evalúa drift mediante prueba Kolmogorov-Smirnov (ks_2samp)."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()

    batch_df = pd.read_sql(
        f"SELECT house_size, price, acre_lot FROM raw_data.properties_raw WHERE batch_id = {batch_id} AND price IS NOT NULL",
        engine
    )
    hist_df = pd.read_sql(
        f"SELECT house_size, price, acre_lot FROM clean_data.properties_clean WHERE raw_batch_id < {batch_id}",
        engine
    )

    if len(hist_df) < 50 or len(batch_df) < 20:
        drift_detected = False
        ks_stat = 0.0
    else:
        stats_list, pvals_list = [], []
        for col in ("house_size", "price", "acre_lot"):
            s_new, s_hist = batch_df[col].dropna(), hist_df[col].dropna()
            if len(s_new) >= 15 and len(s_hist) >= 30:
                res = ks_2samp(s_new, s_hist)
                stats_list.append(float(res.statistic))
                pvals_list.append(float(res.pvalue))
        ks_stat = max(stats_list) if stats_list else 0.0
        min_p = min(pvals_list) if pvals_list else 1.0
        drift_detected = bool(min_p < 0.05 and ks_stat >= 0.08)

    ti.xcom_push(key="drift_detected", value=drift_detected)
    ti.xcom_push(key="ks_stat", value=ks_stat)


def _preprocess_data(**context):
    """Tarea 8: Limpia el lote y persiste los datos trazables en CLEAN DATA."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()

    df = pd.read_sql(f"SELECT * FROM raw_data.properties_raw WHERE batch_id = {batch_id}", engine)
    df = df.dropna(subset=["price"]).copy()
    df = df[(df["price"] > 5000) & (df["price"] < 50_000_000)]
    df = df.drop_duplicates(subset=["brokered_by", "status", "price", "bed", "bath", "city", "state", "zip_code", "house_size"])

    for col, fallback in [("bed", 3.0), ("bath", 2.0), ("acre_lot", 0.2), ("house_size", 1600.0)]:
        med = df[col].median()
        df[col] = df[col].fillna(med if pd.notna(med) else fallback).clip(lower=0)

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
        if not clean_df.empty:
            clean_df.to_sql("properties_clean", conn, schema="clean_data", if_exists="append", index=False, chunksize=1000)
        total_clean = conn.execute(text("SELECT COUNT(*) FROM clean_data.properties_clean")).scalar()

    ti.xcom_push(key="clean_batch_rows", value=int(len(clean_df)))
    ti.xcom_push(key="total_clean_records", value=int(total_clean))


def _decide_training(**context):
    """Tarea 9: Decide técnicamente si se entrena o se omite el entrenamiento."""
    ti = context["ti"]
    duplicate_batch = ti.xcom_pull(key="duplicate_batch", task_ids="store_raw_batch")
    schema_ok = ti.xcom_pull(key="schema_ok", task_ids="validate_schema")
    clean_rows = ti.xcom_pull(key="clean_batch_rows", task_ids="preprocess_data")
    total_clean = ti.xcom_pull(key="total_clean_records", task_ids="preprocess_data")
    drift_detected = ti.xcom_pull(key="drift_detected", task_ids="detect_data_drift")
    ks_stat = ti.xcom_pull(key="ks_stat", task_ids="detect_data_drift")
    new_cat_ratio = ti.xcom_pull(key="new_cat_ratio", task_ids="detect_new_categories")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
        has_prod_model = True
    except Exception:
        has_prod_model = False

    if duplicate_batch and has_prod_model:
        ti.xcom_push(key="trained", value=False)
        ti.xcom_push(key="training_reason", value="No se entrena: lote duplicado (hash ya procesado).")
        return "skip_training"

    if not schema_ok or clean_rows < 30:
        ti.xcom_push(key="trained", value=False)
        ti.xcom_push(key="training_reason", value=f"No se entrena: lote inválido o pequeño ({clean_rows} filas válidas).")
        return "skip_training"

    if not has_prod_model:
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value="Se entrena: Línea base inicial (no existe modelo @champion).")
        return "train_candidate_model"

    if drift_detected:
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=f"Se entrena: Data drift detectado (KS={ks_stat:.3f} >= 0.08).")
        return "train_candidate_model"

    if new_cat_ratio >= 0.10:
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=f"Se entrena: Nuevas categorías geográficas ({new_cat_ratio*100:.1f}% >= 10%).")
        return "train_candidate_model"

    prev_total = max(total_clean - clean_rows, 1)
    if (clean_rows / prev_total) >= 0.25:
        ti.xcom_push(key="trained", value=True)
        ti.xcom_push(key="training_reason", value=f"Se entrena: Incremento de volumen >= 25% ({clean_rows} nuevos registros).")
        return "train_candidate_model"

    ti.xcom_push(key="trained", value=False)
    ti.xcom_push(
        key="training_reason",
        value=f"No se entrena: sin drift relevante (KS={ks_stat:.3f}), nuevas categorías={new_cat_ratio*100:.1f}% y volumen < 25%."
    )
    return "skip_training"


def _skip_training(**context):
    """Tarea 10: Registra omisión de entrenamiento."""
    context["ti"].xcom_push(
        key="promotion_reason",
        value="Sin entrenamiento: se conserva el modelo productivo vigente."
    )


def _train_candidate_model(**context):
    """Tarea 11: Entrena Pipeline Scikit-Learn resistente a categorías desconocidas."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM clean_data.properties_clean", engine)

    feature_cols = [
        "brokered_by", "status", "bed", "bath", "acre_lot",
        "city", "state", "zip_code", "house_size",
        "years_since_sold", "has_prev_sold"
    ]
    X, y = df[feature_cols], df["price"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42)

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
            ("high_cat", TargetEncoder(target_type="continuous", smooth="auto"), high_card_cols),
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
    splits_path = f"/tmp/splits_b{batch_id}.joblib"
    joblib.dump(pipeline, model_path)
    joblib.dump({
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }, splits_path)

    ti.xcom_push(key="model_path", value=model_path)
    ti.xcom_push(key="splits_path", value=splits_path)


def _evaluate_candidate_model(**context):
    """Tarea 12: Calcula métricas train/val/test y genera gráfico de desempeño."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    pipeline = joblib.load(ti.xcom_pull(key="model_path", task_ids="train_candidate_model"))
    splits = joblib.load(ti.xcom_pull(key="splits_path", task_ids="train_candidate_model"))

    metrics = {}
    for split_name in ("train", "val", "test"):
        X_s, y_s = splits[f"X_{split_name}"], splits[f"y_{split_name}"]
        preds = pipeline.predict(X_s)
        metrics[f"{split_name}_mae"] = float(mean_absolute_error(y_s, preds))
        metrics[f"{split_name}_rmse"] = float(root_mean_squared_error(y_s, preds))
        metrics[f"{split_name}_mape"] = safe_mape(y_s, preds)
        metrics[f"{split_name}_r2"] = float(r2_score(y_s, preds))
        if split_name == "test":
            test_preds, y_test = preds, y_s

    fig_path = f"/tmp/residuals_b{batch_id}.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, test_preds, alpha=0.35, s=12)
    lim_max = float(np.percentile(y_test, 98))
    plt.plot([0, lim_max], [0, lim_max], "r--", linewidth=1.2)
    plt.xlim(0, lim_max)
    plt.ylim(0, lim_max)
    plt.xlabel("Precio Real (USD)")
    plt.ylabel("Precio Predicho (USD)")
    plt.title(f"Batch {batch_id} - Test Real vs Predicho (MAE=${metrics['test_mae']:,.0f})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()

    ti.xcom_push(key="metrics_dict", value=metrics)
    ti.xcom_push(key="candidate_mae", value=metrics["test_mae"])
    ti.xcom_push(key="candidate_rmse", value=metrics["test_rmse"])
    ti.xcom_push(key="candidate_r2", value=metrics["test_r2"])
    ti.xcom_push(key="fig_path", value=fig_path)


def _register_candidate_in_mlflow(**context):
    """Tarea 13: Registra experimento, parámetros, métricas, gráfico y modelo en MLflow."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    pipeline = joblib.load(ti.xcom_pull(key="model_path", task_ids="train_candidate_model"))
    fig_path = ti.xcom_pull(key="fig_path", task_ids="evaluate_candidate_model")
    metrics = ti.xcom_pull(key="metrics_dict", task_ids="evaluate_candidate_model")
    reason = ti.xcom_pull(key="training_reason", task_ids="decide_training")
    total_clean = ti.xcom_pull(key="total_clean_records", task_ids="preprocess_data")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("real_estate_continuous_training")
    client = MlflowClient()

    with mlflow.start_run(run_name=f"batch_{batch_id}_training") as run:
        mlflow.log_params({
            "batch_id": batch_id,
            "total_clean_records": total_clean,
            "model_type": "HistGradientBoostingRegressor",
            "max_iter": 200,
            "learning_rate": 0.08,
            "max_depth": 8,
            "git_commit": os.getenv("GIT_COMMIT_SHA", "main"),
        })
        mlflow.set_tag("training_reason", reason)
        mlflow.set_tag("batch_id", str(batch_id))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(fig_path, artifact_path="plots")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"run_id='{run_id}'")
    candidate_version = str(versions[0].version) if versions else "1"

    ti.xcom_push(key="mlflow_run_id", value=run_id)
    ti.xcom_push(key="candidate_version", value=candidate_version)


def _compare_with_production(**context):
    """Tarea 14: Evalúa el modelo productivo @champion actual sobre la misma partición de prueba."""
    ti = context["ti"]
    splits = joblib.load(ti.xcom_pull(key="splits_path", task_ids="train_candidate_model"))
    X_test, y_test = splits["X_test"], splits["y_test"]

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
        prod_version, prod_mae, prod_rmse = None, None, None

    ti.xcom_push(key="prod_version", value=prod_version)
    ti.xcom_push(key="prod_mae", value=prod_mae)
    ti.xcom_push(key="prod_rmse", value=prod_rmse)


def _decide_promotion(**context):
    """Tarea 15: Promueve si es línea base o si MAE mejora >= 3% y RMSE no empeora > 1%."""
    ti = context["ti"]
    cand_mae = ti.xcom_pull(key="candidate_mae", task_ids="evaluate_candidate_model")
    cand_rmse = ti.xcom_pull(key="candidate_rmse", task_ids="evaluate_candidate_model")
    prod_version = ti.xcom_pull(key="prod_version", task_ids="compare_with_production")
    prod_mae = ti.xcom_pull(key="prod_mae", task_ids="compare_with_production")
    prod_rmse = ti.xcom_pull(key="prod_rmse", task_ids="compare_with_production")

    if prod_version is None or prod_mae is None:
        ti.xcom_push(key="promotion_reason", value="Promovido: primer modelo productivo (@champion).")
        return "promote_model"

    mae_improvement = (prod_mae - cand_mae) / prod_mae
    rmse_change = (cand_rmse - prod_rmse) / prod_rmse

    if mae_improvement >= 0.03 and rmse_change <= 0.01:
        ti.xcom_push(
            key="promotion_reason",
            value=f"Promovido: MAE mejoró {mae_improvement*100:.2f}% (${cand_mae:,.0f} vs ${prod_mae:,.0f}) y cambio RMSE={rmse_change*100:+.2f}%."
        )
        return "promote_model"

    ti.xcom_push(
        key="promotion_reason",
        value=f"Rechazado: mejora MAE={mae_improvement*100:+.2f}% (${cand_mae:,.0f} vs ${prod_mae:,.0f}), cambio RMSE={rmse_change*100:+.2f}%."
    )
    return "reject_model"


def _promote_model(**context):
    """Tarea 16: Actualiza el alias @champion en MLflow y solicita recarga en caliente a FastAPI."""
    ti = context["ti"]
    candidate_version = ti.xcom_pull(key="candidate_version", task_ids="register_candidate_in_mlflow")
    mlflow.set_tracking_uri(MLFLOW_URI)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", candidate_version)
    ti.xcom_push(key="was_promoted", value=True)
    try:
        requests.post(f"{FASTAPI_URL}/reload-model", headers={"X-Admin-Token": RELOAD_TOKEN}, timeout=10)
    except requests.RequestException:
        pass


def _explain_champion_model(**context):
    """Tarea 16b: Aplica SHAP exclusivamente al modelo ganador (@champion)."""
    from shap_explainer import run_shap_for_champion
    run_shap_for_champion(sample_size=1500)


def _reject_model(**context):
    """Tarea 17: Etiqueta el candidato como challenger_rejected sin tocar @champion."""
    ti = context["ti"]
    candidate_version = ti.xcom_pull(key="candidate_version", task_ids="register_candidate_in_mlflow")
    mlflow.set_tracking_uri(MLFLOW_URI)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger_rejected", candidate_version)
    ti.xcom_push(key="was_rejected", value=True)


def _notify_or_log_result(**context):
    """Tarea 18: Registra el resultado de la ejecución en clean_data.training_audit_log."""
    ti = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id", task_ids="store_raw_batch")
    promoted_flag = bool(ti.xcom_pull(key="was_promoted", task_ids="promote_model"))
    promotion_reason = (
        ti.xcom_pull(key="promotion_reason", task_ids="decide_promotion")
        or ti.xcom_pull(key="promotion_reason", task_ids="skip_training")
        or "N/A"
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
                "records_in_batch": ti.xcom_pull(key="clean_batch_rows", task_ids="preprocess_data"),
                "total_clean_records": ti.xcom_pull(key="total_clean_records", task_ids="preprocess_data"),
                "drift_detected": ti.xcom_pull(key="drift_detected", task_ids="detect_data_drift"),
                "drift_ks_stat": ti.xcom_pull(key="ks_stat", task_ids="detect_data_drift"),
                "new_categories_ratio": ti.xcom_pull(key="new_cat_ratio", task_ids="detect_new_categories"),
                "trained": bool(ti.xcom_pull(key="trained", task_ids="decide_training")),
                "training_reason": ti.xcom_pull(key="training_reason", task_ids="decide_training"),
                "mlflow_run_id": ti.xcom_pull(key="mlflow_run_id", task_ids="register_candidate_in_mlflow"),
                "candidate_version": ti.xcom_pull(key="candidate_version", task_ids="register_candidate_in_mlflow"),
                "candidate_mae": ti.xcom_pull(key="candidate_mae", task_ids="evaluate_candidate_model"),
                "candidate_rmse": ti.xcom_pull(key="candidate_rmse", task_ids="evaluate_candidate_model"),
                "candidate_r2": ti.xcom_pull(key="candidate_r2", task_ids="evaluate_candidate_model"),
                "prod_version": ti.xcom_pull(key="prod_version", task_ids="compare_with_production"),
                "prod_mae": ti.xcom_pull(key="prod_mae", task_ids="compare_with_production"),
                "prod_rmse": ti.xcom_pull(key="prod_rmse", task_ids="compare_with_production"),
                "promoted": promoted_flag,
                "promotion_reason": promotion_reason,
            }
        )


with DAG(
    dag_id="real_estate_mlops_dag",
    default_args=default_args,
    description="Flujo MLOps Nivel 4: Ingesta incremental, validación, drift, MLflow y promoción",
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
    explain_champion_model = PythonOperator(task_id="explain_champion_model", python_callable=_explain_champion_model)
    reject_model = PythonOperator(task_id="reject_model", python_callable=_reject_model)

    notify_or_log_result = PythonOperator(
        task_id="notify_or_log_result",
        python_callable=_notify_or_log_result,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    (
        start >> fetch_batch_from_api >> store_raw_batch >> validate_schema
        >> validate_data_quality >> detect_new_categories >> detect_data_drift
        >> preprocess_data >> decide_training
    )
    decide_training >> skip_training >> notify_or_log_result >> end
    (
        decide_training >> train_candidate_model >> evaluate_candidate_model
        >> register_candidate_in_mlflow >> compare_with_production >> decide_promotion
    )
    decide_promotion >> promote_model >> explain_champion_model >> notify_or_log_result >> end
    decide_promotion >> reject_model >> notify_or_log_result >> end