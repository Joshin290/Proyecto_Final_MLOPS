import os
import json
import time
import threading
from typing import Optional
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
DB_URL = os.environ["DATABASE_URL"]
ADMIN_TOKEN = os.environ["ADMIN_RELOAD_TOKEN"]
MODEL_NAME = "real_estate_pricer"

mlflow.set_tracking_uri(MLFLOW_URI)
engine = create_engine(DB_URL, pool_pre_ping=True)

REQUEST_COUNT = Counter("inference_requests_total", "Total de peticiones de inferencia", ["status", "model_version"])
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Latencia de inferencia en segundos")
INFERENCE_ERRORS = Counter("inference_errors_total", "Total de errores en inferencia")
MODEL_VERSION_GAUGE = Gauge("loaded_model_version", "Versión numérica del modelo productivo cargado en memoria")


class PropertyInput(BaseModel):
    brokered_by: Optional[str] = Field(default="103378")
    status: str = Field(default="for_sale")
    bed: float = Field(default=3.0, ge=0)
    bath: float = Field(default=2.0, ge=0)
    acre_lot: float = Field(default=0.18, ge=0)
    street: Optional[str] = Field(default="1962661")
    city: str = Field(default="Adjuntas")
    state: str = Field(default="Puerto Rico")
    zip_code: str = Field(default="00601")
    house_size: float = Field(default=1500.0, gt=0)
    prev_sold_date: Optional[str] = Field(default="2021-06-15")


class ThreadSafeModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.version = "unloaded"

    def reload_if_needed(self, force: bool = False) -> dict:
        client = MlflowClient()
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            target_version = str(mv.version)
            if not force and self.model is not None and self.version == target_version:
                return {"status": "unchanged", "version": self.version}

            # Descarga del nuevo modelo fuera del bloqueo para no detener inferencias concurrentes
            candidate_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
            with self._lock:
                self.model = candidate_model
                self.version = target_version
                MODEL_VERSION_GAUGE.set(float(target_version))
            return {"status": "reloaded", "version": self.version}
        except Exception as exc:
            return {"status": "fallback_kept_previous", "version": self.version, "detail": str(exc)}

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            self.reload_if_needed(force=True)
        with self._lock:
            if self.model is None:
                raise RuntimeError("Aún no existe un modelo @champion registrado en MLflow.")
            return float(self.model.predict(df)[0]), self.version


model_manager = ThreadSafeModelManager()
app = FastAPI(title="Real Estate MLOps Inference API", version="1.0.0")


def _background_poller():
    while True:
        model_manager.reload_if_needed(force=False)
        time.sleep(30)


@app.on_event("startup")
def startup_event():
    model_manager.reload_if_needed(force=True)
    threading.Thread(target=_background_poller, daemon=True).start()


def log_inference_to_db(payload: dict, prediction: Optional[float], version: str, status: str, error_msg: Optional[str] = None):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO raw_data.inference_logs (input_payload, prediction, model_version, status, error_message)
                    VALUES (:payload, :pred, :ver, :st, :err)
                """),
                {
                    "payload": json.dumps(payload),
                    "pred": prediction,
                    "ver": version,
                    "st": status,
                    "err": error_msg,
                }
            )
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": model_manager.version}


@app.post("/reload-model")
def reload_model(x_admin_token: Optional[str] = Header(default=None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token administrativo inválido.")
    return model_manager.reload_if_needed(force=True)


@app.post("/predict")
def predict_price(item: PropertyInput):
    start_t = time.perf_counter()
    payload = item.model_dump()
    try:
        parsed_date = pd.to_datetime(item.prev_sold_date, errors="coerce")
        has_prev = 0 if pd.isna(parsed_date) else 1
        years_since = -1.0 if pd.isna(parsed_date) else (pd.Timestamp.now() - parsed_date).days / 365.25

        df = pd.DataFrame([{
            "brokered_by": str(item.brokered_by or "UNKNOWN"),
            "status": str(item.status),
            "bed": float(item.bed),
            "bath": float(item.bath),
            "acre_lot": float(item.acre_lot),
            "city": str(item.city),
            "state": str(item.state),
            "zip_code": str(item.zip_code),
            "house_size": float(item.house_size),
            "years_since_sold": float(years_since),
            "has_prev_sold": int(has_prev),
        }])

        pred, version = model_manager.predict(df)
        INFERENCE_LATENCY.observe(time.perf_counter() - start_t)
        REQUEST_COUNT.labels(status="success", model_version=version).inc()
        log_inference_to_db(payload, pred, version, "SUCCESS")
        return {"predicted_price": round(pred, 2), "model_version": version, "status": "SUCCESS"}
    except Exception as exc:
        INFERENCE_ERRORS.inc()
        REQUEST_COUNT.labels(status="error", model_version=model_manager.version).inc()
        log_inference_to_db(payload, None, model_manager.version, "ERROR", str(exc))
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
