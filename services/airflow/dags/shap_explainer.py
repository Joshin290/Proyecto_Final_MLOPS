import json, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import mlflow
from mlflow.tracking import MlflowClient
import shap

MODEL_NAME = "real_estate_pricer"
FEATURE_COLS = [
    "brokered_by", "status", "bed", "bath", "acre_lot",
    "city", "state", "zip_code", "house_size",
    "years_since_sold", "has_prev_sold",
]

def run_shap_for_champion(sample_size=1500, out_dir="/tmp/shap_champion"):
    os.makedirs(out_dir, exist_ok=True)
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    engine = create_engine(os.environ["DATABASE_URL"])
    cols_sql = ", ".join(FEATURE_COLS + ["price"])
    df = pd.read_sql(f"SELECT {cols_sql} FROM clean_data.properties_clean ORDER BY random() LIMIT {sample_size}", engine)
    X_raw, y_true = df[FEATURE_COLS].copy(), df["price"].values
    preds = model.predict(X_raw)
    prep, reg = model.named_steps["preprocessor"], model.named_steps["regressor"]
    Xt = prep.transform(X_raw)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    enc_names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]

    explainer = shap.TreeExplainer(reg)
    sv = explainer.shap_values(Xt, check_additivity=False)
    base_val = float(np.ravel(explainer.expected_value)[0])

    grouped = np.zeros((len(X_raw), len(FEATURE_COLS)), dtype=float)
    f_idx = {f: i for i, f in enumerate(FEATURE_COLS)}
    for j, name in enumerate(enc_names):
        orig = name if name in f_idx else next((c for c in ("state", "status") if name.startswith(c + "_")), None)
        if orig:
            grouped[:, f_idx[orig]] += sv[:, j]

    mean_abs = np.abs(grouped).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    summary = sorted([
        {"feature": f, "mean_abs_shap_usd": round(float(mean_abs[i]), 2),
         "pct_impact": round(100.0 * float(mean_abs[i]) / total, 2),
         "mean_shap_usd": round(float(grouped[:, i].mean()), 2)}
        for i, f in enumerate(FEATURE_COLS)
    ], key=lambda x: x["mean_abs_shap_usd"], reverse=True)

    plt.close("all")
    plt.figure(figsize=(9, 5))
    s_df = pd.DataFrame(summary)[::-1]
    plt.barh(s_df["feature"], s_df["mean_abs_shap_usd"], color="#1f77b4")
    plt.xlabel("Impacto medio absoluto SHAP (USD)")
    plt.title(f"Importancia Global SHAP - {MODEL_NAME} v{mv.version} (@champion)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/01_shap_global_grouped.png", dpi=150); plt.close("all")

    exp_enc = shap.Explanation(values=sv, base_values=np.full(len(Xt), base_val), data=Xt, feature_names=enc_names)
    shap.plots.beeswarm(exp_enc, max_display=15, show=False)
    plt.gcf().savefig(f"{out_dir}/02_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close("all")
    X_disp = X_raw.copy()
    for c in X_disp.columns:
        X_disp[c] = X_disp[c].round(2) if pd.api.types.is_numeric_dtype(X_disp[c]) else X_disp[c].fillna("N/A").astype(str)

    order = np.argsort(preds)
    i_low, i_high = int(order[int(0.10 * len(order))]), int(order[int(0.95 * len(order))])
    for fname, idx in [("03_shap_waterfall_high.png", i_high), ("04_shap_waterfall_low.png", i_low)]:
        exp_row = shap.Explanation(values=grouped[idx], base_values=base_val, data=X_disp.iloc[idx].values, feature_names=FEATURE_COLS)
        shap.plots.waterfall(exp_row, max_display=11, show=False)
        plt.gcf().savefig(f"{out_dir}/{fname}", dpi=150, bbox_inches="tight"); plt.close("all")

    num_feats = ["house_size", "bath", "bed", "acre_lot", "years_since_sold"]
    top_num = next((r["feature"] for r in summary if r["feature"] in num_feats), "house_size")
    x_v = pd.to_numeric(X_raw[top_num], errors="coerce")
    hi = np.nanpercentile(x_v, 99)
    mask = x_v.notna() & (x_v <= hi)
    plt.figure(figsize=(8, 5))
    plt.scatter(x_v[mask], grouped[mask, f_idx[top_num]], alpha=0.5, s=16, c="#d62728")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel(top_num); plt.ylabel("Valor SHAP (USD)")
    plt.title(f"Dependencia SHAP: {top_num} (recorte P99)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/05_shap_dependence_top.png", dpi=150); plt.close("all")

    meta = {"model_version": mv.version, "run_id": mv.run_id, "sample_size": len(X_raw),
            "base_value_usd": round(base_val, 2), "mean_pred_usd": round(float(preds.mean()), 2),
            "top_numeric_feature": top_num, "global_importance": summary}
    with open(f"{out_dir}/shap_global.json", "w") as f:
        json.dump(meta, f, indent=2)

    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_artifacts(out_dir, artifact_path="shap_analysis")

    print(f"\n=== SHAP CAMPEON: {MODEL_NAME} v{mv.version} | Base E[f(X)]=${base_val:,.2f} ===")
    print(pd.DataFrame(summary).to_string(index=False))
    for lbl, idx in [("ALTA (P95)", i_high), ("BAJA (P10)", i_low)]:
        print(f"\n--- {lbl} | Pred=${preds[idx]:,.2f} | Real=${y_true[idx]:,.2f} ---")
        for i in np.argsort(np.abs(grouped[idx]))[::-1][:6]:
            print(f"  {FEATURE_COLS[i]:18s} ({X_disp.iloc[idx, i]}): {grouped[idx, i]:+,.2f} USD")
    return meta

if __name__ == "__main__":
    run_shap_for_champion()
