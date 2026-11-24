import os
import json
from pathlib import Path
from getpass import getpass
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. إعداد الاتصال بقاعدة بيانات careshield_db
DB_USER = "postgres"
raw_password = os.getenv("PGPASSWORD") or getpass("Enter PostgreSQL password for 'postgres': ")
DB_PASS = quote_plus(raw_password)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "careshield_db"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 2. تحميل ملفات الإنتاج (Artifacts & Config)
base_dir = Path(__file__).resolve().parent
art_dir = base_dir / "careshield_artifacts"

with open(art_dir / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

feature_cols = config["feature_columns"]
weights = config["ensemble_weights"]
threshold = float(config["decision_threshold"])

print("Loading production models...")
preprocessor = joblib.load(art_dir / "preprocessor.joblib")
xgb_model = joblib.load(art_dir / "xgb_tuned.joblib")
rf_model = joblib.load(art_dir / "rf_model.joblib")
lr_model = joblib.load(art_dir / "lr_model.joblib")

# 3. سحب بيانات مجموعة الاختبار (test) من PostgreSQL مباشرة عبر JOIN
query = """
SELECT
    e.*,
    p.race,
    p.gender,
    p.age_numeric
FROM encounters e
JOIN patients p ON e.patient_token = p.patient_token
WHERE e.split = 'test'
ORDER BY e.encounter_id;
"""

print("Fetching test encounters from PostgreSQL...")
df_test = pd.read_sql(query, engine)
print(f"Fetched {len(df_test):,} test records from PostgreSQL.")

# مطابقة أسماء الأعمدة مع الحالة الأصلية المتوقعة في preprocessor.joblib (مثل A1Cresult و diabetesMed)
col_case_map = {col.lower(): col for col in feature_cols}
df_test_aligned = df_test.rename(columns=col_case_map)

X_test = df_test_aligned[feature_cols].copy()
X_test = X_test.mask(X_test.isna(), np.nan)
y_test = df_test["readmitted"].astype(int).values

# 4. المعالجة والتنبؤ بالـ Ensemble
print("Running ML Ensemble inference...")
X_proc = preprocessor.transform(X_test)

xgb_prob = xgb_model.predict_proba(X_proc)[:, 1]
rf_prob = rf_model.predict_proba(X_proc)[:, 1]
lr_prob = lr_model.predict_proba(X_proc)[:, 1]

risk_scores = (
    weights["xgb"] * xgb_prob
    + weights["rf"] * rf_prob
    + weights["lr"] * lr_prob
)
risk_tiers = np.where(risk_scores >= threshold, "High Risk", "Low Risk")

# التحقق من تطابق الأداء مع نتائج التدريب
roc_auc = roc_auc_score(y_test, risk_scores)
pr_auc = average_precision_score(y_test, risk_scores)
print(f"\n--- Live DB Inference Verification ---")
print(f"Test ROC-AUC: {roc_auc:.4f} (Expected: {config['test_roc_auc']:.4f})")
print(f"Test PR-AUC : {pr_auc:.4f} (Expected: {config['test_pr_auc']:.4f})")

# 5. حساب TreeSHAP لكل مريض لاستخراج أهم عامل سريري مؤثر (top_driver)
print("\nComputing per-patient SHAP top drivers...")
if hasattr(preprocessor, "get_feature_names_out"):
    raw_names = list(preprocessor.get_feature_names_out())
    clean_names = [n.split("__", 1)[-1][:100] for n in raw_names]
else:
    clean_names = [f"feature_{i}" for i in range(X_proc.shape[1])]

if hasattr(xgb_model, "get_booster"):
    import xgboost as xgb
    dmat = xgb.DMatrix(X_proc)
    contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]  # استبعاد عمود الـ bias الأخير
else:
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(X_proc)
    shap_vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)

top_idx = np.argmax(np.abs(shap_vals), axis=1)
top_drivers = [clean_names[i] for i in top_idx]
top_driver_shaps = [float(shap_vals[row_i, col_i]) for row_i, col_i in enumerate(top_idx)]

# 6. تجهيز جدول التدقيق ورفعه إلى predictions_audit_log
audit_df = pd.DataFrame({
    "encounter_id": df_test["encounter_id"].astype(int),
    "patient_token": df_test["patient_token"],
    "model_version": "CareShield_Ensemble_v1.0",
    "risk_score": np.round(risk_scores, 6),
    "decision_threshold": threshold,
    "risk_tier": risk_tiers,
    "top_driver": top_drivers,
    "top_driver_shap": np.round(top_driver_shaps, 6),
    "actual_readmitted": y_test,
})

print("Writing audit log to PostgreSQL...")
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE predictions_audit_log RESTART IDENTITY;"))

audit_df.to_sql(
    "predictions_audit_log",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=2000,
)

# 7. ملخص التحقق من داخل PostgreSQL
print("\n--- PostgreSQL Audit Log Summary ---")
summary_sql = """
SELECT
    risk_tier,
    COUNT(*) AS patient_count,
    ROUND(AVG(risk_score)::numeric, 4) AS avg_risk_score,
    ROUND(100.0 * AVG(actual_readmitted)::numeric, 2) AS actual_readmission_pct
FROM predictions_audit_log
GROUP BY risk_tier
ORDER BY avg_risk_score DESC;
"""
with engine.connect() as conn:
    summary_df = pd.read_sql(text(summary_sql), conn)
    print(summary_df.to_string(index=False))