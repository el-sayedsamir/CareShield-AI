import os
from pathlib import Path
from getpass import getpass
from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine, text

# 1. إعداد الاتصال بقاعدة بيانات careshield_db بدون حفظ الباسورد في الملف
DB_USER = "postgres"
raw_password = os.getenv("PGPASSWORD") or getpass("Enter PostgreSQL password for 'postgres': ")
DB_PASS = quote_plus(raw_password)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "careshield_db"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 2. قراءة الداتا المؤمّنة وتوحيد أسماء الأعمدة لتطابق PostgreSQL
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "careshield_artifacts" / "final_secure_data.csv"

df_db = pd.read_csv(csv_path)
df_db.columns = df_db.columns.str.lower()
df_db["encounter_id"] = range(1, len(df_db) + 1)

# 3. تجهيز الـ 3 جداول الأساسية
patients_df = df_db[["patient_token", "race", "gender", "age_numeric"]].drop_duplicates(subset=["patient_token"])

encounters_cols = [
    "encounter_id", "patient_token", "split", "time_in_hospital", "medical_specialty",
    "admission_type_group", "admission_source_group", "discharge_group",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses",
    "number_outpatient", "number_emergency", "number_inpatient",
    "total_prior_visits", "log_total_visits", "inpatient_ratio",
    "meds_per_day", "labs_per_day", "clinical_intensity",
    "has_prior_inpatient", "has_prior_emergency", "max_glu_serum", "a1cresult",
    "metformin", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "insulin", "change", "diabetesmed",
    "num_active_meds", "num_med_changes", "readmitted"
]
encounters_df = df_db[encounters_cols]

notes_df = df_db[["encounter_id", "patient_token", "clinical_note"]].copy()
notes_df.insert(0, "note_id", notes_df["encounter_id"])

# 4. تفريغ الجداول ثم الرفع على دفعات
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE predictions_audit_log, clinical_notes, encounters, patients RESTART IDENTITY CASCADE;"))

print("Uploading patients...")
patients_df.to_sql("patients", engine, if_exists="append", index=False, method="multi", chunksize=2000)

print("Uploading encounters...")
encounters_df.to_sql("encounters", engine, if_exists="append", index=False, method="multi", chunksize=2000)

print("Uploading clinical_notes (may take ~15-30 seconds)...")
notes_df.to_sql("clinical_notes", engine, if_exists="append", index=False, method="multi", chunksize=500)

# 5. التحقق من الأعداد من داخل PostgreSQL مباشرة
print("\n--- Row Counts in PostgreSQL ---")
with engine.connect() as conn:
    for tbl in ["patients", "encounters", "clinical_notes", "predictions_audit_log"]:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
        print(f"{tbl}: {cnt:,} rows")