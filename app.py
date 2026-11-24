import os
import json
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ============================================================================
# 1. إعداد الصفحة وتحميل المتغيرات البيئية
# ============================================================================
st.set_page_config(
    page_title="CareShield AI | 30-Day Readmission Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@st.cache_resource
def get_db_engine():
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = quote_plus(os.getenv("DB_PASS", ""))
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "careshield_db")
    url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(url, pool_pre_ping=True)


@st.cache_data
def load_config():
    with open(BASE_DIR / "careshield_artifacts" / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def run_query(sql: str) -> pd.DataFrame:
    engine = get_db_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

@st.cache_resource
def load_production_models():
    art_dir = BASE_DIR / "careshield_artifacts"
    return {
        "preprocessor": joblib.load(art_dir / "preprocessor.joblib"),
        "xgb": joblib.load(art_dir / "xgb_tuned.joblib"),
        "rf": joblib.load(art_dir / "rf_model.joblib"),
        "lr": joblib.load(art_dir / "lr_model.joblib"),
    }


@st.cache_data(ttl=300)
def get_simulator_options():
    sample_patients = run_query("""
        (
            SELECT e.*, p.race, p.gender, p.age_numeric, a.risk_score AS logged_risk_score, a.risk_tier AS logged_risk_tier
            FROM encounters e
            JOIN patients p ON e.patient_token = p.patient_token
            JOIN predictions_audit_log a ON e.encounter_id = a.encounter_id
            WHERE a.risk_tier = 'High Risk'
            ORDER BY a.risk_score DESC
            LIMIT 15
        )
        UNION ALL
        (
            SELECT e.*, p.race, p.gender, p.age_numeric, a.risk_score AS logged_risk_score, a.risk_tier AS logged_risk_tier
            FROM encounters e
            JOIN patients p ON e.patient_token = p.patient_token
            JOIN predictions_audit_log a ON e.encounter_id = a.encounter_id
            WHERE a.risk_tier = 'Low Risk'
            ORDER BY a.risk_score ASC
            LIMIT 15
        );
    """)

    cat_options = {}
    for tbl, col in [
        ("patients", "race"),
        ("patients", "gender"),
        ("encounters", "discharge_group"),
        ("encounters", "admission_type_group"),
        ("encounters", "admission_source_group"),
        ("encounters", "diag_1_group"),
        ("encounters", "medical_specialty"),
        ("encounters", "max_glu_serum"),
        ("encounters", "a1cresult"),
        ("encounters", "insulin"),
        ("encounters", "metformin"),
        ("encounters", "change"),
        ("encounters", "diabetesmed"),
    ]:
        vals = run_query(f"SELECT DISTINCT {col} FROM {tbl} ORDER BY {col};")[col].tolist()
        cat_options[col] = [v if pd.notna(v) else "Missing" for v in vals]

    return sample_patients, cat_options

# ============================================================================
# 2. الشريط الجانبي (Sidebar: حالة النظام ومعلومات الموديل)
# ============================================================================
config = load_config()

with st.sidebar:
    st.title("🛡️ CareShield AI")
    st.caption("Enterprise 30-Day Hospital Readmission Risk Engine")
    st.divider()

    # اختبار الاتصال بقاعدة البيانات
    try:
        total_encounters = int(run_query("SELECT COUNT(*) AS cnt FROM encounters")["cnt"].iloc[0])
        st.success("PostgreSQL Connected (`careshield_db`)")
    except Exception as e:
        st.error("Database connection failed. Check your `.env` password.")
        st.exception(e)
        st.stop()

    st.subheader("Production Model Specs")
    st.markdown(
        f"""
        - **Ensemble:** XGB ({config['ensemble_weights']['xgb']}) + RF ({config['ensemble_weights']['rf']}) + LR ({config['ensemble_weights']['lr']})
        - **Decision Threshold:** `{config['decision_threshold']}`
        - **Test ROC-AUC:** `{config['test_roc_auc']}`
        - **Test PR-AUC:** `{config['test_pr_auc']}`
        - **Privacy:** HIPAA De-identified (`patient_token`)
        """
    )

    st.divider()
    if st.button("🔄 Refresh Live DB Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# 3. العنوان الرئيسي والتبويبات الـ 3
# ============================================================================
st.title("🛡️ CareShield AI — Clinical Readmission Command Center")
st.caption("Live analytics powered by PostgreSQL, Weighted ML Ensemble, and TreeSHAP Explainability")

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Clinical Dashboard",
    "🚨 High-Risk Triage Queue",
    "🧪 Live Risk Simulator & SHAP",
])

# ============================================================================
# التبويب 1: لوحة المؤشرات التنفيذية (Executive Clinical Dashboard)
# ============================================================================
with tab1:
    # سحب البيانات الحية من PostgreSQL
    df_strat = run_query("""
        WITH baseline AS (
            SELECT AVG(actual_readmitted) AS base_rate
            FROM predictions_audit_log
        )
        SELECT
            a.risk_tier,
            COUNT(*) AS patient_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS cohort_share_pct,
            ROUND(AVG(a.risk_score)::numeric, 4) AS avg_risk_score,
            ROUND(100.0 * AVG(a.actual_readmitted)::numeric, 2) AS actual_readmission_pct,
            ROUND((AVG(a.actual_readmitted) / NULLIF(b.base_rate, 0))::numeric, 2) AS lift_vs_baseline
        FROM predictions_audit_log a
        CROSS JOIN baseline b
        GROUP BY a.risk_tier, b.base_rate
        ORDER BY avg_risk_score DESC;
    """)

    df_discharge = run_query("""
        SELECT
            discharge_group,
            COUNT(*) AS total_encounters,
            ROUND(AVG(time_in_hospital)::numeric, 2) AS avg_los_days,
            ROUND(AVG(num_medications)::numeric, 2) AS avg_medications,
            ROUND(100.0 * AVG(readmitted)::numeric, 2) AS readmission_rate_pct
        FROM encounters
        GROUP BY discharge_group
        HAVING COUNT(*) >= 100
        ORDER BY readmission_rate_pct ASC;
    """)

    df_shap = run_query("""
        SELECT
            top_driver,
            COUNT(*) AS high_risk_alerts,
            ROUND(AVG(top_driver_shap)::numeric, 4) AS avg_shap_impact,
            ROUND(100.0 * AVG(actual_readmitted)::numeric, 2) AS actual_readmission_pct
        FROM predictions_audit_log
        WHERE risk_tier = 'High Risk'
        GROUP BY top_driver
        ORDER BY high_risk_alerts DESC
        LIMIT 10;
    """)

    # استخراج أرقام الـ KPIs
    audited_total = int(df_strat["patient_count"].sum())
    high_row = df_strat[df_strat["risk_tier"] == "High Risk"].iloc[0]
    low_row = df_strat[df_strat["risk_tier"] == "Low Risk"].iloc[0]
    risk_ratio = float(high_row["actual_readmission_pct"]) / max(float(low_row["actual_readmission_pct"]), 0.01)

    # صف الكروت العلوية (4 KPI Cards)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total EHR Encounters", f"{total_encounters:,}", "Full PostgreSQL Cohort")
    c2.metric("Audited Test Cohort", f"{audited_total:,}", "Live ML + SHAP Scored")
    c3.metric(
        "High-Risk Alerts",
        f"{int(high_row['patient_count']):,}",
        f"{high_row['cohort_share_pct']}% of Test Cohort",
    )
    c4.metric(
        "High-Risk Actual Readmission",
        f"{high_row['actual_readmission_pct']}%",
        f"{high_row['lift_vs_baseline']}x Baseline ({risk_ratio:.2f}x vs Low Risk)",
    )

    st.divider()

    # الصف الأول من الرسومات
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Tier Stratification (Actual Readmission %)")
        fig_strat = px.bar(
            df_strat,
            x="risk_tier",
            y="actual_readmission_pct",
            color="risk_tier",
            text=df_strat["actual_readmission_pct"].apply(lambda x: f"{x:.2f}%"),
            color_discrete_map={"High Risk": "#EF4444", "Low Risk": "#10B981"},
            hover_data=["patient_count", "cohort_share_pct", "avg_risk_score", "lift_vs_baseline"],
            labels={
                "risk_tier": "Predicted Risk Tier",
                "actual_readmission_pct": "Actual 30-Day Readmission (%)",
            },
        )
        fig_strat.update_traces(textposition="outside")
        fig_strat.update_layout(showlegend=False, height=380, yaxis_range=[0, max(df_strat["actual_readmission_pct"]) * 1.25])
        st.plotly_chart(fig_strat, use_container_width=True)

    with col_right:
        st.subheader("Readmission Rate by Discharge Disposition")
        fig_disc = px.bar(
            df_discharge,
            x="readmission_rate_pct",
            y="discharge_group",
            orientation="h",
            text=df_discharge["readmission_rate_pct"].apply(lambda x: f"{x:.2f}%"),
            color="readmission_rate_pct",
            color_continuous_scale="Reds",
            hover_data=["total_encounters", "avg_los_days", "avg_medications"],
            labels={
                "readmission_rate_pct": "30-Day Readmission Rate (%)",
                "discharge_group": "Discharge Disposition",
            },
        )
        fig_disc.update_traces(textposition="outside")
        fig_disc.update_layout(
            coloraxis_showscale=False,
            height=380,
            xaxis_range=[0, max(df_discharge["readmission_rate_pct"]) * 1.2],
        )
        st.plotly_chart(fig_disc, use_container_width=True)

    st.divider()

    # الصف الثاني: أهم 10 أسباب SHAP
    st.subheader("Top 10 Clinical Drivers (TreeSHAP) Behind High-Risk Alerts")
    df_shap_plot = df_shap.sort_values("high_risk_alerts", ascending=True)
    fig_shap = px.bar(
        df_shap_plot,
        x="high_risk_alerts",
        y="top_driver",
        orientation="h",
        text="high_risk_alerts",
        color="actual_readmission_pct",
        color_continuous_scale="YlOrRd",
        hover_data=["avg_shap_impact", "actual_readmission_pct"],
        labels={
            "high_risk_alerts": "Number of High-Risk Alerts",
            "top_driver": "Primary SHAP Clinical Driver",
            "actual_readmission_pct": "Actual Readmission %",
        },
    )
    fig_shap.update_traces(textposition="outside")
    fig_shap.update_layout(height=420, xaxis_range=[0, max(df_shap_plot["high_risk_alerts"]) * 1.15])
    st.plotly_chart(fig_shap, use_container_width=True)

    with st.expander("📋 View Underlying SQL Analytics Tables"):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.markdown("**Risk Stratification Summary**")
            st.dataframe(df_strat, use_container_width=True, hide_index=True)
        with t_col2:
            st.markdown("**Discharge Disposition Breakdown**")
            st.dataframe(df_discharge.sort_values("readmission_rate_pct", ascending=False), use_container_width=True, hide_index=True)


# ============================================================================
# التبويب 2 و 3 (هنبنيهم في الخطوات الجاية)
# ============================================================================
with tab2:
    st.subheader("🚨 Prioritized High-Risk Patient Triage Queue")
    st.caption("Live feed from PostgreSQL view `vw_high_risk_triage_queue` ordered by predicted 30-day readmission risk")

    df_triage = run_query("SELECT * FROM vw_high_risk_triage_queue ORDER BY priority_rank;")

    # توحيد أنواع الأرقام لضمان عمل الفلاتر والرسوم بسلاسة
    df_triage["risk_score"] = df_triage["risk_score"].astype(float)
    df_triage["top_driver_shap"] = df_triage["top_driver_shap"].astype(float)
    df_triage["age_numeric"] = df_triage["age_numeric"].astype(int)

    # صف الفلاتر التفاعلية
    f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.1, 1.1])

    with f1:
        discharge_opts = sorted(df_triage["discharge_group"].dropna().unique().tolist())
        selected_discharge = st.multiselect(
            "Discharge Disposition",
            options=discharge_opts,
            default=discharge_opts,
        )

    with f2:
        diag_opts = sorted(df_triage["diag_1_group"].dropna().unique().tolist())
        selected_diag = st.multiselect(
            "Primary Diagnosis",
            options=diag_opts,
            default=diag_opts,
        )

    with f3:
        min_score_val = float(df_triage["risk_score"].min())
        max_score_val = float(df_triage["risk_score"].max())
        min_risk_filter = st.slider(
            "Min Risk Score",
            min_value=round(min_score_val, 2),
            max_value=round(max_score_val, 2),
            value=round(min_score_val, 2),
            step=0.01,
        )

    with f4:
        min_age = int(df_triage["age_numeric"].min())
        max_age = int(df_triage["age_numeric"].max())
        age_range = st.slider(
            "Patient Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
        )

    # تطبيق الفلاتر
    filtered_df = df_triage[
        (df_triage["discharge_group"].isin(selected_discharge))
        & (df_triage["diag_1_group"].isin(selected_diag))
        & (df_triage["risk_score"] >= min_risk_filter)
        & (df_triage["age_numeric"].between(age_range[0], age_range[1]))
    ].copy()

    # ملخص القائمة المفلترة + زرار تصدير CSV
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1.2])
    m1.metric("Matching Patients", f"{len(filtered_df):,}")
    m2.metric(
        "Avg Risk Score",
        f"{filtered_df['risk_score'].mean():.4f}" if not filtered_df.empty else "0.0000",
    )
    m3.metric(
        "Avg Length of Stay",
        f"{filtered_df['time_in_hospital'].mean():.1f} days" if not filtered_df.empty else "0 days",
    )
    with m4:
        st.write("")
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Queue (CSV)",
            data=csv_bytes,
            file_name="careshield_high_risk_triage_queue.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=filtered_df.empty,
        )

    if filtered_df.empty:
        st.warning("No patients match the current filters. Try widening your filter criteria.")
    else:
        # عرض الجدول التفاعلي مع شريط تقدم لدرجة الخطورة
        display_cols = [
            "priority_rank",
            "encounter_id",
            "patient_token",
            "gender",
            "age_numeric",
            "discharge_group",
            "diag_1_group",
            "time_in_hospital",
            "prior_inpatient_visits",
            "risk_score",
            "top_driver",
            "top_driver_shap",
            "actual_readmitted",
        ]

        st.dataframe(
            filtered_df[display_cols].head(200),
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "priority_rank": st.column_config.NumberColumn("Rank", format="#%d"),
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                ),
                "top_driver_shap": st.column_config.NumberColumn("SHAP Impact", format="%+.4f"),
                "age_numeric": st.column_config.NumberColumn("Age", format="%d yrs"),
                "time_in_hospital": st.column_config.NumberColumn("LOS (Days)"),
            },
        )
        st.caption("Showing top 200 matching patients in table (full filtered list available via CSV export).")

        st.divider()

        # عارض تفاصيل المريض والملاحظة الطبية الكاملة من PostgreSQL
        st.subheader("🩺 Patient Case Inspector & Full De-identified Clinical Note")

        inspect_df = filtered_df.head(100).reset_index(drop=True)
        options_labels = [
            f"Rank #{int(r.priority_rank)} | Encounter {int(r.encounter_id)} | Risk: {r.risk_score:.4f} | {r.gender}, {int(r.age_numeric)}y | {r.discharge_group}"
            for r in inspect_df.itertuples()
        ]
        selected_idx = st.selectbox(
            "Select a high-risk patient to inspect full clinical context:",
            options=range(len(options_labels)),
            format_func=lambda i: options_labels[i],
        )

        selected_row = inspect_df.iloc[selected_idx]

        # سحب الملاحظة الطبية الكاملة من جدول clinical_notes
        engine = get_db_engine()
        with engine.connect() as conn:
            full_note = conn.execute(
                text("SELECT clinical_note FROM clinical_notes WHERE encounter_id = :eid"),
                {"eid": int(selected_row["encounter_id"])},
            ).scalar()

        d_col1, d_col2 = st.columns([1, 1.6])

        with d_col1:
            st.markdown("#### Clinical Risk Profile")
            st.markdown(
                f"""
                - **Patient Token (HIPAA):** `{selected_row['patient_token']}`
                - **Encounter ID:** `{int(selected_row['encounter_id'])}`
                - **Demographics:** {selected_row['gender']}, Age {int(selected_row['age_numeric'])}
                - **Discharge Group:** `{selected_row['discharge_group']}`
                - **Primary Diagnosis:** `{selected_row['diag_1_group']}`
                - **Length of Stay:** `{int(selected_row['time_in_hospital'])} days`
                - **Prior Inpatient Visits:** `{int(selected_row['prior_inpatient_visits'])}`
                - **Predicted Risk Score:** **`{selected_row['risk_score']:.4f}`** (`{selected_row['risk_tier']}`)
                - **Primary SHAP Driver:** `{selected_row['top_driver']}` (`{selected_row['top_driver_shap']:+.4f}`)
                """
            )

        with d_col2:
            st.markdown("#### Full De-identified Clinical Note (from `clinical_notes`)")
            st.text_area(
                label="Clinical Note",
                value=full_note or "No clinical note available for this encounter.",
                height=260,
                disabled=True,
                label_visibility="collapsed",
            )



with tab3:
    st.subheader("🧪 Live Clinical Risk Simulator & Real-Time TreeSHAP Explainer")
    st.caption("Load a real de-identified patient encounter from PostgreSQL and adjust clinical levers to simulate What-If readmission risk.")

    models = load_production_models()
    sample_patients, cat_options = get_simulator_options()

    # اختيار حالة مريض حقيقية كنقطة بداية للمحاكاة
    case_labels = [
        f"[{r.logged_risk_tier}] Encounter #{int(r.encounter_id)} | Logged Risk: {float(r.logged_risk_score):.4f} | {r.gender}, {int(r.age_numeric)}y | {r.discharge_group}"
        for r in sample_patients.itertuples()
    ]
    selected_case_idx = st.selectbox(
        "Select a baseline patient case from PostgreSQL (15 Highest-Risk & 15 Lowest-Risk):",
        options=range(len(case_labels)),
        format_func=lambda i: case_labels[i],
    )
    base_row = sample_patients.iloc[selected_case_idx]

    def _safe_val(val):
        return val if pd.notna(val) else "Missing"

    def _safe_idx(options_list, val):
        clean_v = _safe_val(val)
        return options_list.index(clean_v) if clean_v in options_list else 0

    st.markdown("#### Adjust Clinical Parameters (What-If Scenario)")
    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        st.markdown("**Demographics & Disposition**")
        sim_age = st.slider("Age (Years)", 5, 95, int(base_row["age_numeric"]), step=5)
        sim_gender = st.selectbox("Gender", cat_options["gender"], index=_safe_idx(cat_options["gender"], base_row["gender"]))
        sim_discharge = st.selectbox(
            "Discharge Disposition",
            cat_options["discharge_group"],
            index=_safe_idx(cat_options["discharge_group"], base_row["discharge_group"]),
        )
        sim_diag1 = st.selectbox(
            "Primary Diagnosis",
            cat_options["diag_1_group"],
            index=_safe_idx(cat_options["diag_1_group"], base_row["diag_1_group"]),
        )
        sim_adm_type = st.selectbox(
            "Admission Type",
            cat_options["admission_type_group"],
            index=_safe_idx(cat_options["admission_type_group"], base_row["admission_type_group"]),
        )

    with col_in2:
        st.markdown("**Utilization & Hospital Stay**")
        sim_los = st.slider("Length of Stay (Days)", 1, 14, int(base_row["time_in_hospital"]))
        sim_inpatient = st.number_input("Prior Inpatient Visits", 0, 25, int(base_row["number_inpatient"]))
        sim_emergency = st.number_input("Prior Emergency Visits", 0, 40, int(base_row["number_emergency"]))
        sim_outpatient = st.number_input("Prior Outpatient Visits", 0, 40, int(base_row["number_outpatient"]))
        sim_diagnoses = st.slider("Number of Diagnoses", 1, 16, int(base_row["number_diagnoses"]))

    with col_in3:
        st.markdown("**Labs & Medications**")
        sim_meds = st.slider("Number of Medications", 1, 80, int(base_row["num_medications"]))
        sim_labs = st.slider("Number of Lab Procedures", 1, 130, int(base_row["num_lab_procedures"]))
        sim_procs = st.slider("Number of Procedures", 0, 6, int(base_row["num_procedures"]))
        sim_a1c = st.selectbox("A1C Result", cat_options["a1cresult"], index=_safe_idx(cat_options["a1cresult"], base_row["a1cresult"]))
        sim_insulin = st.selectbox("Insulin Status", cat_options["insulin"], index=_safe_idx(cat_options["insulin"], base_row["insulin"]))

    # تجهيز سجل الإدخال وتحديث الميزات الهندسية تلقائياً عند التعديل
    sim_record = base_row.to_dict()
    sim_record["age_numeric"] = sim_age
    sim_record["gender"] = None if sim_gender == "Missing" else sim_gender
    sim_record["discharge_group"] = None if sim_discharge == "Missing" else sim_discharge
    sim_record["diag_1_group"] = None if sim_diag1 == "Missing" else sim_diag1
    sim_record["admission_type_group"] = None if sim_adm_type == "Missing" else sim_adm_type
    sim_record["time_in_hospital"] = sim_los
    sim_record["number_inpatient"] = sim_inpatient
    sim_record["number_emergency"] = sim_emergency
    sim_record["number_outpatient"] = sim_outpatient
    sim_record["number_diagnoses"] = sim_diagnoses
    sim_record["num_medications"] = sim_meds
    sim_record["num_lab_procedures"] = sim_labs
    sim_record["num_procedures"] = sim_procs
    sim_record["a1cresult"] = None if sim_a1c == "Missing" else sim_a1c
    sim_record["insulin"] = None if sim_insulin == "Missing" else sim_insulin

    # إعادة حساب الميزات المشتقة بدقة إذا غيّر المستخدم أياً من القيم الأساسية
    visits_changed = (
        sim_inpatient != int(base_row["number_inpatient"])
        or sim_emergency != int(base_row["number_emergency"])
        or sim_outpatient != int(base_row["number_outpatient"])
    )
    if visits_changed:
        tot_v = sim_inpatient + sim_emergency + sim_outpatient
        sim_record["total_prior_visits"] = tot_v
        sim_record["log_total_visits"] = float(np.log1p(tot_v))
        sim_record["has_prior_inpatient"] = int(sim_inpatient > 0)
        sim_record["has_prior_emergency"] = int(sim_emergency > 0)
        # مطابقة صيغة inpatient_ratio المستخدمة في قاعدة البيانات
        base_tot = int(base_row["total_prior_visits"])
        base_inp = int(base_row["number_inpatient"])
        if base_tot > 0 and abs(float(base_row["inpatient_ratio"]) - (base_inp / base_tot)) < 1e-3:
            sim_record["inpatient_ratio"] = float(sim_inpatient / tot_v) if tot_v > 0 else 0.0
        else:
            sim_record["inpatient_ratio"] = float(sim_inpatient / (tot_v + 1))

    stay_changed = (
        sim_los != int(base_row["time_in_hospital"])
        or sim_meds != int(base_row["num_medications"])
        or sim_labs != int(base_row["num_lab_procedures"])
        or sim_procs != int(base_row["num_procedures"])
    )
    if stay_changed:
        sim_record["meds_per_day"] = float(sim_meds / max(sim_los, 1))
        sim_record["labs_per_day"] = float(sim_labs / max(sim_los, 1))
        sim_record["clinical_intensity"] = float((sim_labs + sim_procs + sim_meds) / max(sim_los, 1))

    if sim_insulin != _safe_val(base_row["insulin"]):
        med_cols = ["metformin", "glimepiride", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "insulin"]
        sim_record["num_active_meds"] = sum(1 for c in med_cols if sim_record.get(c) not in (None, "No", "Missing"))
        sim_record["num_med_changes"] = sum(1 for c in med_cols if sim_record.get(c) in ("Up", "Down"))

    # مطابقة أسماء الأعمدة وتشغيل الـ ML Ensemble
    feature_cols = config["feature_columns"]
    col_case_map = {c.lower(): c for c in feature_cols}
    X_sim_df = pd.DataFrame([sim_record]).rename(columns=col_case_map)[feature_cols]
    X_sim_df = X_sim_df.mask(X_sim_df.isna(), np.nan)

    X_proc = models["preprocessor"].transform(X_sim_df)
    w = config["ensemble_weights"]
    threshold = float(config["decision_threshold"])

    xgb_p = float(models["xgb"].predict_proba(X_proc)[0, 1])
    rf_p = float(models["rf"].predict_proba(X_proc)[0, 1])
    lr_p = float(models["lr"].predict_proba(X_proc)[0, 1])
    ensemble_p = w["xgb"] * xgb_p + w["rf"] * rf_p + w["lr"] * lr_p

    pred_tier = "High Risk" if ensemble_p >= threshold else "Low Risk"
    delta_vs_logged = ensemble_p - float(base_row["logged_risk_score"])

    # حساب قيم TreeSHAP اللحظية لهذا المريض
    raw_names = list(models["preprocessor"].get_feature_names_out())
    clean_names = [n.split("__", 1)[-1] for n in raw_names]
    dmat = xgb.DMatrix(X_proc)
    shap_vals = models["xgb"].get_booster().predict(dmat, pred_contribs=True)[0, :-1]

    shap_df = pd.DataFrame({"Feature": clean_names, "SHAP Value": shap_vals})
    shap_df["Abs"] = shap_df["SHAP Value"].abs()
    top_shap_df = shap_df.sort_values("Abs", ascending=False).head(10).sort_values("SHAP Value", ascending=True)
    top_shap_df["Direction"] = np.where(top_shap_df["SHAP Value"] >= 0, "Increases Risk", "Reduces Risk")

    st.divider()

    # عرض النتائج الحية (عداد الخطورة + رسم SHAP)
    res_left, res_right = st.columns([1, 1.4])

    with res_left:
        st.subheader("Live Risk Prediction")
        gauge_color = "#EF4444" if pred_tier == "High Risk" else "#10B981"

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=ensemble_p * 100,
                number={"suffix": "%", "valueformat": ".2f"},
                delta={
                    "reference": float(base_row["logged_risk_score"]) * 100,
                    "suffix": "%",
                    "valueformat": ".2f",
                    "increasing": {"color": "#EF4444"},
                    "decreasing": {"color": "#10B981"},
                },
                title={"text": f"Decision: <b>{pred_tier}</b> (Threshold: {threshold*100:.2f}%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#ECFDF5"},
                        {"range": [threshold * 100, 100], "color": "#FEF2F2"},
                    ],
                    "threshold": {
                        "line": {"color": "#111827", "width": 3},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=300, margin=dict(l=35, r=35, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # تفاصيل توقعات النماذج الـ 3
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("XGBoost (50%)", f"{xgb_p:.4f}")
        m_c2.metric("Random Forest (30%)", f"{rf_p:.4f}")
        m_c3.metric("Logistic Reg (20%)", f"{lr_p:.4f}")

        if pred_tier == "High Risk":
            st.error("🚨 **Clinical Action Recommended:** Schedule 48-hour post-discharge nurse call & medication reconciliation.")
        else:
            st.success("✅ **Standard Care Pathway:** Routine discharge instructions and standard outpatient follow-up.")

    with res_right:
        st.subheader("Per-Patient TreeSHAP Explanation (Top 10 Drivers)")
        fig_local_shap = px.bar(
            top_shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={"Increases Risk": "#EF4444", "Reduces Risk": "#10B981"},
            text=top_shap_df["SHAP Value"].apply(lambda v: f"{v:+.3f}"),
            labels={"SHAP Value": "Log-Odds Impact on 30-Day Readmission Risk", "Feature": "Clinical Feature"},
        )
        fig_local_shap.update_traces(textposition="outside")
        fig_local_shap.add_vline(x=0, line_width=1.5, line_color="#374151")
        max_abs = max(float(top_shap_df["Abs"].max()), 0.1)
        fig_local_shap.update_layout(
            height=420,
            xaxis_range=[-max_abs * 1.45, max_abs * 1.45],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text="Risk Effect"),
        )
        st.plotly_chart(fig_local_shap, use_container_width=True)

        