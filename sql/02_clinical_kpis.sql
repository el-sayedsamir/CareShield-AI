-- 1. High-Risk Triage Queue View
CREATE OR REPLACE VIEW vw_high_risk_triage_queue AS
SELECT
    ROW_NUMBER() OVER (ORDER BY a.risk_score DESC) AS priority_rank,
    a.encounter_id,
    a.patient_token,
    p.gender,
    p.age_numeric,
    e.discharge_group,
    e.diag_1_group,
    e.time_in_hospital,
    e.number_inpatient AS prior_inpatient_visits,
    ROUND(a.risk_score::numeric, 4) AS risk_score,
    a.risk_tier,
    a.top_driver,
    ROUND(a.top_driver_shap::numeric, 4) AS top_driver_shap,
    a.actual_readmitted,
    LEFT(cn.clinical_note, 150) AS clinical_note_preview,
    a.predicted_at
FROM predictions_audit_log a
JOIN encounters e      ON a.encounter_id = e.encounter_id
JOIN patients p        ON a.patient_token = p.patient_token
JOIN clinical_notes cn ON a.encounter_id = cn.encounter_id
WHERE a.risk_tier = 'High Risk';


-- 2. Risk Tier Stratification & Lift vs Baseline
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



-- 3. Top 10 Clinical Drivers (SHAP) Behind High-Risk Alerts
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


-- 4. Readmission Rate by Discharge Disposition (Full Cohort: 69,970 Encounters)
SELECT
    discharge_group,
    COUNT(*) AS total_encounters,
    ROUND(AVG(time_in_hospital)::numeric, 2) AS avg_los_days,
    ROUND(AVG(num_medications)::numeric, 2) AS avg_medications,
    ROUND(100.0 * AVG(readmitted)::numeric, 2) AS readmission_rate_pct
FROM encounters
GROUP BY discharge_group
HAVING COUNT(*) >= 100
ORDER BY readmission_rate_pct DESC;



-- 5. Top 20 Highest-Risk Patients Triage Queue (Ready for Clinical Dashboard)
SELECT *
FROM vw_high_risk_triage_queue
LIMIT 20;