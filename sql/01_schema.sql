DROP TABLE IF EXISTS predictions_audit_log CASCADE;
DROP TABLE IF EXISTS clinical_notes CASCADE;
DROP TABLE IF EXISTS encounters CASCADE;
DROP TABLE IF EXISTS patients CASCADE;

CREATE TABLE patients (
    patient_token VARCHAR(100) PRIMARY KEY,
    race VARCHAR(50),
    gender VARCHAR(20),
    age_numeric DOUBLE PRECISION
);

CREATE TABLE encounters (
    encounter_id INTEGER PRIMARY KEY,
    patient_token VARCHAR(100) NOT NULL REFERENCES patients(patient_token) ON DELETE CASCADE,
    split VARCHAR(10) NOT NULL CHECK (split IN ('train', 'val', 'test')),
    time_in_hospital INTEGER,
    medical_specialty VARCHAR(100),
    admission_type_group VARCHAR(100),
    admission_source_group VARCHAR(100),
    discharge_group VARCHAR(100),
    diag_1_group VARCHAR(100),
    diag_2_group VARCHAR(100),
    diag_3_group VARCHAR(100),
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_diagnoses INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    total_prior_visits INTEGER,
    log_total_visits DOUBLE PRECISION,
    inpatient_ratio DOUBLE PRECISION,
    meds_per_day DOUBLE PRECISION,
    labs_per_day DOUBLE PRECISION,
    clinical_intensity DOUBLE PRECISION,
    has_prior_inpatient SMALLINT,
    has_prior_emergency SMALLINT,
    max_glu_serum VARCHAR(30),
    a1cresult VARCHAR(30),
    metformin VARCHAR(20),
    glimepiride VARCHAR(20),
    glipizide VARCHAR(20),
    glyburide VARCHAR(20),
    pioglitazone VARCHAR(20),
    rosiglitazone VARCHAR(20),
    insulin VARCHAR(20),
    change VARCHAR(20),
    diabetesmed VARCHAR(20),
    num_active_meds INTEGER,
    num_med_changes INTEGER,
    readmitted SMALLINT CHECK (readmitted IN (0, 1))
);

CREATE TABLE clinical_notes (
    note_id INTEGER PRIMARY KEY,
    encounter_id INTEGER NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    patient_token VARCHAR(100) NOT NULL REFERENCES patients(patient_token) ON DELETE CASCADE,
    clinical_note TEXT
);

CREATE TABLE predictions_audit_log (
    prediction_id SERIAL PRIMARY KEY,
    encounter_id INTEGER NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    patient_token VARCHAR(100) NOT NULL REFERENCES patients(patient_token) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    decision_threshold DOUBLE PRECISION NOT NULL,
    risk_tier VARCHAR(20) NOT NULL CHECK (risk_tier IN ('High Risk', 'Low Risk')),
    top_driver VARCHAR(100),
    top_driver_shap DOUBLE PRECISION,
    actual_readmitted SMALLINT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_encounters_patient ON encounters(patient_token);
CREATE INDEX idx_encounters_split ON encounters(split);
CREATE INDEX idx_encounters_discharge ON encounters(discharge_group);
CREATE INDEX idx_audit_risk_tier ON predictions_audit_log(risk_tier);