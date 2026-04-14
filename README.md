# Obesity Predictor — Project README

## Overview
- Objective: Predict obesity category (Normal / Overweight / Obese) from routine biomarkers to aid early risk stratification.
- Pipeline: User inputs → feature engineering (HOMA‑IR, TG/HDL, interactions) → CatBoost model → probabilities + interpretation.

## Quick Start
1. Create and activate a Python venv.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app locally:

```bash
streamlit run app.py
```

## Files
- `app.py`: Streamlit UI and inference pipeline.
- `final_model.pkl`: Trained CatBoost model artifact.
- `requirements.txt`: Python dependencies.
- `runtime.txt`: Python version for deployment.

## Short Answers for Viva (Concise)

### Project Summary
- Builds a clinical decision‑support web app to classify obesity using biomarkers and engineered metabolic indicators; served via Streamlit for easy demo and validation.

### Data & Features
- Data: clinical/observational biomarker dataset (describe origin and counts in your viva). Target: categorical obesity label (1/2/3). Feature engineering: HOMA‑IR = (Insulin*Glucose)/405, TG/HDL ratio, Insulin_TG and Glucose_TG capture interactions.

### Model & Training
- Model: CatBoost chosen for tabular performance and robustness. Tuned learning rate, depth, iterations; used early stopping and CV where appropriate. Addressed class imbalance via stratified sampling or weighting.

### Evaluation
- Report accuracy and per‑class precision/recall/F1; prefer recall for high‑risk detection. Use confusion matrix to interpret misclassifications and check calibration if probabilities guide decisions.

### Deployment
- App served with Streamlit; `requirements.txt` and `runtime.txt` control the cloud build. Model loaded with `joblib.load('final_model.pkl')`.

### Code & Implementation
- `app.py` aligns input columns to `model.feature_names_` to avoid schema mismatch and flattens model outputs to scalars before label lookup. Validate numeric ranges and handle zero/division for ratios.

### Interpretability & Ethics
- Provide feature importances and (optionally) SHAP explanations for individual predictions. Check model fairness across demographics and document limitations before clinical use.

### Troubleshooting
- Missing package error: add the package/version to `requirements.txt` and re-deploy. Schema mismatch: ensure input DataFrame columns match `model.feature_names_`.

## Demo Checklist
- [ ] Activate virtual environment and install requirements.
- [ ] Start Streamlit locally: `streamlit run app.py`.
- [ ] Verify UI inputs accept expected ranges (Age, Glucose, Insulin, Triglycerides, HDL, Uric Acid).
- [ ] Click Predict and confirm probabilities and interpretation render.
- [ ] (Optional) Upload a test CSV to the evaluation section (if present) and inspect confusion matrix & metrics.
- [ ] Show `requirements.txt` and `runtime.txt` to explain reproducibility/deployment.

## Next Steps (suggested)
- Add SHAP explanations, logging for prediction drift, and a README paragraph describing the dataset provenance and ethics review status.

---
For the viva, be ready to explain modeling choices succinctly and to demo the app flow end-to-end.
