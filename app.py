# app.py
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ------------------------------------------------------------------
# Paths (relative to this file, so it works both locally and on Railway)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "encoder.pkl")
CAT_COLUMNS_FILE = os.path.join(BASE_DIR, "cat_columns.pkl")
FEATURE_COLUMNS_FILE = os.path.join(BASE_DIR, "feature_columns.pkl")

# ------------------------------------------------------------------
# Flask app
# index.html lives in templates/, which is Flask's default template
# folder, so render_template() finds it automatically.
# ------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------
# Load trained artifacts ONCE at startup (not per-request)
# ------------------------------------------------------------------
print("Loading trained model artifacts...")
model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
cat_columns = joblib.load(CAT_COLUMNS_FILE)
feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
print("Model artifacts loaded successfully.")

# Numeric columns the model was trained on (everything else is categorical)
NUMERIC_COLUMNS = [
    "Age",
    "Residence_Years",
    "Annual_Family_Income",
    "Monthly_Expenses",
    "Family_Size",
    "Disability_Percentage",
    "Land_Ownership_Acres",
]

# The HTML form does not collect "Previous_Beneficiary" even though the
# model was trained on it as a categorical column. We default it so a
# missing field from the form doesn't break prediction.
CATEGORICAL_DEFAULTS = {
    "Previous_Beneficiary": "No",
}


# ------------------------------------------------------------------
# Serve the frontend
# ------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------------
# Prediction endpoint used by index.html's fetch("/predict", ...)
# ------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No input data received."}), 400

        # ---- Build one row in the shape the model expects ----
        row = {}

        for col in NUMERIC_COLUMNS:
            value = data.get(col, 0)
            try:
                row[col] = float(value) if value != "" else 0.0
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid numeric value for '{col}'."}), 400

        for col in cat_columns:
            value = data.get(col, CATEGORICAL_DEFAULTS.get(col, "No"))
            row[col] = str(value) if value != "" else CATEGORICAL_DEFAULTS.get(col, "No")

        input_df = pd.DataFrame([row])

        # ---- One-hot encode categorical columns using the saved encoder ----
        encoded = encoder.transform(input_df[cat_columns])
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=encoder.get_feature_names_out(cat_columns),
            index=input_df.index,
        )

        numeric_df = input_df.drop(columns=cat_columns)
        final_input = pd.concat([numeric_df, encoded_df], axis=1)

        # ---- Align columns exactly with training-time feature order ----
        final_input = final_input.reindex(columns=feature_columns, fill_value=0)

        # ---- Predict ----
        pred = int(model.predict(final_input)[0])
        prob = model.predict_proba(final_input)[0]
        confidence = round(float(prob.max()) * 100, 2)

        if pred == 1:
            prediction = "Eligible"
            reason = "Applicant satisfies the eligibility criteria of the Sanjay Gandhi Niradhar Pension Yojana."
            recommendation = (
                "Proceed to submit the physical application at your district "
                "Social Justice office with original documents."
            )
        else:
            prediction = "Not Eligible"
            reasons = []

            if row["Age"] < 65:
                reasons.append("Applicant is below the minimum eligible age.")
            if row["Annual_Family_Income"] > 100000:
                reasons.append("Family income exceeds the scheme limit.")
            if row.get("BPL_Status") == "No":
                reasons.append("Applicant is not listed under the BPL category.")
            if row.get("Bank_Account") == "No":
                reasons.append("Bank account is not available.")
            if row.get("Income_Certificate") == "No":
                reasons.append("Income certificate is missing.")

            if not reasons:
                reasons.append("Applicant does not satisfy one or more eligibility conditions.")

            reason = "; ".join(reasons)
            recommendation = (
                "Please complete pending certificates (income/disability) and "
                "Aadhaar linkage, then re-check eligibility."
            )

        return jsonify(
            {
                "prediction": prediction,
                "confidence": confidence,
                "reason": reason,
                "recommendation": recommendation,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ------------------------------------------------------------------
# Health check (useful for Railway)
# ------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------
# Local dev entry point. On Railway, gunicorn (see Procfile) runs
# "app:app" directly, so this block is skipped there.
# ------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)