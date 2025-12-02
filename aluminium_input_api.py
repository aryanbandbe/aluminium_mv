from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import traceback

app = FastAPI(title="Aluminium Input Imputation API")

# Load trained model and encoder
model = joblib.load("xgb_input_imputer.joblib")
encoder = joblib.load("categorical_encoder.joblib")

@app.post("/predict/aluminium/inputs")
def predict_aluminium_inputs(payload: dict):
    try:
        print("📩 Incoming Payload:", payload)

        # Step 1. Convert input to DataFrame
        df = pd.DataFrame([payload])

        # Step 2. Encode categorical variables
        encoded = encoder.transform(df[["metal", "route", "stage", "region"]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

        # Step 3. Get model’s expected features
        model_features = model.estimators_[0].get_booster().feature_names

        # Step 4. Add all missing columns as zeros
        for col in model_features:
            if col not in encoded_df.columns:
                encoded_df[col] = 0

        # Step 5. Reorder columns to match training order
        encoded_df = encoded_df[model_features]

        print(f"✅ Final columns fed to model: {list(encoded_df.columns)}")

        # Step 6. Predict missing input parameters
        preds = model.predict(encoded_df)

        # Step 7. Prepare readable output
        cols = [
            "electricity_MJ", "natural_gas_MJ", "diesel_MJ",
            "heavy_oil_MJ", "coal_MJ", "bauxite_input_kg",
            "alumina_input_kg", "scrap_input_kg", "total_energy_MJ"
        ]

        predicted_inputs = pd.DataFrame(preds, columns=cols)
        result = predicted_inputs.to_dict(orient="records")[0]

        # Step 8. Convert all numpy data types to Python floats
        result = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in result.items()}

        print("✅ Prediction successful:", result)
        return {"success": True, "predicted_inputs": result}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}
