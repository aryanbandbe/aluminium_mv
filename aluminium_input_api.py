import os
import joblib
import requests
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ----------------------------------------------------
# Initialize FastAPI app
# ----------------------------------------------------
app = FastAPI(
    title="Aluminium Input Imputation API",
    description="Predict missing input parameters for Aluminium LCA using trained XGBoost model.",
    version="1.0"
)

# ----------------------------------------------------
# GitHub release URLs for joblib files
# ----------------------------------------------------
MODEL_URL = "https://github.com/aryanbandbe/aluminium_mv/releases/download/model-files-v1/xgb_input_imputer.joblib"
ENCODER_URL = "https://github.com/aryanbandbe/aluminium_mv/releases/download/model-files-v1/categorical_encoder.joblib"


# ----------------------------------------------------
# Utility function: Download file if missing
# ----------------------------------------------------
def download_if_missing(url, filename):
    """Downloads a file from a URL if it's not already present locally."""
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Download complete: {filename}")
    else:
        print(f"⚡ {filename} already exists locally.")


# ----------------------------------------------------
# Ensure models are available before startup
# ----------------------------------------------------
download_if_missing(MODEL_URL, "xgb_input_imputer.joblib")
download_if_missing(ENCODER_URL, "categorical_encoder.joblib")

# ----------------------------------------------------
# Load models
# ----------------------------------------------------
print("🔄 Loading models...")
model = joblib.load("xgb_input_imputer.joblib")
encoder = joblib.load("categorical_encoder.joblib")
print("✅ Models loaded successfully.")


# ----------------------------------------------------
# Define input schema
# ----------------------------------------------------
class AluminiumInput(BaseModel):
    metal: str
    route: str
    stage: str
    region: str


# ----------------------------------------------------
# Root endpoint (health check)
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "✅ Aluminium Input Imputation API is running successfully!"}


# ----------------------------------------------------
# Prediction endpoint
# ----------------------------------------------------
@app.post("/predict/aluminium/inputs")
async def predict_aluminium_inputs(data: AluminiumInput):
    try:
        # Step 1: Convert input JSON to DataFrame
        df = pd.DataFrame([data.dict()])

        # Step 2: Encode categorical data
        encoded = encoder.transform(df)

        # Step 3: Handle feature shape mismatch (align with training columns)
        if hasattr(encoder, "get_feature_names_out"):
            expected_cols = encoder.get_feature_names_out()
        else:
            expected_cols = encoder.get_feature_names()

        encoded_df = pd.DataFrame(encoded, columns=expected_cols)

        # Add missing columns if any are missing
        for col in expected_cols:
            if col not in encoded_df.columns:
                encoded_df[col] = 0

        encoded_df = encoded_df.reindex(columns=expected_cols, fill_value=0)

        # Step 4: Predict missing inputs
        preds = model.predict(encoded_df)

        # Step 5: Create readable output
        pred_df = pd.DataFrame(preds, columns=[
            "electricity_MJ", "natural_gas_MJ", "diesel_MJ",
            "heavy_oil_MJ", "coal_MJ", "bauxite_input_kg",
            "alumina_input_kg", "scrap_input_kg", "total_energy_MJ"
        ])

        # Step 6: Return response
        return JSONResponse(content={
            "success": True,
            "predictions": pred_df.to_dict(orient="records")[0]
        })

    except Exception as e:
        # Error handling
        return JSONResponse(content={"success": False, "error": str(e)})


# ----------------------------------------------------
# Local run configuration
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
