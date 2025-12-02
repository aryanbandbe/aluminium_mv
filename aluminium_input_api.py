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
app = FastAPI(title="Aluminium Input Imputation API",
              description="Predict missing input parameters for Aluminium LCA using trained XGBoost model.",
              version="1.0")

# ----------------------------------------------------
# GitHub release URLs for joblib files
# ----------------------------------------------------
MODEL_URL = "https://github.com/aryanbandbe/aluminium_mv/releases/download/model-files-v1/xgb_input_imputer.joblib"
ENCODER_URL = "https://github.com/aryanbandbe/aluminium_mv/releases/download/model-files-v1/categorical_encoder.joblib"


# ----------------------------------------------------
# Utility function to auto-download model files
# ----------------------------------------------------
def download_if_missing(url, filename):
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
# Ensure models are available
# ----------------------------------------------------
download_if_missing(MODEL_URL, "xgb_input_imputer.joblib")
download_if_missing(ENCODER_URL, "categorical_encoder.joblib")

# ----------------------------------------------------
# Load models once at startup
# ----------------------------------------------------
print("🔄 Loading models...")
model = joblib.load("xgb_input_imputer.joblib")
encoder = joblib.load("categorical_encoder.joblib")
print("✅ Models loaded successfully.")


# ----------------------------------------------------
# Define input schema using Pydantic
# ----------------------------------------------------
class AluminiumInput(BaseModel):
    metal: str
    route: str
    stage: str
    region: str


# ----------------------------------------------------
# Root endpoint (for testing Render deployment)
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
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Encode categorical variables
        encoded = encoder.transform(df)

        # Predict missing values
        preds = model.predict(encoded)

        # Convert predictions into readable dictionary
        pred_df = pd.DataFrame(preds, columns=[
            "electricity_MJ", "natural_gas_MJ", "diesel_MJ",
            "heavy_oil_MJ", "coal_MJ", "bauxite_input_kg",
            "alumina_input_kg", "scrap_input_kg", "total_energy_MJ"
        ])

        # Return as JSON
        return JSONResponse(content={
            "success": True,
            "predictions": pred_df.to_dict(orient="records")[0]
        })

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})


# ----------------------------------------------------
# Run locally (optional)
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
