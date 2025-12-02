import os
import requests
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# -----------------------------------------------
# URLs to your model files in Firebase Storage
# -----------------------------------------------
MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/YOUR_BUCKET/o/xgb_input_imputer.joblib?alt=media"
ENCODER_URL = "https://firebasestorage.googleapis.com/v0/b/YOUR_BUCKET/o/categorical_encoder.joblib?alt=media"

def ensure_model_files():
    """Download model files if not present locally (Render ephemeral storage)."""
    files = {
        "xgb_input_imputer.joblib": MODEL_URL,
        "categorical_encoder.joblib": ENCODER_URL
    }
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"📥 Downloading {filename} ...")
            r = requests.get(url)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)
                print(f"✅ {filename} downloaded successfully.")
            else:
                raise RuntimeError(f"Failed to download {filename} (HTTP {r.status_code})")

# Ensure models are available before app starts
ensure_model_files()

# Now safely load them
model = joblib.load("xgb_input_imputer.joblib")
encoder = joblib.load("categorical_encoder.joblib")

# ------------------------------------------------
# FastAPI setup (as before)
# ------------------------------------------------
app = FastAPI(title="Aluminium Input Imputation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example endpoint (unchanged)
@app.post("/predict/aluminium/inputs")
def predict_aluminium_inputs(data: dict):
    try:
        df = pd.DataFrame([data])
        encoded_df = encoder.transform(df)
        preds = model.predict(encoded_df)
        cols = [
            "electricity_MJ", "natural_gas_MJ", "diesel_MJ",
            "heavy_oil_MJ", "coal_MJ", "bauxite_input_kg",
            "alumina_input_kg", "scrap_input_kg", "total_energy_MJ"
        ]
        result = dict(zip(cols, preds[0].tolist()))
        return {"success": True, "predicted_inputs": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
