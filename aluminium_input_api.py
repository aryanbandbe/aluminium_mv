import os
import joblib
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI(title="Aluminium Input Imputation API")

# ---- MODEL DOWNLOAD LINKS ----
MODEL_URL = "https://github.com/aryanbandbe/aluminium_mv/releases/download/model-files-v1/xgb_input_imputer.joblib"
ENCODER_URL = "https://github.com/aryanbandbe/aluminium_mv/releases/download/model-files-v1/categorical_encoder.joblib"

# ---- DOWNLOAD MODELS IF MISSING ----
def download_if_missing(url, filename):
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename} from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"⚡ {filename} already exists locally")

download_if_missing(MODEL_URL, "xgb_input_imputer.joblib")
download_if_missing(ENCODER_URL, "categorical_encoder.joblib")

# ---- LOAD MODELS ----
model = joblib.load("xgb_input_imputer.joblib")
encoder = joblib.load("categorical_encoder.joblib")

# ---- PREDICTION ROUTE ----
@app.post("/predict/aluminium/inputs")
async def predict_aluminium_inputs(request: Request):
    try:
        data = await request.json()
        df = pd.DataFrame([data])
        encoded = encoder.transform(df)
        preds = model.predict(encoded)
        pred_df = pd.DataFrame(preds, columns=[
            "electricity_MJ", "natural_gas_MJ", "diesel_MJ", "heavy_oil_MJ", "coal_MJ",
            "bauxite_input_kg", "alumina_input_kg", "scrap_input_kg", "total_energy_MJ"
        ])
        return JSONResponse(content={"success": True, "predictions": pred_df.to_dict(orient="records")[0]})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
