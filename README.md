# 🧠 Aluminium Input Imputation API

This project is a **FastAPI-based machine learning API** that predicts missing aluminium input parameters (like energy usage and material inputs) using an **XGBoost regression model**.

---

## 🚀 Features
- Predicts missing aluminium LCA input parameters
- Uses a trained XGBoost + MultiOutputRegressor model
- Deployable on Render or Railway
- Integrates easily with Firebase Web App for live predictions

---

## 🗂️ Project Structure
aluminium_mv/
│
├── aluminium_input_api.py # FastAPI app
├── render.yaml # Render deployment configuration
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files and folders
└── README.md # Project info



## ⚙️ Run Locally
```bash
pip install -r requirements.txt
uvicorn aluminium_input_api:app --reload
Visit: http://127.0.0.1:8000/docs
