import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = "this is my secret API"
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML Model paths
MODEL_PATH = "stock_model.joblib"
SCALER_PATH = "scaler.joblib"
LE_TARGET_PATH = "le_target.joblib"

def train_model():
    logger.info("Starting model training...")
    try:
        df = pd.read_excel("Stock data from market.xlsx", sheet_name=0)
        
        # Clean numeric columns
        def clean_numeric(col):
            col = col.astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.replace('-', 'NaN', regex=False)
            col = pd.to_numeric(col, errors='coerce')
            def convert_suffix(val):
                if pd.isna(val):
                    return 0
                val_str = str(val)
                multipliers = {'B': 1e9, 'M': 1e6, 'K': 1e3}
                for suffix, mult in multipliers.items():
                    if val_str.endswith(suffix):
                        return float(val_str[:-1]) * mult
                return float(val)
            col = col.apply(convert_suffix)
            return col.fillna(0)
        
        numeric_cols = ['Market Cap', 'Stock Price', 'Volume', 'PE Ratio', 'Enterprise Value', 'Dollar Volume', 'Employees', 'Revenue', 'Revenue Growth', 'Gross Profit', 'Net Income', 'NetInc Growth', 'Liabilities', 'Gross Margin', 'Oper. Margin', 'Profit Margin', 'R&D', 'R&D / Rev', 'ROIC', 'Total Cash', 'Assets', 'Empl. Growth', 'Price Target', 'Shares Insiders', 'Op. Income', 'GP Growth', 'OpInc Growth', 'Equity', 'Working Capital', 'Net WC']
        for col in numeric_cols:
            df[col] = clean_numeric(df[col])
        
        # Encode categorical
        categorical_cols = ['Industry', 'Sector', 'Exchange']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Target
        target = 'Analyst Rating'
        df[target] = df[target].replace('-', np.nan).fillna('Hold')
        le_target = LabelEncoder()
        df[target] = le_target.fit_transform(df[target])
        
        # Features
        X = df.drop([target, 'Symbol', 'Company Name'], axis=1)
        y = df[target]
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        
        # Save
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le_target, LE_TARGET_PATH)
        
        logger.info("Model trained and saved successfully.")
        return model, scaler, le_target
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

# Load or train
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_target = joblib.load(LE_TARGET_PATH)
    logger.info("Loaded existing model.")
else:
    model, scaler, le_target = train_model()

class ChatRequest(BaseModel):
    message: str
    language: str

class AnalysisRequest(BaseModel):
    company_name: str
    pe_ratio: float
    revenue_growth: float
    netinc_growth: float
    gross_margin: float
    roic: float
    profit_margin: float

@app.get("/health")
def health_check():
    return {"status": "Backend is running"}

@app.get("/models")
def list_models():
    try:
        models = genai.list_models()
        return {"models": [model.name for model in models if 'generateContent' in model.supported_generation_methods]}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return {"error": str(e)}

@app.post("/chat")
def chat(request: ChatRequest = Body(...)):
    try:
        if request.language == 'english':
            prompt = request.message
            response_language = "Respond in English."
        else:
            prompt = f"User message in Banglish: {request.message}. Interpret as Bengali and respond in Bengali script."
            response_language = ""
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(f"{prompt} {response_language}")
        logger.info("Chat request processed successfully.")
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {"response": f"Error: {str(e)}. Check /models endpoint for available models."}

@app.post("/analyze")
def analyze(request: AnalysisRequest = Body(...)):
    try:
        features = np.zeros(33)
        # Updated indices based on dataset order
        feature_columns = ['Market Cap', 'Stock Price', 'Volume', 'PE Ratio', 'Enterprise Value', 'Dollar Volume', 'Employees', 'Revenue', 'Revenue Growth', 'Gross Profit', 'Net Income', 'NetInc Growth', 'Liabilities', 'Gross Margin', 'Oper. Margin', 'Profit Margin', 'R&D', 'R&D / Rev', 'ROIC', 'Total Cash', 'Assets', 'Empl. Growth', 'Price Target', 'Shares Insiders', 'Op. Income', 'GP Growth', 'OpInc Growth', 'Equity', 'Working Capital', 'Net WC', 'Industry', 'Sector', 'Exchange']
        features[feature_columns.index('PE Ratio')] = request.pe_ratio
        features[feature_columns.index('Revenue Growth')] = request.revenue_growth
        features[feature_columns.index('NetInc Growth')] = request.netinc_growth
        features[feature_columns.index('Gross Margin')] = request.gross_margin
        features[feature_columns.index('ROIC')] = request.roic
        features[feature_columns.index('Profit Margin')] = request.profit_margin
        
        scaled_features = scaler.transform([features])
        pred = model.predict(scaled_features)[0]
        rating = le_target.inverse_transform([pred])[0]
        
        prompt = f"For {request.company_name} with Analyst Rating {rating}, provide 2 lines of technical stock market advice in English, focusing on buy/sell/hold reasoning."
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        advice = response.text
        
        logger.info(f"Analysis for {request.company_name} completed.")
        return {"rating": rating, "advice": advice}
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"rating": "", "advice": f"Error: {str(e)}. Check /models endpoint for available models."}

logger.info("Starting FastAPI server on http://localhost:8000")
