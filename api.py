from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import numpy as np
import os
from joblib import load

app = FastAPI(
    title="Institutional Quant NLP Alpha Inference API",
    version="2.0.0",
    description="Production-grade ML Microservice serving trained Random Forest Alpha Classifier on Earnings Transcripts"
)

# Load Serialized Machine Learning Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"model.joblib not found at {MODEL_PATH}")

clf_model = load(MODEL_PATH)

class TranscriptRequest(BaseModel):
    ticker: str
    executive_remarks: str
    analyst_qa: str

FIN_POSITIVE = set(['record', 'strong', 'growth', 'profit', 'expansion', 'exceeded', 'outperformed', 'favorable', 'momentum', 'gain', 'positive', 'solid', 'improved', 'resilient', 'tailwind', 'innovative', 'upside', 'superior', 'robust', 'accelerated', 'dividend', 'efficiency', 'achieved', 'beat', 'breakthrough', 'success', 'confidence', 'optimistic'])
FIN_NEGATIVE = set(['decline', 'loss', 'headwind', 'inflation', 'unfavorable', 'challenging', 'weakness', 'downturn', 'risk', 'pressure', 'slowdown', 'drop', 'slump', 'impairment', 'contraction', 'tariff', 'uncertainty', 'adversely', 'miss', 'deterioration', 'default', 'recession', 'restructuring', 'layoffs', 'litigation', 'volatility', 'shortfall'])
HEDGING_WORDS = set(['may', 'could', 'might', 'possibly', 'uncertain', 'approximate', 'contingent', 'tentative'])

def extract_features_from_text(exec_text: str, qa_text: str):
    exec_words = re.findall(r'\b[a-z]{3,}\b', exec_text.lower())
    qa_words = re.findall(r'\b[a-z]{3,}\b', qa_text.lower())
    
    n_exec = max(len(exec_words), 1)
    n_qa = max(len(qa_words), 1)
    
    # NSI calculation
    pos_e = sum(1 for w in exec_words if w in FIN_POSITIVE)
    neg_e = sum(1 for w in exec_words if w in FIN_NEGATIVE)
    tot_e = pos_e + neg_e
    diff_e = (pos_e - neg_e) / np.sqrt(tot_e + 1.0) if tot_e > 0 else 0.0
    nsi_exec = round(float(np.tanh(diff_e / 4.0) * 0.38), 4)
    
    pos_q = sum(1 for w in qa_words if w in FIN_POSITIVE)
    neg_q = sum(1 for w in qa_words if w in FIN_NEGATIVE)
    tot_q = pos_q + neg_q
    diff_q = (pos_q - neg_q) / np.sqrt(tot_q + 1.0) if tot_q > 0 else 0.0
    nsi_qa = round(float(np.tanh(diff_q / 4.0) * 0.38), 4)
    
    divergence = round(nsi_exec - nsi_qa, 4)
    qa_ratio = round(n_qa / n_exec, 2)
    
    # Hedging
    exec_hedge_pct = round(sum(1 for w in exec_words if w in HEDGING_WORDS) / n_exec * 100, 2)
    qa_hedge_pct = round(sum(1 for w in qa_words if w in HEDGING_WORDS) / n_qa * 100, 2)
    hedge_div = round(exec_hedge_pct - qa_hedge_pct, 2)
    is_overconfident = 1 if divergence > 0.05 else 0
    
    feature_vector = np.array([[nsi_exec, nsi_qa, divergence, qa_ratio, exec_hedge_pct, hedge_div, is_overconfident]])
    return feature_vector, {
        'nsi_exec': nsi_exec,
        'nsi_qa': nsi_qa,
        'divergence': divergence,
        'qa_ratio': qa_ratio,
        'exec_hedge_pct': exec_hedge_pct,
        'qa_hedge_pct': qa_hedge_pct,
        'hedge_divergence': hedge_div,
        'is_overconfident': is_overconfident
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Institutional Quant NLP Alpha ML Engine",
        "version": "2.0.0",
        "model_loaded": True
    }

@app.post("/predict_alpha")
def predict_alpha(req: TranscriptRequest):
    if len(req.executive_remarks) < 50 or len(req.analyst_qa) < 50:
        raise HTTPException(status_code=400, detail="Executive Remarks and Analyst Q&A must contain valid financial text.")
        
    feat_vector, metrics = extract_features_from_text(req.executive_remarks, req.analyst_qa)
    
    # Run Inference on trained Random Forest Model
    pred_class = int(clf_model.predict(feat_vector)[0])
    prob_outperform = float(clf_model.predict_proba(feat_vector)[0, 1])
    
    # Confidence Calibration
    if prob_outperform >= 0.70:
        confidence = "HIGH_CONFIDENCE_BUY"
        action = "STRONG BUY (Alpha Outperformance Expected)"
    elif prob_outperform >= 0.50:
        confidence = "MODERATE_BUY"
        action = "MODERATE BUY"
    elif prob_outperform <= 0.30:
        confidence = "HIGH_CONFIDENCE_AVOID"
        action = "AVOID / SHORT (Underperformance Expected)"
    else:
        confidence = "MODERATE_AVOID"
        action = "AVOID / HOLD"
        
    return {
        "ticker": req.ticker.upper(),
        "model": "RandomForestAlphaClassifier",
        "signal": action,
        "prediction_class": pred_class,
        "outperform_probability": f"{prob_outperform:.1%}",
        "confidence_regime": confidence,
        "linguistic_features": metrics
    }
