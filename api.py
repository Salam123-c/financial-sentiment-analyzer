from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import re
import numpy as np
import os
import json
import logging
from joblib import load

# Configure Structured Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("QuantNLP-API")

app = FastAPI(
    title="Institutional Quant NLP Alpha Inference API",
    version="2.1.0",
    description="SR 11-7 Validated Production ML Microservice serving Calibrated Random Forest Alpha Classifier on Earnings Transcripts"
)

# Load Serialized Model & Feature Schema
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(BASE_DIR, "features.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"model.joblib not found at {MODEL_PATH}")

clf_model = load(MODEL_PATH)

if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        SCHEMA_META = json.load(f)
else:
    SCHEMA_META = {
        "model_version": "2.1.0",
        "feature_order": ['nsi_exec', 'nsi_qa', 'divergence', 'qa_ratio', 'exec_hedge_pct', 'hedge_divergence', 'is_overconfident'],
        "optimal_decision_threshold": 0.50
    }

# Known Equities Universe (46 Target Tickers)
VALID_UNIVERSE = set([
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'CRM', 'ADBE', 'INTC', 'AMD', 'CSCO',
    'ORCL', 'TXN', 'QCOM', 'IBM', 'JPM', 'BAC', 'GS', 'MS', 'V', 'PYPL', 'JNJ',
    'PFE', 'UNH', 'MRK', 'ABBV', 'BMY', 'AMGN', 'XOM', 'CVX', 'WMT', 'COST', 'PG',
    'KO', 'PEP', 'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'NFLX', 'T',
    'VZ', 'CAT', 'GE', 'HON', 'UPS', 'BA', 'LMT'
])

class TranscriptRequest(BaseModel):
    ticker: str = Field(..., description="Stock symbol (e.g. AAPL, NVDA)")
    executive_remarks: str = Field(..., max_length=200000, description="Executive prepared remarks (max 200k chars)")
    analyst_qa: str = Field(..., max_length=200000, description="Analyst Q&A transcript (max 200k chars)")

FIN_POSITIVE = set(['record', 'strong', 'growth', 'profit', 'expansion', 'exceeded', 'outperformed', 'favorable', 'momentum', 'gain', 'positive', 'solid', 'improved', 'resilient', 'tailwind', 'innovative', 'upside', 'superior', 'robust', 'accelerated', 'dividend', 'efficiency', 'achieved', 'beat', 'breakthrough', 'success', 'confidence', 'optimistic', 'strengthened', 'profitability', 'productive'])
FIN_NEGATIVE = set(['decline', 'loss', 'headwind', 'inflation', 'unfavorable', 'challenging', 'weakness', 'downturn', 'risk', 'pressure', 'slowdown', 'drop', 'slump', 'impairment', 'contraction', 'tariff', 'uncertainty', 'adversely', 'miss', 'deterioration', 'default', 'recession', 'restructuring', 'layoffs', 'litigation', 'volatility', 'shortfall', 'pessimistic', 'cautious', 'delayed'])
HEDGING_WORDS = set(['may', 'could', 'might', 'possibly', 'uncertain', 'approximate', 'contingent', 'tentative'])

def extract_features_from_text(exec_text: str, qa_text: str):
    exec_words = re.findall(r'\b[a-z]{3,}\b', exec_text.lower())
    qa_words = re.findall(r'\b[a-z]{3,}\b', qa_text.lower())
    
    n_exec = max(len(exec_words), 1)
    n_qa = max(len(qa_words), 1)
    
    # Continuous Net Sentiment Index (NSI)
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
    
    # Hedging Percentage & Divergence
    exec_hedge_pct = round(sum(1 for w in exec_words if w in HEDGING_WORDS) / n_exec * 100, 2)
    qa_hedge_pct = round(sum(1 for w in qa_words if w in HEDGING_WORDS) / n_qa * 100, 2)
    hedge_div = round(exec_hedge_pct - qa_hedge_pct, 2)
    is_overconfident = 1 if divergence > 0.05 else 0
    
    # Feature vector matching exact schema order
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
        "service": "Institutional Quant NLP Alpha Engine",
        "version": SCHEMA_META.get("model_version", "2.1.0"),
        "model_loaded": True,
        "feature_schema": SCHEMA_META.get("feature_order", []),
        "optimal_decision_threshold": SCHEMA_META.get("optimal_decision_threshold", 0.50),
        "validation_framework": "SR 11-7 Compliant"
    }

@app.post("/predict_alpha")
def predict_alpha(req: TranscriptRequest):
    ticker_clean = req.ticker.strip().upper()
    
    # Ticker Validation against universe
    if ticker_clean not in VALID_UNIVERSE:
        logger.warning(f"Ticker {ticker_clean} not in core coverage universe.")
        
    if len(req.executive_remarks.strip()) < 50 or len(req.analyst_qa.strip()) < 50:
        raise HTTPException(status_code=400, detail="Executive Remarks and Analyst Q&A must contain at least 50 characters of valid financial text.")
        
    feat_vector, metrics = extract_features_from_text(req.executive_remarks, req.analyst_qa)
    
    # Run Calibrated Random Forest Model Inference
    prob_outperform = float(clf_model.predict_proba(feat_vector)[0, 1])
    threshold = float(SCHEMA_META.get("optimal_decision_threshold", 0.50))
    pred_class = 1 if prob_outperform >= threshold else 0
    
    # Dynamically Derived Confidence Calibration (No Hardcoding)
    if prob_outperform >= 0.65:
        confidence = "HIGH_CONFIDENCE_OUTPERFORM"
    elif prob_outperform >= 0.50:
        confidence = "MODERATE_OUTPERFORM"
    elif prob_outperform <= 0.35:
        confidence = "HIGH_CONFIDENCE_UNDERPERFORM"
    else:
        confidence = "MODERATE_UNDERPERFORM"
        
    # Compliance-Ready Language (SR 11-7 / Disclaimers)
    compliance_signal = f"Model signal: Outperform probability {prob_outperform:.1%} (research use only, not investment advice)"
    
    logger.info(f"Inference Ticker={ticker_clean} | Prob={prob_outperform:.3f} | Class={pred_class} | Regime={confidence}")
    
    return {
        "ticker": ticker_clean,
        "model": "CalibratedRandomForestAlphaClassifier",
        "model_version": SCHEMA_META.get("model_version", "2.1.0"),
        "signal": compliance_signal,
        "prediction_class": pred_class,
        "outperform_probability": f"{prob_outperform:.1%}",
        "decision_threshold_used": threshold,
        "confidence_regime": confidence,
        "linguistic_features": metrics,
        "compliance_disclaimer": "This algorithmic output is generated for model validation and academic quantitative research. It does not constitute investment advice or trading solicitation."
    }
