from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import numpy as np

app = FastAPI(
    title="Institutional Quant NLP Alpha Inference API",
    version="1.0.0",
    description="Enterprise API endpoint for evaluating Earnings Call Tone Divergence & 5-Day Alpha Signals"
)

class TranscriptRequest(BaseModel):
    ticker: str
    executive_remarks: str
    analyst_qa: str

FIN_POSITIVE = set(['record', 'strong', 'growth', 'profit', 'expansion', 'exceeded', 'outperformed', 'favorable', 'momentum', 'gain', 'positive', 'solid', 'improved', 'resilient', 'tailwind', 'innovative', 'upside', 'superior', 'robust', 'accelerated', 'dividend', 'efficiency', 'achieved', 'beat', 'breakthrough', 'success', 'confidence', 'optimistic'])
FIN_NEGATIVE = set(['decline', 'loss', 'headwind', 'inflation', 'unfavorable', 'challenging', 'weakness', 'downturn', 'risk', 'pressure', 'slowdown', 'drop', 'slump', 'impairment', 'contraction', 'tariff', 'uncertainty', 'adversely', 'miss', 'deterioration', 'default', 'recession', 'restructuring', 'layoffs', 'litigation', 'volatility', 'shortfall'])

def compute_nsi(text: str) -> float:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in FIN_POSITIVE)
    neg = sum(1 for w in words if w in FIN_NEGATIVE)
    tot = pos + neg
    if tot == 0:
        return 0.0
    raw_diff = (pos - neg) / np.sqrt(tot + 1.0)
    return round(float(np.tanh(raw_diff / 4.0) * 0.38), 4)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Quant NLP Alpha Engine", "version": "1.0.0"}

@app.post("/predict_alpha")
def predict_alpha(req: TranscriptRequest):
    if len(req.executive_remarks) < 50 or len(req.analyst_qa) < 50:
        raise HTTPException(status_code=400, detail="Executive and Q&A sections must contain valid transcript text.")
        
    nsi_exec = compute_nsi(req.executive_remarks)
    nsi_qa = compute_nsi(req.analyst_qa)
    divergence = round(nsi_exec - nsi_qa, 4)
    
    # Overconfidence flag
    is_overconfident = bool(divergence > 0.05)
    
    # Quantitative Alpha Signal Logic
    if is_overconfident:
        signal = "AVOID / SELL (High Overconfidence Risk)"
        predicted_car_5d = round(float(-0.35 * divergence + 0.10 * nsi_qa), 4)
    elif nsi_qa > 0.05 and divergence <= 0.0:
        signal = "STRONG BUY (Analyst Bullish Confirmation)"
        predicted_car_5d = round(float(0.02 + 0.15 * nsi_qa), 4)
    else:
        signal = "NEUTRAL / HOLD"
        predicted_car_5d = 0.0010
        
    return {
        "ticker": req.ticker.upper(),
        "nsi_executive": nsi_exec,
        "nsi_analyst_qa": nsi_qa,
        "tone_divergence": divergence,
        "is_overconfident": is_overconfident,
        "trading_signal": signal,
        "forecasted_5d_car": f"{predicted_car_5d:+.2%}"
    }
