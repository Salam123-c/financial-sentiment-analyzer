# 📈 Institutional Quant NLP Sentiment & Sector-Neutral Alpha Terminal

An institutional-grade Quantitative NLP and Event-Driven Alpha Terminal built with **Streamlit**, **FastAPI**, **PyTorch**, **ProsusAI/FinBERT**, **statsmodels**, and **scikit-learn**.

## 🚀 Key Quantitative Innovations
1. **Executive Tone Divergence**: Quantifies management overconfidence by contrasting Executive Remarks vs Analyst Q&A sentiment ($NSI_{Exec} - NSI_{QA}$).
2. **Linguistic Hedging Spread**: Detects corporate uncertainty using domain-specific hedging word distributions (`may`, `could`, `might` vs `will`, `confident`).
3. **Sector-Neutral Alpha (CAR)**: Evaluates 5-day post-earnings cumulative abnormal returns benchmarked against specific Sector ETFs (XLK, XLF, XLV, XLE, XLP, XLY, XLI, XLC) and the S&P 500 (`^GSPC`).
4. **Econometric White HC1 Regression**: Multi-variable OLS predicting 5-day sector alpha with heteroskedasticity-consistent standard errors.
5. **Walk-Forward 5-Fold TimeSeriesSplit**: Cross-validated machine learning classification evaluated against **3 Industry Baselines** (Random Coin-Flip, Buy & Hold, Naive Sentiment).
6. **Net-of-Fee Equity Curve Simulation**: Real-world backtesting factoring in **10 bps institutional transaction friction**.
7. **Production FastAPI Endpoint**: Microservice API for real-time transcript inference and containerized Docker deployment.

## 🛠️ Quickstart

### 1. Web Terminal (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Enterprise API Microservice (FastAPI)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- **Health check**: `GET /health`
- **Predict alpha**: `POST /predict_alpha`

### 3. Docker Deployment
```bash
docker build -t quant-nlp-alpha .
docker run -p 8501:8501 -p 8000:8000 quant-nlp-alpha
```
