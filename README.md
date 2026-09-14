# 📈 Institutional Quant NLP Sentiment & Alpha Terminal

An institutional-grade Quantitative Finance and Natural Language Processing (NLP) web terminal built with **Streamlit**, **PyTorch**, **ProsusAI/FinBERT**, **statsmodels**, and **scikit-learn**.

## 🚀 Key Features
1. **Executive Tone Divergence**: Quantifies management overconfidence by contrasting Executive Remarks vs Analyst Q&A sentiment ({Exec} - NSI_{QA}$).
2. **Cumulative Abnormal Return (CAR) Trajectory**: Tracks post-earnings 5-day stock alpha against the S&P 500 benchmark (^GSPC).
3. **Econometric Regression Engine**: OLS regression predicting 5-day CAR from tone divergence and analyst sentiment with diagnostic p-values and R².
4. **Machine Learning Directional Alpha Classifier**: Random Forest Classifier evaluating precision, recall, F1-score, and confusion matrix for actionable trading signals.
5. **Interactive FinBERT Sandbox**: Live financial text inference with domain-adapted tone extraction.

## 🛠️ Installation & Local Run
`ash
# Clone the repository
git clone https://github.com/your-username/financial-sentiment-analyzer.git
cd financial-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run Streamlit App
streamlit run app.py
`

## 📦 Deployment on Streamlit Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch (main), and main file (pp.py).
4. Click **Deploy!**
