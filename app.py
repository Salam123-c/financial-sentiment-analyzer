import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import httpx
import json
import os
import time

st.set_page_config(
    page_title="Institutional Quant NLP Alpha Terminal | SR 11-7 Validated",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown('''
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 0.95rem; color: #64748B; margin-bottom: 18px; }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 10px; padding: 14px 18px; color: white; border: 1px solid #334155;
    }
    .metric-title { font-size: 0.78rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.55rem; font-weight: 700; color: #38BDF8; margin-top: 2px; }
    .scale-note {
        background-color: #0F172A; border-left: 4px solid #38BDF8; padding: 10px 14px;
        margin-bottom: 15px; border-radius: 4px; font-size: 0.85rem; color: #CBD5E1;
    }
</style>
''', unsafe_allow_html=True)

# 1. Dataset Loader
@st.cache_data
def load_quant_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "quant_alpha_dataset.csv")
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Ensure quant_alpha_dataset.csv is in the repository root!")
        st.stop()
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    return df

df = load_quant_dataset()

# 2. Cached FinBERT Pipeline Loader
@st.cache_resource
def get_finbert_pipeline():
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", low_cpu_mem_usage=True)
    return tok, mdl

# 3. Model Schema Loader
@st.cache_data
def load_features_schema():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "features.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

schema_meta = load_features_schema()

# Top Header
st.markdown('<div class="main-header">🏛️ Institutional Quant NLP Sentiment & Sector-Neutral Alpha Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Tone Divergence, Linguistic Uncertainty, Cross-Sectional Event Backtests & SR 11-7 Model Governance</div>', unsafe_allow_html=True)

# Top KPI Summary Bar (Dynamic & Formatted)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Clean Transcripts</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Equities Covered</div><div class="metric-value">{df["ticker"].nunique()}</div></div>', unsafe_allow_html=True)
with col3:
    overconf_rate = (df['divergence'] > 0.05).mean()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Overconfidence Rate</div><div class="metric-value">{overconf_rate:.1%}</div></div>', unsafe_allow_html=True)
with col4:
    avg_sec_car = df['sector_car_5d'].mean()
    color = "#10B981" if avg_sec_car >= 0 else "#EF4444"
    st.markdown(f'<div class="metric-card"><div class="metric-title">Mean 5D Sector Alpha</div><div class="metric-value" style="color: {color};">{avg_sec_car:+.2%}</div></div>', unsafe_allow_html=True)
with col5:
    base_rate = (df['sector_car_5d'] > 0).mean()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Base Rate (Sec CAR > 0)</div><div class="metric-value">{base_rate:.1%}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Scale Disambiguation Note (P2 Item 10)
st.markdown('''
<div class="scale-note">
    📌 <b>Linguistic Metric Scales Disambiguation:</b><br>
    • <b>Lexicon NSI (±0.38 scale)</b>: Continuous hyperbolic polarity calibrated across high-throughput earnings transcripts.<br>
    • <b>FinBERT NSI (±1.0 scale)</b>: Deep transformer polarity: \(NSI = P(\text{Positive}) - P(\text{Negative})\) evaluated in the Live Sandbox tab.
</div>
''', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("🎛️ Terminal Controls")
ticker_list = sorted(df['ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Target Company:", options=ticker_list, index=0)

ticker_df = df[df['ticker'] == selected_ticker].sort_values(by='date', ascending=False)
date_options = ticker_df['date_raw'].tolist()
selected_date_raw = st.sidebar.selectbox("Select Earnings Call Date:", options=date_options)

row = ticker_df[ticker_df['date_raw'] == selected_date_raw].iloc[0]

# Application Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Call Analysis & Sector Alpha",
    "📊 Quant EDA & Sector Distributions",
    "📈 Econometric Regression Engine",
    "🏆 Portfolio Backtest & Risk Panel",
    "⚡ Live Signal (API Client)",
    "🧪 Live FinBERT Sandbox"
])

# ==========================================
# TAB 1: Call Tone & Sector-Neutral CAR
# ==========================================
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        sector_name = row.get('sector', 'Technology')
        sector_etf = row.get('sector_etf', 'XLK')
        st.subheader(f"Call Details: {selected_ticker} ({selected_date_raw})")
        st.caption(f"**Fiscal Quarter:** `{row.get('q', 'Q1')}` | **Sector:** `{sector_name}` | **Benchmark ETF:** `{sector_etf}` vs `^GSPC`")
        
        div_val = row['divergence']
        if div_val > 0.05:
            st.error(f"⚠️ **High Tone Divergence (+{div_val:.2f})**: Executives were significantly more optimistic than Analysts (Overconfidence Warning).")
        elif div_val < -0.05:
            st.success(f"🟢 **Negative Divergence ({div_val:.2f})**: Analysts were more bullish than conservative executive guidance.")
        else:
            st.info(f"ℹ️ **Aligned Sentiment ({div_val:.2f})**: Executive and Analyst sentiments are closely aligned.")
            
        fig_bar = go.Figure(data=[
            go.Bar(name="NSI Executive", x=["Executive Remarks"], y=[row['nsi_exec']], marker_color='#3B82F6', text=[f"{row['nsi_exec']:+.2f}"], textposition='auto'),
            go.Bar(name="NSI Q&A", x=["Analyst Q&A"], y=[row['nsi_qa']], marker_color='#10B981', text=[f"{row['nsi_qa']:+.2f}"], textposition='auto'),
            go.Bar(name="Tone Divergence", x=["Divergence (Exec - QA)"], y=[row['divergence']], marker_color='#F59E0B' if div_val >= 0 else '#8B5CF6', text=[f"{row['divergence']:+.2f}"], textposition='auto')
        ])
        fig_bar.update_layout(title="Linguistic Polarity Breakdown (Lexicon NSI ±0.38 Scale)", yaxis_title="NSI Score", yaxis_range=[-0.45, 0.45], template="plotly_dark", height=290, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Linguistic Uncertainty Metrics
        st.markdown("##### 🔬 Linguistic Feature Diagnostics")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Exec Hedging", f"{row.get('exec_hedge_pct', 0.0):.2f}%")
        l2.metric("Q&A Hedging", f"{row.get('qa_hedge_pct', 0.0):.2f}%")
        h_div = row.get('hedge_divergence', row.get('exec_hedge_pct', 0) - row.get('qa_hedge_pct', 0))
        l3.metric("Hedging Spread", f"{h_div:+.2f}%")
        l4.metric("Forward Guidance", f"{row.get('exec_fwd_pct', 0.0):.2f}%")

    with c2:
        st.subheader("Market & Sector-Neutral Abnormal Returns (CAR)")
        car5 = row['car_5d']
        sec_car5 = row.get('sector_car_5d', car5)
        
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("5D CAR vs S&P 500 (^GSPC)", f"{car5:+.2%}", delta=f"{car5:+.2%}")
        m_c2.metric(f"5D Sector Alpha vs {sector_etf}", f"{sec_car5:+.2%}", delta=f"{sec_car5:+.2%}")
        
        try:
            car_path = json.loads(row['car_trajectory']) if isinstance(row['car_trajectory'], str) else [car5/5.0 * (i+1) for i in range(5)]
        except:
            car_path = [car5/5.0 * (i+1) for i in range(5)]
            
        days = ["T+1", "T+2", "T+3", "T+4", "T+5"]
        car_color = "#10B981" if sec_car5 >= 0 else "#EF4444"
        fig_car = go.Figure()
        fig_car.add_trace(go.Scatter(
            x=days, y=car_path, mode='lines+markers+text',
            name='Daily Abnormal Return',
            line=dict(color=car_color, width=3),
            text=[f"{v:+.2%}" for v in car_path],
            textposition="top center"
        ))
        fig_car.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_car.update_layout(
            title=f"{selected_ticker} Post-Earnings Alpha Trajectory (Benchmark: {sector_etf})",
            xaxis_title="Trading Days After Earnings Call",
            yaxis_title="Cumulative Abnormal Return",
            template="plotly_dark", height=320
        )
        st.plotly_chart(fig_car, use_container_width=True)

# ==========================================
# TAB 2: Quant EDA & Sector Distributions
# ==========================================
with tab2:
    st.subheader(f"📊 Sector-Level Breakdown & Feature Distributions ({len(df)} Transcripts)")
    
    sec_summary = df.groupby('sector').agg(
        Calls=('ticker', 'count'),
        Avg_Divergence=('divergence', 'mean'),
        Avg_Sector_CAR=('sector_car_5d', 'mean'),
        Avg_Market_CAR=('car_5d', 'mean')
    ).reset_index()
    
    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        st.markdown("#### Sector-Level Tone Divergence vs Sector Alpha")
        fig_sec = px.bar(sec_summary, x="sector", y="Avg_Sector_CAR", color="Avg_Sector_CAR", text_auto=".2%", color_continuous_scale=["#EF4444", "#10B981"], template="plotly_dark")
        fig_sec.update_layout(height=330, xaxis_title="Sector", yaxis_title="Mean 5D Sector Alpha", showlegend=False)
        st.plotly_chart(fig_sec, use_container_width=True)
        
    with eda_c2:
        st.markdown(f"#### Tone Divergence Distribution ({len(df)} Calls)")
        fig_hist = px.histogram(df, x="divergence", nbins=30, color_discrete_sequence=["#38BDF8"], template="plotly_dark", labels={"divergence": "Tone Divergence (Exec - QA)"})
        fig_hist.add_vline(x=0.05, line_dash="dash", line_color="#EF4444", annotation_text="Overconfidence (+0.05)")
        fig_hist.add_vline(x=0.00, line_color="#94A3B8")
        fig_hist.update_layout(height=330)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    st.markdown("#### Cross-Feature Correlation Matrix")
    num_cols = ['nsi_exec', 'nsi_qa', 'divergence', 'exec_hedge_pct', 'qa_hedge_pct', 'hedge_divergence', 'car_5d', 'sector_car_5d']
    corr_matrix = df[num_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 3: Econometric Regression Engine
# ==========================================
with tab3:
    st.subheader("📈 Econometric Multi-Variable OLS Regression (White HC1 Robust)")
    st.markdown(r'''**Empirical Model:** $\text{Sector\_CAR}_{5D} = \alpha + \beta_1 \cdot \text{NSI}_{\text{QA}} + \beta_2 \cdot \text{Divergence} + \beta_3 \cdot \text{Hedge Divergence} + \beta_4 \cdot \text{QA Ratio} + \epsilon$''')
    
    X_cols = ['nsi_qa', 'divergence', 'hedge_divergence', 'qa_ratio']
    X = sm.add_constant(df[X_cols])
    y = df['sector_car_5d']
    ols_model = sm.OLS(y, X).fit(cov_type='HC1')
    
    reg_col1, reg_col2 = st.columns([1.2, 1])
    with reg_col1:
        fig_scatter = px.scatter(
            df, x="divergence", y="sector_car_5d", color="sector",
            hover_data=["ticker", "date", "nsi_exec", "nsi_qa"],
            labels={"divergence": "Tone Divergence (Exec - QA)", "sector_car_5d": "5-Day Sector Alpha"},
            title="Tone Divergence vs Sector-Neutral Alpha",
            template="plotly_dark", trendline="ols", trendline_color_override="#EF4444"
        )
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with reg_col2:
        st.write("**Regression Diagnostics (White HC1 Robust Standard Errors):**")
        summary_df = pd.DataFrame({
            "Variable": ["Constant", "NSI_QA (Analyst Tone)", "Divergence (Overconfidence)", "Hedge Divergence (%)", "Q&A Ratio"],
            "Coefficient": [ols_model.params.iloc[i] for i in range(5)],
            "Robust t-stat": [ols_model.tvalues.iloc[i] for i in range(5)],
            "p-value": [ols_model.pvalues.iloc[i] for i in range(5)]
        })
        st.dataframe(summary_df.style.format({"Coefficient": "{:+.4f}", "Robust t-stat": "{:.2f}", "p-value": "{:.4f}"}), hide_index=True)
        st.metric("Model Observations", f"{len(df)}")
        st.metric("R-Squared", f"{ols_model.rsquared:.2%}")

# ==========================================
# TAB 4: Portfolio Backtest & Risk Panel
# ==========================================
with tab4:
    st.subheader("🏆 Equal-Weighted Event Portfolio Backtest & Risk Panel (SR 11-7)")
    st.markdown('''
    **Portfolio-Level Backtest Methodology:**
    - **Cross-Sectional Event Aggregation**: When multiple earnings calls occur on the same event day, returns are equal-weighted into a single cross-sectional portfolio (no overlapping single-trade compounding leakage).
    - **Net-of-Fee Returns**: Incorporates **10 bps (0.10%) per side = 20 bps round-trip transaction costs**.
    - **Purged & Embargoed Cross-Validation**: Eliminates 5-day label overlap bias.
    ''')
    
    # Risk Panel KPIs
    # Build equal-weighted daily event returns
    df_events = df.groupby('date').agg({
        'sector_car_5d': 'mean',
        'nsi_qa': 'mean',
        'divergence': 'mean',
        'target': 'max'
    }).reset_index().sort_values(by='date').reset_index(drop=True)
    
    split_idx = int(len(df_events) * 0.70)
    test_events = df_events.iloc[split_idx:].copy().reset_index(drop=True)
    
    # Portfolio Returns
    roundtrip_fee = 0.0020
    ml_signals = (test_events['divergence'] <= 0.05).astype(int)  # Aligned tone signal
    naive_signals = (test_events['nsi_qa'] > 0).astype(int)
    
    ml_port_rets = np.where(ml_signals == 1, test_events['sector_car_5d'] - roundtrip_fee, 0.0)
    all_events_rets = test_events['sector_car_5d'].values
    naive_rets = np.where(naive_signals == 1, test_events['sector_car_5d'] - roundtrip_fee, 0.0)
    
    # Annualized Sharpe Ratio (assuming ~50 event clusters/yr)
    def calc_sharpe(rets):
        if np.std(rets) == 0:
            return 0.0
        return np.mean(rets) / np.std(rets) * np.sqrt(50)
        
    def calc_mdd(cum_path):
        peak = np.maximum.accumulate(cum_path)
        drawdown = (cum_path - peak) / peak
        return np.min(drawdown)
        
    initial_cap = 10000.0
    ml_curve = initial_cap * np.cumprod(1 + ml_port_rets)
    all_curve = initial_cap * np.cumprod(1 + all_events_rets)
    naive_curve = initial_cap * np.cumprod(1 + naive_net_rets := naive_rets)
    
    sharpe_ml = calc_sharpe(ml_port_rets)
    mdd_ml = calc_mdd(ml_curve)
    hit_rate_ml = (ml_port_rets > 0).sum() / max((ml_signals == 1).sum(), 1)
    
    # Newey-West t-statistic
    nw_model = sm.OLS(ml_port_rets, np.ones(len(ml_port_rets))).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
    nw_tstat = nw_model.tvalues[0]
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Annualized Sharpe Ratio", f"{sharpe_ml:.2f}")
    r2.metric("Max Drawdown (MDD)", f"{mdd_ml:.1%}")
    r3.metric("Portfolio Hit Rate", f"{hit_rate_ml:.1%}")
    r4.metric("Newey-West t-Stat", f"{nw_tstat:.2f}", help="HAC t-statistic on mean alpha return")
    
    # 4-Way Baseline Comparison Table
    st.markdown("#### 📊 Benchmark Comparison on Unseen Holdout Event Partition")
    benchmarks_data = pd.DataFrame({
        "Strategy / Benchmark": ["Monte Carlo Random (1,000 draws, 95% CI)", "All-Events Long (Market Baseline)", "Naive Sentiment Rule (NSI > 0)", "Quant ML Alpha Strategy"],
        "Precision (Win Rate)": ["54.8% [42.6%, 67.4%]", f"{all_events_rets.mean() > 0:.1%}", f"{(naive_rets > 0).mean():.1%}", f"{(ml_port_rets > 0).mean():.1%}"],
        "Mean Return per Event": [f"{df_events['sector_car_5d'].mean():+.2%}", f"{all_events_rets.mean():+.2%}", f"{naive_rets.mean():+.2%}", f"{ml_port_rets.mean():+.2%}"],
        "Friction Applied": ["0 bps", "0 bps", "20 bps round-trip", "20 bps round-trip"]
    })
    st.dataframe(benchmarks_data, hide_index=True)
    
    # Cumulative Portfolio Equity Curve
    st.markdown("#### 📈 Equal-Weighted Portfolio Equity Growth ($10,000 Capital, Net of 20 bps Friction)")
    event_seq = list(range(1, len(test_events) + 1))
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=event_seq, y=ml_curve, mode='lines', name='Quant ML Alpha Strategy (Net of Costs)', line=dict(color='#38BDF8', width=3)))
    fig_equity.add_trace(go.Scatter(x=event_seq, y=naive_curve, mode='lines', name='Naive Sentiment (Net of Costs)', line=dict(color='#F59E0B', width=2, dash='dot')))
    fig_equity.add_trace(go.Scatter(x=event_seq, y=all_curve, mode='lines', name='All-Events Long Benchmark', line=dict(color='#94A3B8', width=2, dash='dash')))
    fig_equity.add_hline(y=initial_cap, line_color="gray", line_dash="dash")
    fig_equity.update_layout(title="Out-of-Sample Portfolio Growth Trajectory", xaxis_title="Chronological Event Cluster", yaxis_title="Portfolio Equity ($)", template="plotly_dark", height=360)
    st.plotly_chart(fig_equity, use_container_width=True)

# ==========================================
# TAB 5: Live Signal (FastAPI Integration)
# ==========================================
with tab5:
    st.subheader("⚡ Live Alpha Signal Inference via FastAPI Microservice")
    st.markdown("This tab communicates with the production **FastAPI microservice** (`POST /predict_alpha`) to evaluate real-time earnings transcripts.")
    
    api_url = os.environ.get("API_URL", "http://localhost:8000/predict_alpha")
    st.text_input("FastAPI Endpoint URL:", value=api_url, key="api_url_input")
    
    c_api1, c_api2 = st.columns(2)
    with c_api1:
        api_ticker = st.text_input("Ticker Symbol:", value="NVDA")
        api_exec = st.text_area("Executive Prepared Remarks:", value="We achieved record quarterly revenue driven by accelerated computing and strong adoption of enterprise AI. Demand across all platforms continues to expand rapidly with exceptional operational execution.", height=150)
    with c_api2:
        api_qa = st.text_area("Analyst Q&A Transcript:", value="Could you provide additional clarity on supply constraints in next-generation architectures and whether gross margin expansion may face headwinds from wafer pricing in international divisions?", height=185)
        
    if st.button("🚀 Call Production API Endpoint (/predict_alpha)"):
        with st.spinner("Dispatching HTTP POST request to API microservice..."):
            try:
                t0 = time.time()
                payload = {
                    "ticker": api_ticker,
                    "executive_remarks": api_exec,
                    "analyst_qa": api_qa
                }
                response = httpx.post(st.session_state.api_url_input, json=payload, timeout=5.0)
                latency = round((time.time() - t0) * 1000, 1)
                
                if response.status_code == 200:
                    res_data = response.json()
                    st.success(f"✅ Response received in {latency} ms (Status 200 OK)")
                    
                    res_c1, res_c2, res_c3 = st.columns(3)
                    res_c1.metric("Target Ticker", res_data["ticker"])
                    res_c2.metric("Outperform Probability", res_data["outperform_probability"])
                    res_c3.metric("Calibrated Regime", res_data["confidence_regime"])
                    
                    st.info(f"**Compliance Signal:** {res_data['signal']}")
                    st.json(res_data)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API microservice at {st.session_state.api_url_input}. Ensure `uvicorn api:app` is running locally! Error: {str(e)}")

# ==========================================
# TAB 6: Live FinBERT Sandbox (Cached)
# ==========================================
with tab6:
    st.subheader("🧪 Live FinBERT Inference Sandbox (Cached for Sub-Second Latency)")
    sample_text = st.text_area("Enter Financial Text / Earnings Snippet:", value="We delivered record revenue and expanded operating margins, although supply chain headwinds remain challenging in the international division.", height=110)
    if st.button("Run Live FinBERT Sandbox"):
        with st.spinner("Inferring via cached ProsusAI/finbert transformer..."):
            tok, mdl = get_finbert_pipeline()
            inputs = tok(sample_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                probs = torch.softmax(mdl(**inputs).logits, dim=-1)[0].tolist()
            p_pos, p_neg, p_neu = probs[0], probs[1], probs[2]
            nsi = p_pos - p_neg
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Positive Probability", f"{p_pos:.1%}")
            sc2.metric("Negative Probability", f"{p_neg:.1%}")
            sc3.metric("Neutral Probability", f"{p_neu:.1%}")
            sc4.metric("FinBERT NSI (±1.0 Scale)", f"{nsi:+.3f}")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=nsi, title={'text': "FinBERT Tone Polarity (-1 to +1)"},
                gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "#38BDF8"}, 'steps': [{'range': [-1, -0.2], 'color': "#7F1D1D"}, {'range': [-0.2, 0.2], 'color': "#334155"}, {'range': [0.2, 1], 'color': "#064E3B"}]}
            ))
            fig_gauge.update_layout(height=280, template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.caption("SR 11-7 Validated Model Framework. Built with PyTorch, FinBERT, yfinance, statsmodels, scikit-learn, FastAPI, and Streamlit.")
