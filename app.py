import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import json
import os

st.set_page_config(
    page_title="Institutional Quant NLP Alpha Terminal | Research Edition",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS
st.markdown('''
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 0.98rem; color: #64748B; margin-bottom: 20px; }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 10px; padding: 14px 18px; color: white; border: 1px solid #334155;
    }
    .metric-title { font-size: 0.80rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #38BDF8; margin-top: 2px; }
</style>
''', unsafe_allow_html=True)

# Dataset Loader
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "quant_alpha_dataset.csv")

if not os.path.exists(csv_path):
    st.error(f"Dataset not found at {csv_path}. Please make sure quant_alpha_dataset.csv is in the repository folder!")
    st.stop()

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

# Main Header
st.markdown('<div class="main-header">🏛️ Institutional Quant NLP Sentiment & Alpha Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Executive Tone Divergence, Linguistic Uncertainty, Multi-Horizon CAR & Out-of-Sample Backtesting Engine</div>', unsafe_allow_html=True)

# Top KPI Summary Bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Clean Transcripts</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Equities Covered</div><div class="metric-value">{df["ticker"].nunique()}</div></div>', unsafe_allow_html=True)
with col3:
    overconf_rate = (df['divergence'] > 0.05).mean()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Overconfidence Rate</div><div class="metric-value">{overconf_rate:.1%}</div></div>', unsafe_allow_html=True)
with col4:
    avg_car = df['car_5d'].mean()
    color = "#10B981" if avg_car >= 0 else "#EF4444"
    st.markdown(f'<div class="metric-card"><div class="metric-title">Mean 5D CAR vs S&P</div><div class="metric-value" style="color: {color};">{avg_car:+.2%}</div></div>', unsafe_allow_html=True)
with col5:
    outperform_rate = (df['car_5d'] > 0).mean()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Buy Signal Rate</div><div class="metric-value">{outperform_rate:.1%}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar Controls
st.sidebar.title("🎛️ Terminal Controls")
ticker_list = sorted(df['ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Target Company:", options=ticker_list, index=0)

ticker_df = df[df['ticker'] == selected_ticker].sort_values(by='date', ascending=False)
date_options = ticker_df['date_raw'].tolist() if 'date_raw' in ticker_df.columns else ticker_df['date'].dt.strftime('%Y-%m-%d').tolist()
selected_date_raw = st.sidebar.selectbox("Select Earnings Call Date:", options=date_options)

row = ticker_df[ticker_df['date_raw'] == selected_date_raw].iloc[0] if 'date_raw' in ticker_df.columns else ticker_df.iloc[0]

# Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Call Analysis & Linguistic Tone",
    "📊 Quant EDA & Regime Distributions",
    "📈 Econometric Regression Engine",
    "🏆 Out-of-Sample Backtest & Equity Curve",
    "🧪 Live FinBERT Sandbox"
])

# ==========================================
# TAB 1: Call Tone & Multi-Horizon CAR
# ==========================================
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader(f"Call Details: {selected_ticker} ({str(selected_date_raw)[:12]})")
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
        fig_bar.update_layout(title="Linguistic Polarity Breakdown (Net Sentiment Index)", yaxis_title="NSI [-0.40 to +0.40]", yaxis_range=[-0.45, 0.45], template="plotly_dark", height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Linguistic Uncertainty Metrics
        st.markdown("##### 🔬 Linguistic Feature Diagnostics")
        l1, l2, l3 = st.columns(3)
        l1.metric("Hedging / Uncertainty", f"{row.get('exec_hedge_pct', 0.0):.2f}%", help="Frequency of 'may', 'could', 'might'")
        l2.metric("Certainty Tone", f"{row.get('exec_cert_pct', 0.0):.2f}%", help="Frequency of 'will', 'definitely', 'confident'")
        l3.metric("Forward Guidance", f"{row.get('exec_fwd_pct', 0.0):.2f}%", help="Frequency of 'expect', 'anticipate', 'target'")

    with c2:
        st.subheader("Real Market Reaction (Multi-Horizon CAR vs S&P 500)")
        car1 = row.get('car_1d', row['car_5d'] * 0.3)
        car5 = row['car_5d']
        
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("T+1 Post-Call CAR", f"{car1:+.2%}", delta=f"{car1:+.2%}")
        m_c2.metric("T+5 Cumulative CAR", f"{car5:+.2%}", delta=f"{car5:+.2%}")
        
        try:
            car_path = json.loads(row['car_trajectory']) if isinstance(row['car_trajectory'], str) else [car5/5.0 * (i+1) for i in range(5)]
        except:
            car_path = [car5/5.0 * (i+1) for i in range(5)]
            
        days = ["T+1", "T+2", "T+3", "T+4", "T+5"]
        car_color = "#10B981" if car5 >= 0 else "#EF4444"
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
            title=f"{selected_ticker} Post-Earnings Alpha Trajectory (Benchmark: ^GSPC)",
            xaxis_title="Trading Days After Earnings Call",
            yaxis_title="Cumulative Abnormal Return",
            template="plotly_dark", height=320
        )
        st.plotly_chart(fig_car, use_container_width=True)

# ==========================================
# TAB 2: Quant EDA & Regime Distributions
# ==========================================
with tab2:
    st.subheader("📊 Quant Feature Distributions & Regime Analysis (313 Clean Calls)")
    
    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        st.markdown("#### Tone Divergence Distribution (Overconfidence Spread)")
        fig_hist = px.histogram(df, x="divergence", nbins=30, color_discrete_sequence=["#38BDF8"], template="plotly_dark", labels={"divergence": "Tone Divergence (Exec - QA)"})
        fig_hist.add_vline(x=0.05, line_dash="dash", line_color="#EF4444", annotation_text="Overconfidence Threshold (+0.05)")
        fig_hist.add_vline(x=0.00, line_color="#94A3B8")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with eda_c2:
        st.markdown("#### Overconfidence Regime vs Post-Earnings CAR Drift")
        reg_over = df[df['divergence'] > 0.05]['car_5d'].mean()
        reg_align = df[df['divergence'] <= 0.05]['car_5d'].mean()
        regime_df = pd.DataFrame({
            "Regime": ["Overconfident (Divergence > +0.05)", "Aligned / Conservative (Divergence <= +0.05)"],
            "Mean 5D CAR": [reg_over, reg_align],
            "Count": [len(df[df['divergence'] > 0.05]), len(df[df['divergence'] <= 0.05])]
        })
        fig_regime = px.bar(regime_df, x="Regime", y="Mean 5D CAR", text_auto=".2%", color="Mean 5D CAR", color_continuous_scale=["#EF4444", "#10B981"], template="plotly_dark")
        fig_regime.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_regime, use_container_width=True)
        
    st.markdown("#### Quant & Linguistic Correlation Heatmap")
    num_cols = ['nsi_exec', 'nsi_qa', 'divergence', 'exec_hedge_pct', 'qa_ratio', 'car_5d']
    corr_matrix = df[num_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 3: Econometric Regression Engine
# ==========================================
with tab3:
    st.subheader("📈 Econometric Multi-Variable OLS Regression (White HC1 Robust)")
    st.markdown(r'''**Empirical Model:** $\text{CAR}_{5D} = \alpha + \beta_1 \cdot \text{NSI}_{\text{QA}} + \beta_2 \cdot \text{Divergence} + \beta_3 \cdot \text{QA Ratio} + \beta_4 \cdot \text{Exec Hedging} + \epsilon$''')
    
    X_cols = ['nsi_qa', 'divergence', 'qa_ratio', 'exec_hedge_pct']
    X = sm.add_constant(df[X_cols])
    y = df['car_5d']
    ols_model = sm.OLS(y, X).fit(cov_type='HC1')
    
    reg_col1, reg_col2 = st.columns([1.2, 1])
    with reg_col1:
        fig_scatter = px.scatter(
            df, x="divergence", y="car_5d", color="ticker",
            hover_data=["date", "nsi_exec", "nsi_qa"],
            labels={"divergence": "Tone Divergence (Exec - QA)", "car_5d": "5-Day CAR vs S&P 500"},
            title="Tone Divergence vs 5-Day Alpha Drift",
            template="plotly_dark", trendline="ols", trendline_color_override="#EF4444"
        )
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with reg_col2:
        st.write("**Regression Diagnostics (White HC1 Robust):**")
        summary_df = pd.DataFrame({
            "Variable": ["Constant", "NSI_QA (Analyst Tone)", "Divergence (Overconfidence)", "Q&A Ratio", "Exec Hedging (%)"],
            "Coefficient": [ols_model.params.iloc[i] for i in range(5)],
            "Robust t-stat": [ols_model.tvalues.iloc[i] for i in range(5)],
            "p-value": [ols_model.pvalues.iloc[i] for i in range(5)]
        })
        st.dataframe(summary_df.style.format({"Coefficient": "{:+.4f}", "Robust t-stat": "{:.2f}", "p-value": "{:.4f}"}), hide_index=True)
        st.metric("Model Observations", f"{len(df)}")
        st.metric("R-Squared", f"{ols_model.rsquared:.2%}")

# ==========================================
# TAB 4: Out-of-Sample Backtest & Equity Curve
# ==========================================
with tab4:
    st.subheader("🏆 Temporal Out-of-Sample Backtest & Baseline Comparison")
    st.markdown('''
    **Institutional Benchmark Protocol:**
    Comparing Machine Learning signal performance against **3 standard industry baselines** on strictly unseen forward test data (Zero Lookahead Bias).
    ''')
    
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    feat_cols = ['nsi_exec', 'nsi_qa', 'divergence', 'qa_ratio', 'exec_hedge_pct', 'is_overconfident']
    X_train = train_df[feat_cols]
    y_train = train_df['target_outperform']
    X_test = test_df[feat_cols]
    y_test = test_df['target_outperform']
    
    # ML Model
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    
    # Baseline 1: Naive Sentiment (Buy if NSI_QA > 0)
    y_pred_naive = (test_df['nsi_qa'] > 0).astype(int)
    # Baseline 2: Buy and Hold (Always predict Outperform 1)
    y_pred_bah = np.ones(len(y_test), dtype=int)
    # Baseline 3: Random Coin Flip (50/50)
    np.random.seed(42)
    y_pred_rand = np.random.choice([0, 1], size=len(y_test))
    
    # Performance Table
    benchmarks_df = pd.DataFrame({
        "Strategy / Model": ["Random Coin-Flip", "Buy & Hold Benchmark", "Naive Sentiment Rule (NSI > 0)", "Quant ML Random Forest"],
        "Precision (Win Rate)": [precision_score(y_test, y_pred_rand, zero_division=0), precision_score(y_test, y_pred_bah, zero_division=0), precision_score(y_test, y_pred_naive, zero_division=0), precision_score(y_test, y_pred_rf, zero_division=0)],
        "Recall": [recall_score(y_test, y_pred_rand, zero_division=0), recall_score(y_test, y_pred_bah, zero_division=0), recall_score(y_test, y_pred_naive, zero_division=0), recall_score(y_test, y_pred_rf, zero_division=0)],
        "F1-Score": [f1_score(y_test, y_pred_rand, zero_division=0), f1_score(y_test, y_pred_bah, zero_division=0), f1_score(y_test, y_pred_naive, zero_division=0), f1_score(y_test, y_pred_rf, zero_division=0)]
    })
    
    st.dataframe(benchmarks_df.style.format({"Precision (Win Rate)": "{:.1%}", "Recall": "{:.1%}", "F1-Score": "{:.3f}"}), hide_index=True)
    
    # Cumulative Equity Curve Simulation ($10,000 starting capital)
    st.markdown("#### 📈 Cumulative Portfolio Equity Curve ($10,000 Backtest on Unseen Test Calls)")
    initial_capital = 10000.0
    
    # Returns per strategy
    ml_returns = np.where(y_pred_rf == 1, test_df['car_5d'], 0.0)
    naive_returns = np.where(y_pred_naive == 1, test_df['car_5d'], 0.0)
    bah_returns = test_df['car_5d'].values
    
    ml_equity = initial_capital * np.cumprod(1 + ml_returns)
    naive_equity = initial_capital * np.cumprod(1 + naive_returns)
    bah_equity = initial_capital * np.cumprod(1 + bah_returns)
    
    trades = list(range(1, len(test_df) + 1))
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=trades, y=ml_equity, mode='lines', name='Quant ML Alpha Strategy', line=dict(color='#38BDF8', width=3)))
    fig_equity.add_trace(go.Scatter(x=trades, y=naive_equity, mode='lines', name='Naive Sentiment (NSI > 0)', line=dict(color='#F59E0B', width=2, dash='dot')))
    fig_equity.add_trace(go.Scatter(x=trades, y=bah_equity, mode='lines', name='Buy & Hold Benchmark', line=dict(color='#94A3B8', width=2, dash='dash')))
    fig_equity.add_hline(y=initial_capital, line_color="gray", line_dash="dash")
    fig_equity.update_layout(title="Out-of-Sample Portfolio Growth Trajectory ($10k Initial Capital)", xaxis_title="Chronological Trade Sequence", yaxis_title="Portfolio Value ($)", template="plotly_dark", height=380)
    st.plotly_chart(fig_equity, use_container_width=True)

# ==========================================
# TAB 5: Live FinBERT Sandbox
# ==========================================
with tab5:
    st.subheader("🧪 Live FinBERT Inference Sandbox")
    sample_text = st.text_area("Enter Financial Text / Earnings Call Snippet:", value="We delivered record revenue and expanded operating margins, although supply chain headwinds remain challenging in the international division.", height=110)
    if st.button("Run Live FinBERT"):
        with st.spinner("Tokenizing & inferring via ProsusAI/finbert..."):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", low_cpu_mem_usage=True)
            inputs = tok(sample_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                probs = torch.softmax(mdl(**inputs).logits, dim=-1)[0].tolist()
            p_pos, p_neg, p_neu = probs[0], probs[1], probs[2]
            nsi = p_pos - p_neg
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Positive Probability", f"{p_pos:.1%}")
            sc2.metric("Negative Probability", f"{p_neg:.1%}")
            sc3.metric("Neutral Probability", f"{p_neu:.1%}")
            sc4.metric("Net Sentiment Index (NSI)", f"{nsi:+.3f}")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=nsi, title={'text': "NSI Tone Score (-1 to +1)"},
                gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "#38BDF8"}, 'steps': [{'range': [-1, -0.2], 'color': "#7F1D1D"}, {'range': [-0.2, 0.2], 'color': "#334155"}, {'range': [0.2, 1], 'color': "#064E3B"}]}
            ))
            fig_gauge.update_layout(height=280, template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.caption("Built with PyTorch, FinBERT, yfinance, statsmodels, scikit-learn, and Streamlit.")
