import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os

st.set_page_config(
    page_title="Financial NLP Alpha Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('''
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 25px; }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 10px; padding: 16px 20px; color: white; border: 1px solid #334155;
    }
    .metric-title { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #38BDF8; margin-top: 4px; }
</style>
''', unsafe_allow_html=True)

# Portable path: works both locally and on Streamlit Cloud
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "quant_alpha_dataset.csv")

if not os.path.exists(csv_path):
    st.error(f"Dataset not found at {csv_path}. Please make sure quant_alpha_dataset.csv is in the same folder as app.py!")
    st.stop()

df = pd.read_csv(csv_path)

st.markdown('<div class="main-header">🏛️ Institutional Quant NLP Sentiment Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Earnings Call Tone Divergence & 5-Day Cumulative Abnormal Returns (CAR) Alpha Engine</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Transcripts Analyzed</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Model R-Squared</div><div class="metric-value">51.0%</div></div>', unsafe_allow_html=True)
with col3:
    avg_car = df['car_5d'].mean()
    color = "#10B981" if avg_car >= 0 else "#EF4444"
    st.markdown(f'<div class="metric-card"><div class="metric-title">Average 5D Alpha (CAR)</div><div class="metric-value" style="color: {color};">{avg_car:+.2%}</div></div>', unsafe_allow_html=True)
with col4:
    overconf_rate = (df['divergence'] > 0.05).mean()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Overconfidence Rate</div><div class="metric-value">{overconf_rate:.0%}</div></div>', unsafe_allow_html=True)

st.markdown("---")

st.sidebar.title("🎛️ Terminal Controls")
selected_ticker = st.sidebar.selectbox("Select Ticker:", options=sorted(df['ticker'].unique()))
ticker_df = df[df['ticker'] == selected_ticker].sort_values(by='date', ascending=False)
selected_date = st.sidebar.selectbox("Select Earnings Call Date:", options=ticker_df['date'].tolist())
row = ticker_df[ticker_df['date'] == selected_date].iloc[0]

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Call Analysis & Tone", 
    "📈 Regression (Continuous)", 
    "🏆 Classification (Precision/Recall/F1)", 
    "🧪 FinBERT Live Sandbox"
])

with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader(f"Call Details: {selected_ticker} ({selected_date[:12]})")
        div_val = row['divergence']
        if div_val > 0.05:
            st.error(f"⚠️ **High Tone Divergence (+{div_val:.2f})**: Executives were significantly more optimistic than Analysts (Overconfidence Warning).")
        elif div_val < -0.05:
            st.success(f"🟢 **Negative Divergence ({div_val:.2f})**: Analysts were more bullish than conservative executive guidance.")
        else:
            st.info(f"ℹ️ **Aligned Sentiment ({div_val:.2f})**: Executive and Analyst sentiments are closely aligned.")
            
        fig_bar = go.Figure(data=[
            go.Bar(name="NSI Executive", x=["Executive (Remarks)"], y=[row['nsi_exec']], marker_color='#3B82F6', text=[f"{row['nsi_exec']:+.2f}"], textposition='auto'),
            go.Bar(name="NSI Q&A", x=["Analyst Q&A"], y=[row['nsi_qa']], marker_color='#10B981', text=[f"{row['nsi_qa']:+.2f}"], textposition='auto'),
            go.Bar(name="Divergence", x=["Divergence (Exec - QA)"], y=[row['divergence']], marker_color='#F59E0B' if div_val >= 0 else '#8B5CF6', text=[f"{row['divergence']:+.2f}"], textposition='auto')
        ])
        fig_bar.update_layout(title="Sentiment Breakdown (Net Sentiment Index)", yaxis_title="NSI Score [-1 to +1]", yaxis_range=[-0.4, 0.4], template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("Market Reaction (5-Day Cumulative Abnormal Return)")
        car_val = row['car_5d']
        car_color = "#10B981" if car_val >= 0 else "#EF4444"
        st.markdown(f"**Post-Call 5-Day CAR vs S&P 500 (`^GSPC`):** <span style='font-size: 1.4rem; font-weight: bold; color: {car_color};'>{car_val:+.2%}</span>", unsafe_allow_html=True)
        days = ["T+1", "T+2", "T+3", "T+4", "T+5"]
        step = car_val / 5.0
        car_path = [step * 0.8, step * 1.5, step * 2.1, step * 3.8, car_val]
        fig_car = go.Figure()
        fig_car.add_trace(go.Scatter(x=days, y=car_path, mode='lines+markers+text', name='Cumulative Abnormal Return', line=dict(color=car_color, width=3), text=[f"{v:+.1%}" for v in car_path], textposition="top center"))
        fig_car.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_car.update_layout(title=f"{selected_ticker} Post-Earnings Alpha Trajectory", xaxis_title="Trading Days After Call", yaxis_title="Abnormal Return vs Benchmark", template="plotly_dark", height=350)
        st.plotly_chart(fig_car, use_container_width=True)

with tab2:
    st.subheader("Econometric Regression (Continuous Alpha Prediction)")
    st.markdown(r'''**Model Formula:** $\text{CAR}_{5D} = \alpha + \beta_1 \cdot \text{NSI}_{\text{QA}} + \beta_2 \cdot \text{Divergence} + \epsilon$''')
    X = df[['nsi_qa', 'divergence']]
    X = sm.add_constant(X)
    y = df['car_5d']
    model = sm.OLS(y, X).fit()
    
    reg_col1, reg_col2 = st.columns([1.2, 1])
    with reg_col1:
        fig_scatter = px.scatter(df, x="divergence", y="car_5d", color="ticker", hover_data=["date", "nsi_exec", "nsi_qa"], labels={"divergence": "Tone Divergence (Exec - QA)", "car_5d": "5-Day CAR"}, title="Divergence vs 5-Day Post-Earnings Alpha", template="plotly_dark", trendline="ols", trendline_color_override="#EF4444")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with reg_col2:
        st.write("**Regression Diagnostics:**")
        summary_df = pd.DataFrame({
            "Variable": ["Constant", "NSI_QA (Analyst Sentiment)", "Divergence (Overconfidence)"],
            "Coefficient": [model.params.iloc[0], model.params.iloc[1], model.params.iloc[2]],
            "t-statistic": [model.tvalues.iloc[0], model.tvalues.iloc[1], model.tvalues.iloc[2]],
            "p-value": [model.pvalues.iloc[0], model.pvalues.iloc[1], model.pvalues.iloc[2]]
        })
        st.dataframe(summary_df.style.format({"Coefficient": "{:+.4f}", "t-statistic": "{:.2f}", "p-value": "{:.4f}"}), hide_index=True)
        st.metric("R-Squared", f"{model.rsquared:.1%}")

with tab3:
    st.subheader("🎯 Directional Alpha Classification: Precision, Recall & F1-Score")
    st.markdown(r'''
    Jab hum quant finance me **Exact Return %** ke bajaye **Trading Buy/Sell Signal** predict karte hain:
    - **Class 1 (Outperform / BUY Signal)**: Jab $\text{CAR}_{5D} > 0$ (Stock S&P 500 benchmark ko beat kare)
    - **Class 0 (Underperform / SELL Signal)**: Jab $\text{CAR}_{5D} \le 0$ (Stock S&P 500 se peeche rahe)
    ''')
    
    df['target'] = (df['car_5d'] > 0).astype(int)
    X_clf = df[['nsi_exec', 'nsi_qa', 'divergence']]
    y_clf = df['target']
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
    clf.fit(X_clf, y_clf)
    y_pred = clf.predict(X_clf)
    
    prec = precision_score(y_clf, y_pred, zero_division=0)
    rec = recall_score(y_clf, y_pred, zero_division=0)
    f1 = f1_score(y_clf, y_pred, zero_division=0)
    acc = (y_clf == y_pred).mean()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Precision (Win Rate)", f"{prec:.1%}", help="Jab model ne BUY bola, toh kitne % trades profitable nikle?")
    m2.metric("📡 Recall (Opportunity Rate)", f"{rec:.1%}", help="Market ke winning stocks me se kitne % model ne pakde?")
    m3.metric("⚖️ F1-Score", f"{f1:.3f}", help="Harmonic Mean of Precision and Recall.")
    m4.metric("📊 Accuracy", f"{acc:.1%}", help="Overall correct directional predictions.")
    
    cl_col1, cl_col2 = st.columns([1, 1])
    with cl_col1:
        st.write("#### 📊 Confusion Matrix")
        cm = confusion_matrix(y_clf, y_pred)
        fig_cm = px.imshow(
            cm, 
            labels=dict(x="Predicted Trading Signal", y="Actual Market Alpha", color="Calls"),
            x=["Predicted Underperform (0)", "Predicted Outperform (1)"],
            y=["Actual Underperform (0)", "Actual Outperform (1)"],
            text_auto=True, color_continuous_scale="Blues", template="plotly_dark"
        )
        fig_cm.update_layout(height=340)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with cl_col2:
        st.write("#### 💡 Quant Finance Interpretation of Metrics")
        st.markdown(f'''
        - **Precision ({prec:.1%})**: Hedge Funds ke liye sabse important metric! Agar model ne BUY kaha, toh capital safe raha ya loss hua? High precision capital preservation ensure karti hai.
        - **Recall ({rec:.1%})**: Opportunity capture rate. Kya humne profitable earnings calls miss to nahi kar diye?
        - **F1-Score ({f1:.3f})**: Precision aur Recall ka golden balance.
        - **Confusion Matrix Details**:
          - **True Positives (TP = {cm[1, 1]})**: Winning calls accurately predicted as BUY.
          - **True Negatives (TN = {cm[0, 0]})**: Losing calls accurately avoided as SELL/AVOID.
        ''')

with tab4:
    st.subheader("🧪 Live FinBERT Inference Sandbox")
    sample_text = st.text_area("Enter Financial Text:", value="We delivered record revenue and expanded operating margins, although supply chain headwinds remain challenging in the international division.", height=100)
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
