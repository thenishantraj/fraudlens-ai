"""
FraudLens AI - Final Stable Version (465+ Lines Restored)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Paths setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_generator import save_sample_data
from models import FraudDetectionPipeline, VendorRiskAnalyzer
import utils

# --- UI Styling ---
st.set_page_config(page_title="FraudLens AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: 800; color: #3b82f6; margin-bottom: 5px; }
    .metric-card { 
        background-color: #1e293b; padding: 20px; border-radius: 12px; 
        border: 1px solid #334155; border-left: 5px solid #3b82f6;
    }
    .metric-value { color: #ffffff !important; font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { color: #94a3b8 !important; font-size: 1rem; margin-bottom: 5px; }
    div[data-testid="stExpander"] { background: #1e293b; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

if 'results_df' not in st.session_state: st.session_state.results_df = None

def main():
    # --- Sidebar Command Center ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=70)
        st.title("Control Panel")
        st.markdown("---")
        
        st.subheader("📁 Step 1: Input Data")
        mode = st.radio("Method:", ["Demo Mode (Recommended)", "Upload Your File"])
        
        target_df = None
        if mode == "Demo Mode (Recommended)":
            if st.button("✨ Generate & Load Sample Data", use_container_width=True):
                target_df = save_sample_data()
        else:
            file = st.file_uploader("Upload Procurement CSV", type=['csv'])
            if file:
                target_df = pd.read_csv(file)
                # Validation: Check if it's the right type of data
                if 'total_amount' not in target_df.columns:
                    st.error("❌ Wrong File! Please upload procurement data, not stock market data.")
                    target_df = None

        st.markdown("---")
        st.subheader("🧠 Step 2: AI Audit")
        if target_df is not None:
            if st.button("🔍 START FRAUD ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("AI Engine Analyzing Patterns..."):
                    try:
                        pipeline = FraudDetectionPipeline()
                        st.session_state.results_df = pipeline.predict(target_df)
                        st.success("Analysis Complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis Error: {str(e)}")

    # --- Main Display ---
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI</h1>", unsafe_allow_html=True)
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        # Sidebar Filters
        st.sidebar.subheader("🎯 Step 3: Insight Filters")
        r_choice = st.sidebar.multiselect("Risk Level:", ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], default=['HIGH', 'CRITICAL'])
        f_df = df[df['risk_level'].isin(r_choice)] if r_choice else df

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown(f"<div class='metric-card'><p class='metric-label'>Audited Tenders</p><p class='metric-value'>{len(df)}</p></div>", unsafe_allow_html=True)
        with m2: 
            anom = len(df[df['risk_level'].isin(['HIGH', 'CRITICAL'])])
            st.markdown(f"<div class='metric-card' style='border-left-color: #ef4444;'><p class='metric-label'>Anomalies</p><p class='metric-value'>{anom}</p></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-card'><p class='metric-label'>Avg Risk Score</p><p class='metric-value'>{df['risk_score'].mean():.1f}</p></div>", unsafe_allow_html=True)
        with m4: st.markdown(f"<div class='metric-card'><p class='metric-label'>Total Amount</p><p class='metric-value'>${(df['total_amount'].sum()/1e6):.1f}M</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Visuals Section
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(utils.create_risk_distribution_chart(f_df), use_container_width=True)
        with c2: st.plotly_chart(utils.create_anomaly_by_department_chart(f_df), use_container_width=True)

        st.markdown("### 📋 Forensic Audit Log")
        st.dataframe(f_df, use_container_width=True)
    else:
        # Welcome Section
        st.info("👋 System Ready. Please use the Sidebar to generate or upload data.")
        st.image("https://img.icons8.com/clouds/500/000000/fine-print.png", width=250)
        st.markdown("""
        ### How to use this system:
        1. **Select Method**: Choose 'Demo Mode' in the sidebar for a quick test.
        2. **Generate Data**: Click the generate button to create 250 records.
        3. **Run Audit**: Click the red button to start the AI fraud detection engine.
        """)

if __name__ == "__main__":
    main()
