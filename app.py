

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add custom modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_generator import save_sample_data
from models import FraudDetectionPipeline
import utils

# Page configuration
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-card { background-color: #1e293b; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6; }
    .high-risk { border-left: 4px solid #ef4444 !important; }
    div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=80)
        st.markdown("<h2 style='text-align: center;'>FraudLens AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Data Source
        st.markdown("### 📊 Data Source")
        if st.button("🚀 Generate & Analyze Data", use_container_width=True):
            with st.spinner("AI Analysis in progress..."):
                save_sample_data()
                df_raw = pd.read_csv('procurement_data.csv')
                pipeline = FraudDetectionPipeline()
                # Store full results
                st.session_state.results_df = pipeline.predict(df_raw)
                st.rerun()

        st.markdown("---")
        # Filters Section with Safety Check
        st.markdown("### 🔍 Filters")
        if st.session_state.results_df is not None:
            if 'risk_level' in st.session_state.results_df.columns:
                risk_levels = st.multiselect("Risk Level", ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], default=['HIGH', 'CRITICAL'])
                
                # Apply filters safely
                f_df = st.session_state.results_df.copy()
                if risk_levels:
                    f_df = f_df[f_df['risk_level'].isin(risk_levels)]
                st.session_state.filtered_df = f_df
            else:
                st.warning("Run analysis to enable filters.")

    # --- Main Content ---
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Advanced Anomaly Detection in Public Procurement</p>", unsafe_allow_html=True)

    if st.session_state.results_df is not None:
        # Check if we have filtered data
        display_df = st.session_state.get('filtered_df', st.session_state.results_df)
        
        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Tenders", len(st.session_state.results_df))
        
        # Calculate risk metrics safely
        anomalies = len(st.session_state.results_df[st.session_state.results_df['risk_level'] != 'LOW']) if 'risk_level' in st.session_state.results_df.columns else 0
        m2.metric("Anomalies", anomalies)
        
        avg_score = st.session_state.results_df['risk_score'].mean() if 'risk_score' in st.session_state.results_df.columns else 0
        m4.metric("Avg Risk Score", f"{avg_score:.1f}")

        st.markdown("---")
        
        # Visuals with Error Handling to fix PlotlyError
        c1, c2 = st.columns(2)
        with c1:
            if not display_df.empty and 'risk_level' in display_df.columns:
                fig_risk = utils.create_risk_distribution_chart(display_df)
                if fig_risk: st.plotly_chart(fig_risk, use_container_width=True)
        
        with c2:
            if not display_df.empty and 'unit_price' in display_df.columns:
                # Fix for line 211
                fig_price = utils.create_price_distribution_chart(display_df)
                if fig_price: st.plotly_chart(fig_price, use_container_width=True)

        st.markdown("### 📋 Detailed Analysis")
        st.dataframe(display_df, use_container_width=True)

    else:
        # Welcome screen
        st.info("👋 Welcome! Use the sidebar to generate sample data and start the AI Audit.")
        st.markdown("""
        ### Core Detection Engines:
        1. **Price Inflation**: Market average comparisons.
        2. **Document Duplication**: NLP-based bid-rigging detection.
        3. **Vendor Analysis**: Suspicious winning patterns.
        """)

if __name__ == "__main__":
    main()
