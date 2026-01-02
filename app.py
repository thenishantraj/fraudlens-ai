"""
FraudLens AI
"""

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
from models import FraudDetectionPipeline, VendorRiskAnalyzer
import utils

# Page configuration
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Updated CSS for Better Visibility ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #3b82f6; margin-bottom: 1rem; }
    /* Metric Card Fix for visibility */
    .metric-card { 
        background-color: #1e293b; 
        padding: 1.5rem; 
        border-radius: 10px; 
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-label { color: #94a3b8 !important; font-size: 0.9rem; font-weight: 600; }
    .metric-value { color: #ffffff !important; font-size: 1.8rem; font-weight: 700; }
    .high-risk { border-left: 4px solid #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state: st.session_state.results_df = None

def main():
    # --- Sidebar for User Interaction ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=80)
        st.header("Control Panel")
        st.markdown("---")
        
        # Interaction Point 1: File Upload
        st.subheader("📁 Step 1: Data Input")
        data_source = st.radio("Select Method:", ["Upload Your File", "Generate Demo Data"])
        
        uploaded_df = None
        if data_source == "Upload Your File":
            file = st.file_uploader("Upload Procurement CSV", type=['csv'])
            if file:
                uploaded_df = pd.read_csv(file)
                st.success("File Uploaded!")
        else:
            if st.button("🚀 Generate Sample Dataset"):
                uploaded_df = save_sample_data()
                st.success("Demo Data Ready!")

        st.markdown("---")
        
        # Interaction Point 2: Analysis Trigger
        st.subheader("🧠 Step 2: AI Audit")
        if uploaded_df is not None:
            if st.button("🔍 START FRAUD DETECTION", type="primary", use_container_width=True):
                with st.spinner("Analyzing patterns..."):
                    pipeline = FraudDetectionPipeline()
                    st.session_state.results_df = pipeline.predict(uploaded_df)
                    st.rerun()

    # --- Main Dashboard ---
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI</h1>", unsafe_allow_html=True)

    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        # Interaction Point 3: Interactive Filters
        st.sidebar.subheader("🎯 Step 3: Filter Results")
        risk_choice = st.sidebar.multiselect("Risk Focus:", ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], default=['HIGH', 'CRITICAL'])
        
        f_df = df[df['risk_level'].isin(risk_choice)] if risk_choice else df

        # Metrics Row with Fixed Visibility
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='metric-card'><p class='metric-label'>Total Audited</p><p class='metric-value'>{len(df)}</p></div>", unsafe_allow_html=True)
        with m2:
            anomalies = len(df[df['risk_level'].isin(['HIGH', 'CRITICAL'])])
            st.markdown(f"<div class='metric-card high-risk'><p class='metric-label'>Anomalies Found</p><p class='metric-value'>{anomalies}</p></div>", unsafe_allow_html=True)
        with m3:
            avg_risk = df['risk_score'].mean()
            st.markdown(f"<div class='metric-card'><p class='metric-label'>Avg Risk Score</p><p class='metric-value'>{avg_risk:.1f}</p></div>", unsafe_allow_html=True)
        with m4:
            ratio = (anomalies / len(df)) * 100
            st.markdown(f"<div class='metric-card'><p class='metric-label'>Suspicious Ratio</p><p class='metric-value'>{ratio:.1f}%</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts and Data Table
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(utils.create_risk_distribution_chart(f_df), use_container_width=True)
        with col_c2:
            st.plotly_chart(utils.create_anomaly_by_department_chart(f_df), use_container_width=True)

        st.markdown("### 📋 Forensic Audit Table")
        st.dataframe(f_df, use_container_width=True)
        
        # Download Interaction
        st.download_button("📥 Export Report", f_df.to_csv(index=False), "fraud_report.csv", "text/csv")

    else:
        # Welcome Interaction for new users
        st.info("👋 Hello! To start checking for fraud, please use the **Control Panel** on the left to upload your data.")
        st.image("https://img.icons8.com/clouds/500/000000/data-protection.png", width=300)

if __name__ == "__main__":
    main()
