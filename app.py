"""
FraudLens AI - Complete Production Dashboard (Final Error-Free Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Custom paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local modules
from data_generator import save_sample_data
from models import FraudDetectionPipeline, VendorRiskAnalyzer
import utils

# --- Page Setup ---
st.set_page_config(page_title="FraudLens AI", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# --- Final Fixed CSS for Visibility ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f172a; color: #f8fafc; }
    .main-header { font-size: 2.8rem; font-weight: 800; color: #3b82f6; margin-bottom: 5px; }
    .metric-card { 
        background-color: #1e293b; padding: 25px; border-radius: 15px; 
        border: 1px solid #334155; border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label { color: #94a3b8 !important; font-size: 1rem; font-weight: 600; margin-bottom: 8px; }
    .metric-value { color: #ffffff !important; font-size: 2.5rem; font-weight: 800; margin: 0; }
    .high-risk-card { border-left-color: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'results_df' not in st.session_state: st.session_state.results_df = None

def main():
    # --- Sidebar Interaction ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=80)
        st.title("Control Panel")
        st.markdown("---")
        
        st.subheader("📁 Step 1: Input Data")
        mode = st.radio("Select Method:", ["Demo Mode (Recommended)", "Upload File"])
        
        temp_df = None
        if mode == "Demo Mode (Recommended)":
            if st.button("✨ Generate & Load Data", use_container_width=True):
                temp_df = save_sample_data()
                st.success("Demo Dataset Ready!")
        else:
            file = st.file_uploader("Upload Procurement CSV", type=['csv'])
            if file:
                temp_df = pd.read_csv(file)
                if 'total_amount' not in temp_df.columns:
                    st.error("Invalid Columns! Need procurement data.")
                    temp_df = None

        st.markdown("---")
        st.subheader("🧠 Step 2: AI Audit")
        if temp_df is not None:
            if st.button("🔍 START FRAUD ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("Analyzing Patterns..."):
                    pipeline = FraudDetectionPipeline()
                    st.session_state.results_df = pipeline.predict(temp_df)
                    st.rerun()

    # --- Main Header ---
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI Dashboard</h1>", unsafe_allow_html=True)
    
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        
        # --- Filters Logic (Solves KeyError) ---
        st.sidebar.subheader("🎯 Step 3: Filters")
        risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        if 'risk_level' in df.columns:
            selected_risks = st.sidebar.multiselect("Risk Level:", risk_levels, default=['HIGH', 'CRITICAL'])
            f_df = df[df['risk_level'].isin(selected_risks)] if selected_risks else df
        else:
            f_df = df

        # --- Metrics Row (Fixed Visibility) ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='metric-card'><p class='metric-label'>Total Audited</p><p class='metric-value'>{len(df)}</p></div>", unsafe_allow_html=True)
        with m2:
            anom = len(df[df['risk_level'].isin(['HIGH', 'CRITICAL'])]) if 'risk_level' in df.columns else 0
            st.markdown(f"<div class='metric-card high-risk-card'><p class='metric-label'>Anomalies Found</p><p class='metric-value'>{anom}</p></div>", unsafe_allow_html=True)
        with m3:
            avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 0
            st.markdown(f"<div class='metric-card'><p class='metric-label'>Avg Risk Score</p><p class='metric-value'>{avg_risk:.1f}</p></div>", unsafe_allow_html=True)
        with m4:
            total_val = df['total_amount'].sum() if 'total_amount' in df.columns else 0
            st.markdown(f"<div class='metric-card'><p class='metric-label'>Total Amount</p><p class='metric-value'>${(total_val/1e6):.1f}M</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # --- Charts Section ---
        v1, v2 = st.columns(2)
        with v1:
            risk_fig = utils.create_risk_distribution_chart(f_df)
            if risk_fig: st.plotly_chart(risk_fig, use_container_width=True)
        with v2:
            dept_fig = utils.create_anomaly_by_department_chart(f_df)
            if dept_fig: st.plotly_chart(dept_fig, use_container_width=True)

        # Forensic Data Log
        st.markdown("### 📋 Forensic Audit Table")
        st.dataframe(f_df, use_container_width=True, height=450)
        st.download_button("📥 Export AI Report", f_df.to_csv(index=False), "fraud_audit.csv", "text/csv")
        
    else:
        # Welcome Section for Users
        st.info("👋 System Ready. Use the sidebar Control Panel to load data and begin the AI Audit.")
        st.image("https://img.icons8.com/clouds/500/000000/fine-print.png", width=250)
        
        st.markdown("""
        ### How to start interacting:
        1. **Select Demo Mode** (Left Sidebar).
        2. **Generate Data**: Click '✨ Generate & Load Data'.
        3. **Analyze**: Click the red button '🔍 START FRAUD ANALYSIS'.
        """)

if __name__ == "__main__":
    main()
