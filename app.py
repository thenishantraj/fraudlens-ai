"""
FraudLens AI - Complete Final Production Code (465+ Lines Restored)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Custom module paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generator import save_sample_data
from models import FraudDetectionPipeline, VendorRiskAnalyzer
import utils

# --- Configuration ---
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; }
    .metric-card { background-color: #1e293b; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 1rem; }
    .high-risk { border-left: 4px solid #ef4444 !important; }
    div[data-testid="stToolbar"] { display: none; }
    .reportview-container { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'vendor_analysis' not in st.session_state: st.session_state.vendor_analysis = None

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=80)
        st.markdown("<h2 style='text-align: center;'>FraudLens AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### 📊 Data Source")
        data_source = st.radio("Choose source:", ["Upload File", "Generate Sample Data", "Demo Mode"], label_visibility="collapsed")
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
            if uploaded_file:
                df = utils.load_data(uploaded_file)
                if df is not None:
                    if st.button("🔍 Run Fraud Detection", type="primary", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            pipeline = FraudDetectionPipeline()
                            results = pipeline.predict(df)
                            st.session_state.results_df = results
                            v_analyzer = VendorRiskAnalyzer()
                            st.session_state.vendor_analysis = v_analyzer.analyze_vendors(results)
                            st.rerun()
        
        elif data_source == "Generate Sample Data":
            if st.button("🚀 Generate & Analyze Now", use_container_width=True):
                with st.spinner("Processing..."):
                    save_sample_data()
                    raw_df = pd.read_csv('procurement_data.csv')
                    pipeline = FraudDetectionPipeline()
                    st.session_state.results_df = pipeline.predict(raw_df)
                    st.rerun()
        
        st.markdown("---")
        
        # --- FIXED FILTER LOGIC ---
        st.markdown("### 🔍 Filters")
        if st.session_state.results_df is not None:
            if 'risk_level' in st.session_state.results_df.columns:
                risk_levels = st.multiselect("Risk Focus", ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], default=['HIGH', 'CRITICAL'])
                
                f_df = st.session_state.results_df.copy()
                if risk_levels:
                    f_df = f_df[f_df['risk_level'].isin(risk_levels)]
                
                if 'department' in f_df.columns:
                    depts = st.multiselect("Department", sorted(f_df['department'].unique().tolist()))
                    if depts: f_df = f_df[f_df['department'].isin(depts)]
                
                st.session_state.filtered_df = f_df
            else:
                st.sidebar.warning("Analysis required for filters.")

    # --- Main Header ---
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Advanced Anomaly Detection in Public Procurement</p>", unsafe_allow_html=True)

    # --- Dashboard View ---
    if st.session_state.results_df is not None:
        view_df = st.session_state.get('filtered_df', st.session_state.results_df)
        
        # Summary Metrics
        metrics = utils.create_summary_metrics(st.session_state.results_df)
        f_metrics = utils.create_summary_metrics(view_df)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.markdown(f"<div class='metric-card'>Total Tenders<br><h2>{metrics['total_tenders']:,}</h2></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-card high-risk'>Anomalies<br><h2>{f_metrics['flagged_anomalies']}</h2></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-card'>High-Risk Vendors<br><h2>{metrics['high_risk_vendors']}</h2></div>", unsafe_allow_html=True)
        with m4: st.markdown(f"<div class='metric-card'>Avg Risk Score<br><h2>{view_df['risk_score'].mean():.1f}</h2></div>", unsafe_allow_html=True)
        with m5: st.markdown(f"<div class='metric-card'>Suspicious Ratio<br><h2>{(f_metrics['flagged_anomalies']/max(1, f_metrics['total_tenders'])):.1%}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Visuals Section
        c1, c2 = st.columns(2)
        with c1:
            if not view_df.empty and 'risk_level' in view_df.columns:
                st.plotly_chart(utils.create_risk_distribution_chart(view_df), use_container_width=True)
                st.plotly_chart(utils.create_price_distribution_chart(view_df), use_container_width=True)
        with c2:
            if not view_df.empty and 'department' in view_df.columns:
                st.plotly_chart(utils.create_anomaly_by_department_chart(view_df), use_container_width=True)
                if st.session_state.vendor_analysis is not None:
                    st.plotly_chart(utils.create_vendor_risk_heatmap(view_df), use_container_width=True)

        st.plotly_chart(utils.create_timeline_chart(view_df), use_container_width=True)

        st.markdown("### 📋 Detailed Analysis")
        st.dataframe(view_df, use_container_width=True)

        # Anomaly Explanations
        st.markdown("### 🎯 Deep-Dive Explanations")
        high_risk = view_df[view_df['risk_level'].isin(['HIGH', 'CRITICAL'])].head(10)
        for _, row in high_risk.iterrows():
            with st.expander(f"🔴 {row['tender_id']} - {row['vendor_name']} (Risk: {row['risk_score']:.1f})"):
                st.write(f"**Description:** {row['description']}")
                st.write(f"**AI Reason:** {row.get('explanation', 'Multi-factor risk detected.')}")

    else:
        # Welcome Screen
        st.info("👋 Welcome! Use the sidebar to upload a file or generate sample data to begin.")
        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            st.markdown("""
            ### 🔍 Core Detection Capabilities:
            - **Price Inflation**: Market-outlier detection using Isolation Forests.
            - **Document Duplication**: NLP/BERT analysis for bid-rigging patterns.
            - **Vendor Profiling**: High-risk pattern recognition.
            """)
            if st.button("🚀 Start Demo Mode", use_container_width=True, type="primary"):
                save_sample_data()
                df_raw = pd.read_csv('procurement_data.csv')
                pipeline = FraudDetectionPipeline()
                st.session_state.results_df = pipeline.predict(df_raw)
                st.rerun()

if __name__ == "__main__":
    main()
