"""
FraudLens AI - Main Streamlit Application (Final Corrected Version)
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
from models import FraudDetectionPipeline
import utils

# Page configuration
st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
    .high-risk {
        border-left: 4px solid #ef4444 !important;
    }
    .medium-risk {
        border-left: 4px solid #f59e0b !important;
    }
    .low-risk {
        border-left: 4px solid #10b981 !important;
    }
    .stDataFrame {
        background-color: #1e293b;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    .reportview-container {
        background: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'vendor_analysis' not in st.session_state:
    st.session_state.vendor_analysis = None

def main():
    """Main application function"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/detective.png", width=80)
        st.markdown("<h2 style='text-align: center;'>FraudLens AI</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Data Source Selection
        st.markdown("### 📊 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Generate Sample Data", "Demo Mode"],
            label_visibility="collapsed"
        )
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file",
                type=['csv', 'xlsx'],
                help="File should contain vendor_name, description, unit_price, total_amount"
            )
            
            if uploaded_file:
                df = utils.load_data(uploaded_file)
                if df is not None:
                    st.success(f"✅ Loaded {len(df)} records")
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🔍 Run Fraud Detection", type="primary", use_container_width=True):
                        with st.spinner("Analyzing data for anomalies..."):
                            try:
                                pipeline = FraudDetectionPipeline()
                                results_df = pipeline.predict(df)
                                st.session_state.pipeline = pipeline
                                st.session_state.results_df = results_df
                                from models import VendorRiskAnalyzer
                                vendor_analyzer = VendorRiskAnalyzer()
                                st.session_state.vendor_analysis = vendor_analyzer.analyze_vendors(results_df)
                                st.success("✅ Analysis complete!")
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
        
        elif data_source == "Generate Sample Data":
            if st.button("Generate Sample Data", use_container_width=True):
                with st.spinner("Generating synthetic data..."):
                    save_sample_data()
                    st.session_state.results_df = pd.read_csv('procurement_data.csv')
                    st.success("✅ Generated 250 sample records!")
                    st.rerun()
        
        else:  # Demo Mode
            if st.button("Load Demo Data", use_container_width=True):
                try:
                    df = pd.read_csv('procurement_data.csv')
                    st.session_state.results_df = df
                    st.success("✅ Loaded demo data!")
                except:
                    save_sample_data()
                    st.session_state.results_df = pd.read_csv('procurement_data.csv')
                    st.success("✅ Generated and loaded demo data!")
        
        st.markdown("---")
        
        # Filters Section (Fixed Indentation & KeyError)
        st.markdown("### 🔍 Filters")
        if st.session_state.results_df is not None:
            # Check for risk_level column before applying filters
            if 'risk_level' in st.session_state.results_df.columns:
                risk_levels = st.multiselect(
                    "Risk Level",
                    options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                    default=['HIGH', 'CRITICAL']
                )
                
                departments = st.multiselect(
                    "Department",
                    options=sorted(st.session_state.results_df['department'].unique().tolist())
                )
                
                # Apply filters
                filtered_df = st.session_state.results_df.copy()
                if risk_levels:
                    filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]
                if departments:
                    filtered_df = filtered_df[filtered_df['department'].isin(departments)]
                
                st.session_state.filtered_df = filtered_df
            else:
                st.sidebar.warning("Please run analysis to unlock filters.")
        
        st.markdown("---")
        st.info("Built by Team VectorX")

    # Main content
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Advanced Anomaly Detection in Public Procurement</p>", unsafe_allow_html=True)
    
    # Dashboard View
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        filtered_df = st.session_state.get('filtered_df', df)
        
        # Summary Metrics
        metrics = utils.create_summary_metrics(df)
        f_metrics = utils.create_summary_metrics(filtered_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"<div class='metric-card'>Total Tenders<br><h2>{metrics['total_tenders']:,}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card high-risk'>Anomalies<br><h2>{f_metrics['flagged_anomalies']}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'>High-Risk Vendors<br><h2>{metrics['high_risk_vendors']}</h2></div>", unsafe_allow_html=True)
        with col4:
            avg_risk = filtered_df['risk_score'].mean() if 'risk_score' in filtered_df.columns else 0
            st.markdown(f"<div class='metric-card'>Avg Risk Score<br><h2>{avg_risk:.1f}</h2></div>", unsafe_allow_html=True)
        with col5:
            ratio = f_metrics['flagged_anomalies'] / max(1, f_metrics['total_tenders'])
            st.markdown(f"<div class='metric-card'>Suspicious Ratio<br><h2>{ratio:.1%}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Charts Section
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if 'risk_level' in filtered_df.columns:
                st.plotly_chart(utils.create_risk_distribution_chart(filtered_df), use_container_width=True)
            st.plotly_chart(utils.create_price_distribution_chart(filtered_df), use_container_width=True)
        
        with col_c2:
            if 'department' in filtered_df.columns:
                st.plotly_chart(utils.create_anomaly_by_department_chart(filtered_df), use_container_width=True)
            if st.session_state.vendor_analysis is not None:
                st.plotly_chart(utils.create_vendor_risk_heatmap(filtered_df), use_container_width=True)
        
        st.plotly_chart(utils.create_timeline_chart(filtered_df), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Detailed Analysis")
        st.dataframe(filtered_df, use_container_width=True)
        
    else:
        # Professional Welcome Screen
        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            st.markdown("""
            ## Welcome to FraudLens AI
            ### 🎯 Mission
            Detect fraud, irregularities, and anomalies in government spending data.
            ### 🔍 What We Detect
            - **Price Inflation**: Market average comparisons.
            - **Document Duplication**: NLP-based bid-rigging detection.
            - **Vendor Patterns**: Suspicious winning streaks.
            """)
        
        with col_w2:
            st.markdown("### ⚡ Quick Actions")
            if st.button("🚀 Generate Sample Data", use_container_width=True):
                save_sample_data()
                st.session_state.results_df = pd.read_csv('procurement_data.csv')
                st.rerun()

if __name__ == "__main__":
    main()
