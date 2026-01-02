"""
FraudLens AI - Main Streamlit Application
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
                help="File should contain procurement data with columns like vendor_name, description, unit_price, total_amount"
            )
            
            if uploaded_file:
                df = utils.load_data(uploaded_file)
                if df is not None:
                    st.success(f"✅ Loaded {len(df)} records")
                    
                    # Show data preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    # Run analysis
                    if st.button("🔍 Run Fraud Detection", type="primary", use_container_width=True):
                        with st.spinner("Analyzing data for anomalies..."):
                            try:
                                # Initialize and run pipeline
                                pipeline = FraudDetectionPipeline()
                                results_df = pipeline.predict(df)
                                
                                # Store in session state
                                st.session_state.pipeline = pipeline
                                st.session_state.results_df = results_df
                                
                                # Run vendor analysis
                                from models import VendorRiskAnalyzer
                                vendor_analyzer = VendorRiskAnalyzer()
                                st.session_state.vendor_analysis = vendor_analyzer.analyze_vendors(results_df)
                                
                                st.success("✅ Analysis complete!")
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
        
        elif data_source == "Generate Sample Data":
            if st.button("Generate Sample Data", use_container_width=True):
                with st.spinner("Generating synthetic data..."):
                    df = save_sample_data()
                    st.session_state.results_df = pd.read_csv('procurement_data.csv')
                    st.success("✅ Generated 250 sample records!")
                    st.rerun()
        
        else:  # Demo Mode
            if st.button("Load Demo Data", use_container_width=True):
                try:
                    # Try to load existing sample data
                    df = pd.read_csv('procurement_data.csv')
                    st.session_state.results_df = df
                    st.success("✅ Loaded demo data!")
                except:
                    # Generate if not exists
                    df = save_sample_data()
                    st.session_state.results_df = pd.read_csv('procurement_data.csv')
                    st.success("✅ Generated and loaded demo data!")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        if st.session_state.results_df is not None:
            # Risk level filter
            risk_levels = st.multiselect(
                "Risk Level",
                options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                default=['HIGH', 'CRITICAL']
            )
            
            # Department filter
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
        
        st.markdown("---")
        
        # Info section
        st.markdown("### ℹ️ About")
        st.info("""
        **FraudLens AI** detects:
        - Price anomalies
        - Document duplication
        - Suspicious vendor patterns
        - Bid-rigging indicators
        """)
        
        st.markdown("---")
        
        # Footer
        st.markdown(
            """
            <div style='text-align: center; color: #94a3b8;'>
                <p>Built by Team - VectorX</p>
                <p>v1.0.0</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Main content
    st.markdown("<h1 class='main-header'>🔍 FraudLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Advanced Anomaly Detection in Public Procurement & Government Spending</p>", unsafe_allow_html=True)
    
    # Dashboard
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        filtered_df = st.session_state.get('filtered_df', df)
        
        # Summary Metrics
        metrics = utils.create_summary_metrics(df)
        filtered_metrics = utils.create_summary_metrics(filtered_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin: 0; color: #94a3b8; font-size: 0.9rem;'>Total Tenders</h3>
                <h2 style='margin: 0; color: white;'>{metrics['total_tenders']:,}</h2>
                <p style='margin: 0; color: #94a3b8; font-size: 0.8rem;'>${metrics['total_amount']:,.0f} total</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_class = "high-risk" if filtered_metrics['flagged_anomalies'] > 0 else ""
            st.markdown(f"""
            <div class='metric-card {risk_class}'>
                <h3 style='margin: 0; color: #94a3b8; font-size: 0.9rem;'>Flagged Anomalies</h3>
                <h2 style='margin: 0; color: white;'>{filtered_metrics['flagged_anomalies']:,}</h2>
                <p style='margin: 0; color: #94a3b8; font-size: 0.8rem;'>
                    {filtered_metrics['high_risk']} high + {filtered_metrics['critical_risk']} critical
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin: 0; color: #94a3b8; font-size: 0.9rem;'>High-Risk Vendors</h3>
                <h2 style='margin: 0; color: white;'>{filtered_metrics['high_risk_vendors']:,}</h2>
                <p style='margin: 0; color: #94a3b8; font-size: 0.8rem;'>
                    {filtered_metrics['high_risk_vendors'] / max(1, df['vendor_name'].nunique()):.0%} of all vendors
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_risk = filtered_df['risk_score'].mean() if 'risk_score' in filtered_df.columns else 0
            risk_color = "#ef4444" if avg_risk > 60 else "#f59e0b" if avg_risk > 30 else "#10b981"
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin: 0; color: #94a3b8; font-size: 0.9rem;'>Avg Risk Score</h3>
                <h2 style='margin: 0; color: {risk_color};'>{avg_risk:.1f}</h2>
                <p style='margin: 0; color: #94a3b8; font-size: 0.8rem;'>
                    Scale: 0-100
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            suspicious_ratio = filtered_metrics['flagged_anomalies'] / max(1, filtered_metrics['total_tenders'])
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin: 0; color: #94a3b8; font-size: 0.9rem;'>Suspicious Ratio</h3>
                <h2 style='margin: 0; color: white;'>{suspicious_ratio:.1%}</h2>
                <p style='margin: 0; color: #94a3b8; font-size: 0.8rem;'>
                    Of filtered tenders
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts Section
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution
            risk_chart = utils.create_risk_distribution_chart(filtered_df)
            if risk_chart:
                st.plotly_chart(risk_chart, use_container_width=True)
            
            # Price Distribution
            price_chart = utils.create_price_distribution_chart(filtered_df)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
        
        with col2:
            # Anomaly by Department
            dept_chart = utils.create_anomaly_by_department_chart(filtered_df)
            if dept_chart:
                st.plotly_chart(dept_chart, use_container_width=True)
            
            # Vendor Risk Heatmap
            if st.session_state.vendor_analysis is not None:
                heatmap_chart = utils.create_vendor_risk_heatmap(filtered_df)
                if heatmap_chart:
                    st.plotly_chart(heatmap_chart, use_container_width=True)
        
        # Timeline Chart (full width)
        timeline_chart = utils.create_timeline_chart(filtered_df)
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed View
        st.markdown("### 📋 Detailed Analysis")
        
        # Show filtered results
        display_df = utils.format_risk_dataframe(filtered_df)
        
        # Color code risk levels
        def color_risk(val):
            if val == 'CRITICAL':
                return 'background-color: #dc2626; color: white'
            elif val == 'HIGH':
                return 'background-color: #ef4444; color: white'
            elif val == 'MEDIUM':
                return 'background-color: #f59e0b; color: white'
            elif val == 'LOW':
                return 'background-color: #10b981; color: white'
            return ''
        
        # Apply styling
        styled_df = display_df.style.applymap(
            color_risk,
            subset=['risk_level']
        )
        
        # Display dataframe
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Export Results (CSV)"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fraudlens_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export High-Risk Only"):
                high_risk_df = filtered_df[filtered_df['risk_level'].isin(['HIGH', 'CRITICAL'])]
                csv = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="Download High-Risk CSV",
                    data=csv,
                    file_name=f"fraudlens_high_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🔄 Reset Filters"):
                st.session_state.filtered_df = st.session_state.results_df
                st.rerun()
        
        # Explanation Section
        st.markdown("---")
        st.markdown("### 🎯 Anomaly Explanations")
        
        high_risk_items = filtered_df[filtered_df['risk_level'].isin(['HIGH', 'CRITICAL'])]
        
        if len(high_risk_items) > 0:
            for idx, row in high_risk_items.head(10).iterrows():
                risk_color = "#ef4444" if row['risk_level'] == 'HIGH' else "#dc2626"
                
                with st.expander(f"🔴 {row['tender_id']} - {row['vendor_name']} (Risk: {row['risk_level']}, Score: {row['risk_score']:.1f})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Department", row['department'])
                        st.metric("Category", row['category'])
                    
                    with col2:
                        st.metric("Total Amount", f"${row['total_amount']:,.2f}")
                        st.metric("Unit Price", f"${row.get('unit_price', 0):,.2f}")
                    
                    with col3:
                        st.metric("Risk Level", row['risk_level'])
                        st.metric("Document Similarity", "Yes" if row.get('document_similarity', 0) == 1 else "No")
                    
                    # Explanation
                    if 'explanation' in row and pd.notna(row['explanation']):
                        st.markdown(f"**Explanation:** {row['explanation']}")
                    else:
                        st.markdown("**Explanation:** Multiple risk factors detected")
                    
                    # Description
                    st.markdown(f"**Description:** {row['description']}")
        else:
            st.info("No high-risk items found with current filters.")
    
    else:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Welcome to FraudLens AI
            
            ### 🎯 **Mission**
            Detect fraud, irregularities, and anomalies in public procurement and government spending data.
            
            ### 🔍 **What We Detect**
            - **Price Inflation**: Unusually high prices compared to market averages
            - **Document Duplication**: Identical or highly similar tender descriptions
            - **Suspicious Vendor Patterns**: Vendors with abnormal winning patterns
            - **Bid-Rigging Indicators**: Patterns suggesting collusion
            
            ### 🚀 **Getting Started**
            1. **Upload** your procurement data (CSV/Excel)
            2. **Generate** sample data for a demo
            3. **Analyze** and visualize results
            
            ### 📊 **Key Features**
            - Real-time anomaly detection
            - Interactive visualizations
            - Risk scoring (0-100)
            - Detailed explanations for flagged items
            - Vendor risk profiling
            """)
        
    
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🚀 Generate Sample Data", use_container_width=True):
                with st.spinner("Generating..."):
                    save_sample_data()
                    st.session_state.results_df = pd.read_csv('procurement_data.csv')
                    st.success("Done! Check the sidebar.")
                    st.rerun()
            
            st.info("""
            **Sample data includes:**
            - 250 synthetic procurement records
            - Built-in anomalies
            - Realistic government spending patterns
            """)

if __name__ == "__main__":
    main()
