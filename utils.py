"""
Utility functions for FraudLens AI
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Basic validation
        required_columns = ['vendor_name', 'description', 'unit_price', 'total_amount']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}. Some features may not work properly.")
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def format_currency(amount):
    """Format amount as currency"""
    if pd.isna(amount):
        return "N/A"
    return f"${amount:,.2f}"

def create_summary_metrics(df):
    """Create summary metrics for dashboard"""
    total_tenders = len(df)
    total_amount = df['total_amount'].sum() if 'total_amount' in df.columns else 0
    avg_amount = df['total_amount'].mean() if 'total_amount' in df.columns else 0
    
    if 'risk_level' in df.columns:
        high_risk = (df['risk_level'] == 'HIGH').sum()
        critical_risk = (df['risk_level'] == 'CRITICAL').sum()
        flagged_anomalies = high_risk + critical_risk
        
        high_risk_vendors = df[df['risk_level'].isin(['HIGH', 'CRITICAL'])]['vendor_name'].nunique()
    else:
        high_risk = 0
        critical_risk = 0
        flagged_anomalies = 0
        high_risk_vendors = 0
    
    return {
        'total_tenders': total_tenders,
        'total_amount': total_amount,
        'avg_amount': avg_amount,
        'flagged_anomalies': flagged_anomalies,
        'high_risk': high_risk,
        'critical_risk': critical_risk,
        'high_risk_vendors': high_risk_vendors
    }

def create_risk_distribution_chart(df):
    """Create risk distribution pie chart"""
    if 'risk_level' not in df.columns:
        return None
    
    risk_counts = df['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['risk_level', 'count']
    
    # Define color mapping
    color_map = {
        'LOW': '#10b981',
        'MEDIUM': '#f59e0b',
        'HIGH': '#ef4444',
        'CRITICAL': '#dc2626'
    }
    
    fig = px.pie(
        risk_counts,
        values='count',
        names='risk_level',
        title='Risk Level Distribution',
        color='risk_level',
        color_discrete_map=color_map,
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_anomaly_by_department_chart(df):
    """Create anomaly distribution by department"""
    if 'risk_level' not in df.columns or 'department' not in df.columns:
        return None
    
    high_risk_df = df[df['risk_level'].isin(['HIGH', 'CRITICAL'])]
    
    if len(high_risk_df) == 0:
        return None
    
    dept_risk = high_risk_df['department'].value_counts().reset_index()
    dept_risk.columns = ['department', 'high_risk_count']
    
    fig = px.bar(
        dept_risk.head(10),
        x='department',
        y='high_risk_count',
        title='High-Risk Tendert by Department (Top 10)',
        color='high_risk_count',
        color_continuous_scale='reds'
    )
    
    fig.update_layout(
        xaxis_title="Department",
        yaxis_title="Number of High-Risk Tenders",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        coloraxis_showscale=False
    )
    
    return fig

def create_price_distribution_chart(df):
    """Create price distribution with anomaly highlighting"""
    if 'unit_price' not in df.columns or 'risk_level' not in df.columns:
        return None
    
    fig = px.box(
        df,
        y='unit_price',
        color='risk_level',
        title='Price Distribution by Risk Level',
        color_discrete_map={
            'LOW': '#10b981',
            'MEDIUM': '#f59e0b',
            'HIGH': '#ef4444',
            'CRITICAL': '#dc2626'
        },
        log_y=True
    )
    
    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Unit Price (log scale)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=True
    )
    
    return fig

def create_vendor_risk_heatmap(df):
    """Create vendor risk heatmap"""
    if 'vendor_name' not in df.columns or 'risk_score' not in df.columns:
        return None
    
    # Aggregate vendor data
    vendor_stats = df.groupby('vendor_name').agg({
        'risk_score': 'mean',
        'tender_id': 'count',
        'total_amount': 'sum'
    }).reset_index()
    
    vendor_stats.columns = ['vendor', 'avg_risk_score', 'contract_count', 'total_amount']
    
    # Sort by risk score
    vendor_stats = vendor_stats.sort_values('avg_risk_score', ascending=False).head(20)
    
    # Create heatmap data
    heatmap_data = vendor_stats[['vendor', 'avg_risk_score', 'contract_count']].set_index('vendor')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Avg Risk Score', 'Contract Count'],
        y=heatmap_data.index,
        colorscale='reds',
        showscale=True,
        hovertemplate='<b>%{y}</b><br>' +
                      'Metric: %{x}<br>' +
                      'Value: %{z:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Vendor Risk Heatmap (Top 20)',
        xaxis_title="Metrics",
        yaxis_title="Vendor",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=600
    )
    
    return fig

def create_timeline_chart(df):
    """Create timeline of tender amounts with risk highlighting"""
    if 'tender_date' not in df.columns or 'total_amount' not in df.columns or 'risk_level' not in df.columns:
        return None
    
    df_timeline = df.copy()
    df_timeline['tender_date'] = pd.to_datetime(df_timeline['tender_date'])
    df_timeline = df_timeline.sort_values('tender_date')
    
    # Color mapping for risk levels
    color_map = {
        'LOW': '#10b981',
        'MEDIUM': '#f59e0b',
        'HIGH': '#ef4444',
        'CRITICAL': '#dc2626'
    }
    
    fig = px.scatter(
        df_timeline,
        x='tender_date',
        y='total_amount',
        color='risk_level',
        size='risk_score',
        hover_data=['vendor_name', 'description', 'risk_score'],
        title='Tender Timeline with Risk Indicators',
        color_discrete_map=color_map,
        log_y=True
    )
    
    fig.update_layout(
        xaxis_title="Tender Date",
        yaxis_title="Total Amount (log scale)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hovermode='closest'
    )
    
    return fig

def format_risk_dataframe(df):
    """Format dataframe for display with styling"""
    display_cols = [
        'tender_id', 'vendor_name', 'department', 'category',
        'total_amount', 'risk_score', 'risk_level', 'explanation'
    ]
    
    # Filter to available columns
    available_cols = [col for col in display_cols if col in df.columns]
    display_df = df[available_cols].copy()
    
    # Format currency
    if 'total_amount' in display_df.columns:
        display_df['total_amount'] = display_df['total_amount'].apply(format_currency)
    
    # Format risk score
    if 'risk_score' in display_df.columns:
        display_df['risk_score'] = display_df['risk_score'].round(1)
    
    return display_df