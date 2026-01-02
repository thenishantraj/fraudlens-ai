"""
Configuration settings for FraudLens AI
"""

# Model parameters
ISOLATION_FOREST_PARAMS = {
    'contamination': 0.1,
    'random_state': 42,
    'n_estimators': 100
}

# Risk score thresholds
RISK_THRESHOLDS = {
    'LOW': 30,
    'MEDIUM': 60,
    'HIGH': 80
}

# Color scheme for UI
COLORS = {
    'background': '#0f172a',
    'card': '#1e293b',
    'primary': '#3b82f6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'text': '#f8fafc'
}

# Feature columns
NUMERICAL_FEATURES = ['amount', 'quantity', 'unit_price']
CATEGORICAL_FEATURES = ['category', 'department']
TEXT_FEATURES = ['description', 'justification']