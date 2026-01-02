# 🔍 FraudLens AI: Intelligent Procurement Forensic Engine
> **Safeguarding Public Funds through AI-Driven Anomaly Detection & Forensic Audit.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge)

---

##  Project Overview
**FraudLens AI** is a cutting-edge forensic tool designed to detect corruption, bid-rigging, and price inflation in public procurement data. Unlike traditional reactive audits, FraudLens provides **Proactive Intelligence**, enabling auditors to stop suspicious transactions *before* they are finalized.

By combining **Machine Learning (Isolation Forest)** and **Natural Language Processing (NLP)**, our platform uncovers hidden collusion patterns that human eyes often miss.

---

##  The Challenge
Public procurement accounts for billions in government spending, yet it remains the #1 sector for financial fraud.
- Manual Audits: Impossible to scan thousands of complex documents manually.
- Price Inflation: No automated way to compare unit prices across different departments.
- Bid-Rigging: Colluding vendors often submit identical bid descriptions with minor changes—undetectable without NLP.

---

##  Key Features
Our dashboard is built for **Decision Makers** and **Auditors**, providing instant clarity through data storytelling:

-  Isolation Forest Anomaly Detection: High-dimensional outlier detection to find price-gouging and abnormal spending.
-  NLP Document Similarity: Uses Cosine Similarity to detect "copy-paste" bid submissions between competing vendors—the smoking gun of bid-rigging.
-  Vendor Risk Heatmaps: Identifies "Habitual Offenders" by tracking risk density across multiple contracts.
-  Risk Timelines: Visualizes the surge of high-risk bids over time to detect seasonal fraud cycles.
-  Explainable AI: Every flagged record comes with a human-readable explanation (e.g., "Price is 5.3 standard deviations above category average").

---

##  Technical Architecture

### **Core Engines**
1. **The Forensic Pipeline (`models.py`):** Integrates Scikit-Learn for ML and TF-IDF Vectorization for NLP analysis.
2. **Dynamic UI (`app.py`):** A responsive Streamlit frontend with custom CSS for an enterprise-dark theme.
3. **Data Simulation (`data_generator.py`):** Sophisticated synthetic data generation to simulate real-world government spending patterns for testing.

---

##  Getting Started

### Prerequisites
- Python 3.9+
- Virtual Environment (Recommended)

### Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/thenishantraj/fraudlens-ai]
   cd fraudlens-ai
   '''
