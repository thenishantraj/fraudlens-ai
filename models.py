"""
ML models for anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
import joblib
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class PriceAnomalyDetector:
    """Detect price anomalies using Isolation Forest and statistical methods"""
    
    def __init__(self, contamination=0.1, random_state=42):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.price_stats = {}
        
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess features for anomaly detection"""
        df = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['category', 'department', 'procurement_method']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].fillna('Unknown'))
        
        # Calculate derived features
        if 'unit_price' in df.columns and 'quantity' in df.columns:
            df['log_unit_price'] = np.log1p(df['unit_price'])
            df['log_quantity'] = np.log1p(df['quantity'])
            df['log_total_amount'] = np.log1p(df['total_amount'])
            
            # Price per unit statistics by category
            if 'category' in df.columns:
                for category in df['category'].unique():
                    if category in self.price_stats:
                        category_mask = df['category'] == category
                        df.loc[category_mask, 'price_zscore'] = (
                            df.loc[category_mask, 'unit_price'] - self.price_stats[category]['mean']
                        ) / self.price_stats[category]['std']
        
        # Select features for training
        feature_cols = [
            'log_unit_price', 'log_quantity', 'log_total_amount',
            'category', 'department', 'procurement_method'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Fill NaN values
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df[feature_cols].values
    
    def fit(self, df: pd.DataFrame):
        """Fit the anomaly detection model"""
        # Calculate price statistics by category
        if 'category' in df.columns and 'unit_price' in df.columns:
            for category in df['category'].unique():
                category_prices = df[df['category'] == category]['unit_price']
                self.price_stats[category] = {
                    'mean': category_prices.mean(),
                    'std': category_prices.std(),
                    'median': category_prices.median()
                }
        
        # Preprocess and fit
        X = self.preprocess_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        
        return self
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores"""
        X = self.preprocess_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly predictions (-1 for anomalies, 1 for normal)
        predictions = self.isolation_forest.predict(X_scaled)
        
        # Convert to 0 (normal) and 1 (anomaly)
        anomaly_labels = np.where(predictions == -1, 1, 0)
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.isolation_forest.decision_function(X_scaled)
        
        # Normalize scores to 0-100 range
        risk_scores = self._normalize_scores(scores)
        
        return anomaly_labels, risk_scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize anomaly scores to 0-100 range"""
        # Isolation Forest returns negative scores for anomalies
        # We want higher numbers = higher risk
        scores_normalized = -scores  # Invert so positive = anomalous
        
        # Scale to 0-100
        min_score = scores_normalized.min()
        max_score = scores_normalized.max()
        
        if max_score > min_score:
            risk_scores = 100 * (scores_normalized - min_score) / (max_score - min_score)
        else:
            risk_scores = np.zeros_like(scores_normalized)
        
        return risk_scores
    
    def get_anomaly_explanations(self, df: pd.DataFrame, anomaly_indices: List[int]) -> List[str]:
        """Generate explanations for detected anomalies"""
        explanations = []
        
        for idx in anomaly_indices:
            record = df.iloc[idx]
            explanations.append(self._explain_anomaly(record))
        
        return explanations
    
    def _explain_anomaly(self, record: pd.Series) -> str:
        """Generate explanation for a single anomaly"""
        explanations = []
        
        # Check price anomalies
        if 'category' in record and 'unit_price' in record:
            category = record['category']
            price = record['unit_price']
            
            if category in self.price_stats:
                stats = self.price_stats[category]
                if stats['std'] > 0:
                    z_score = (price - stats['mean']) / stats['std']
                    
                    if z_score > 3:
                        explanations.append(f"Price is {z_score:.1f} standard deviations above category average")
                    elif price > stats['median'] * 2:
                        explanations.append(f"Price is {price/stats['median']:.1f}x the category median")
        
        # Check quantity anomalies
        if 'quantity' in record:
            if record['quantity'] > 1000:
                explanations.append("Unusually large quantity")
            elif record['quantity'] == 1 and record['unit_price'] > 10000:
                explanations.append("Single item with very high price")
        
        # Check total amount
        if 'total_amount' in record:
            if record['total_amount'] > 1000000:
                explanations.append("Total amount exceeds $1M threshold")
        
        return "; ".join(explanations) if explanations else "Unusual pattern detected"

class SimilarityEngine:
    """Detect document similarity for bid-rigging detection"""
    
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = None
        self.document_ids = None
        
    def fit_transform(self, documents: List[str], document_ids: List[str]) -> np.ndarray:
        """Fit vectorizer and transform documents"""
        self.document_ids = document_ids
        self.document_vectors = self.vectorizer.fit_transform(documents)
        return self.document_vectors
    
    def detect_duplicates(self) -> List[Tuple[str, str, float]]:
        """Detect duplicate or highly similar documents"""
        if self.document_vectors is None:
            raise ValueError("Model not fitted. Call fit_transform first.")
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(self.document_vectors)
        
        duplicates = []
        n_docs = len(self.document_ids)
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                similarity = similarity_matrix[i, j]
                if similarity > self.similarity_threshold:
                    duplicates.append((
                        self.document_ids[i],
                        self.document_ids[j],
                        similarity
                    ))
        
        return duplicates
    
    def get_similarity_scores(self, document: str) -> np.ndarray:
        """Get similarity scores for a new document against trained documents"""
        if self.document_vectors is None:
            raise ValueError("Model not fitted. Call fit_transform first.")
        
        new_vector = self.vectorizer.transform([document])
        similarities = cosine_similarity(new_vector, self.document_vectors)
        return similarities[0]

class VendorRiskAnalyzer:
    """Analyze vendor patterns for suspicious behavior"""
    
    def __init__(self):
        self.vendor_stats = {}
        
    def analyze_vendors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk metrics for each vendor"""
        vendor_metrics = []
        
        for vendor in df['vendor_name'].unique():
            vendor_data = df[df['vendor_name'] == vendor]
            
            metrics = {
                'vendor_name': vendor,
                'total_contracts': len(vendor_data),
                'total_amount': vendor_data['total_amount'].sum(),
                'avg_unit_price': vendor_data['unit_price'].mean(),
                'std_unit_price': vendor_data['unit_price'].std(),
                'unique_departments': vendor_data['department'].nunique(),
                'avg_risk_score': vendor_data.get('risk_score', 0).mean() if 'risk_score' in vendor_data.columns else 0,
                'high_risk_count': sum(vendor_data.get('risk_level', '') == 'HIGH') if 'risk_level' in vendor_data.columns else 0
            }
            
            # Calculate concentration metrics
            if metrics['total_contracts'] > 1:
                metrics['contract_concentration'] = (
                    vendor_data['total_amount'].std() / vendor_data['total_amount'].mean()
                )
            else:
                metrics['contract_concentration'] = 0
            
            vendor_metrics.append(metrics)
        
        vendor_df = pd.DataFrame(vendor_metrics)
        
        # Calculate vendor risk scores
        vendor_df['vendor_risk_score'] = self._calculate_vendor_risk(vendor_df)
        
        return vendor_df
    
    def _calculate_vendor_risk(self, vendor_df: pd.DataFrame) -> np.ndarray:
        """Calculate risk score for vendors based on multiple factors"""
        # Normalize metrics
        metrics_to_normalize = [
            'total_contracts',
            'total_amount',
            'contract_concentration',
            'avg_risk_score',
            'high_risk_count'
        ]
        
        risk_score = np.zeros(len(vendor_df))
        
        for metric in metrics_to_normalize:
            if metric in vendor_df.columns:
                values = vendor_df[metric].fillna(0).values
                if values.max() > values.min():
                    normalized = (values - values.min()) / (values.max() - values.min())
                    risk_score += normalized
        
        # Normalize final score to 0-100
        if risk_score.max() > 0:
            risk_score = 100 * risk_score / risk_score.max()
        
        return risk_score

class FraudDetectionPipeline:
    """Complete fraud detection pipeline"""
    
    def __init__(self):
        self.price_detector = PriceAnomalyDetector()
        self.similarity_engine = SimilarityEngine()
        self.vendor_analyzer = VendorRiskAnalyzer()
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Fit all models"""
        # Fit price anomaly detector
        self.price_detector.fit(df)
        
        # Fit similarity engine on descriptions
        descriptions = df['description'].fillna('').astype(str).tolist()
        document_ids = df['tender_id'].tolist()
        self.similarity_engine.fit_transform(descriptions, document_ids)
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete fraud detection pipeline"""
        if not self.is_fitted:
            self.fit(df)
        
        df_result = df.copy()
        
        # 1. Detect price anomalies
        anomaly_labels, risk_scores = self.price_detector.predict(df)
        df_result['price_anomaly'] = anomaly_labels
        df_result['risk_score'] = risk_scores
        
        # 2. Detect duplicate documents
        descriptions = df_result['description'].fillna('').astype(str).tolist()
        document_ids = df_result['tender_id'].tolist()
        
        self.similarity_engine.fit_transform(descriptions, document_ids)
        duplicates = self.similarity_engine.detect_duplicates()
        
        # Mark duplicate documents
        duplicate_tenders = set()
        for doc1, doc2, similarity in duplicates:
            duplicate_tenders.add(doc1)
            duplicate_tenders.add(doc2)
        
        df_result['document_similarity'] = df_result['tender_id'].isin(duplicate_tenders).astype(int)
        
        # Adjust risk score for document duplicates
        df_result.loc[df_result['document_similarity'] == 1, 'risk_score'] = (
            df_result.loc[df_result['document_similarity'] == 1, 'risk_score'] * 1.3
        ).clip(0, 100)
        
        # 3. Analyze vendor risks
        vendor_risks = self.vendor_analyzer.analyze_vendors(df_result)
        
        # Merge vendor risk scores back to main dataframe
        vendor_risk_map = dict(zip(vendor_risks['vendor_name'], vendor_risks['vendor_risk_score']))
        df_result['vendor_risk_score'] = df_result['vendor_name'].map(vendor_risk_map).fillna(0)
        
        # Adjust risk score based on vendor risk
        df_result['risk_score'] = (
            df_result['risk_score'] * 0.7 + df_result['vendor_risk_score'] * 0.3
        )
        
        # 4. Calculate final risk levels
        df_result['risk_level'] = pd.cut(
            df_result['risk_score'],
            bins=[-1, 30, 60, 80, 101],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )
        
        # 5. Generate explanations
        anomaly_indices = df_result[df_result['price_anomaly'] == 1].index.tolist()
        explanations = self.price_detector.get_anomaly_explanations(df_result, anomaly_indices)
        
        # Create explanations column
        df_result['explanation'] = ''
        for idx, explanation in zip(anomaly_indices, explanations):
            df_result.at[idx, 'explanation'] = explanation
        
        # Add document similarity explanations
        duplicate_indices = df_result[df_result['document_similarity'] == 1].index
        for idx in duplicate_indices:
            current_explanation = df_result.at[idx, 'explanation']
            if current_explanation:
                df_result.at[idx, 'explanation'] = current_explanation + "; Document highly similar to other tenders"
            else:
                df_result.at[idx, 'explanation'] = "Document highly similar to other tenders (possible bid-rigging)"
        
        return df_result
    
    def save(self, filepath: str):
        """Save the trained pipeline"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a trained pipeline"""
        return joblib.load(filepath)