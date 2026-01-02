"""
Synthetic procurement data generator with built-in anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict

class ProcurementDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        self.vendors = [
            'Global Suppliers Inc.', 'Tech Solutions Ltd.', 'Office Depot Corp',
            'Construction Masters', 'Medical Supplies Co', 'EduTech Partners',
            'Food Services Intl', 'Transport Logistics', 'Clean Energy Systems',
            'Security First Ltd', 'Quick Delivery Inc', 'Quality Builders'
        ]
        
        self.departments = [
            'Health', 'Education', 'Defense', 'Transportation',
            'Agriculture', 'Energy', 'Interior', 'Finance'
        ]
        
        self.categories = [
            'Medical Equipment', 'IT Hardware', 'Office Supplies',
            'Construction Materials', 'Food Products', 'Vehicle Purchase',
            'Consulting Services', 'Security Equipment', 'Cleaning Services'
        ]
        
        self.locations = ['Capital City', 'North Region', 'South Region', 
                         'East Region', 'West Region', 'Central Region']
        
        # Common phrases for descriptions (some will be reused for similarity detection)
        self.description_templates = [
            "Procurement of {item} for {department} department {purpose}",
            "Supply and delivery of {item} to support {department} operations",
            "Purchase of {item} for the {location} regional office",
            "Acquisition of {item} for {department} program implementation",
            "Emergency procurement of {item} for {department} needs"
        ]
        
        self.items = [
            "Laptops", "Medical Masks", "Office Chairs", "Cement Bags",
            "Ambulance Vehicles", "Servers", "Desks", "Generators",
            "Security Cameras", "Cleaning Equipment", "School Books",
            "Agricultural Tools", "Uniforms", "Medicines"
        ]
        
        self.purposes = [
            "to enhance operational efficiency",
            "for departmental requirements",
            "as per annual procurement plan",
            "to address urgent needs",
            "for project implementation"
        ]
    
    def generate_normal_price(self, category):
        """Generate normal market prices for different categories"""
        base_prices = {
            'Medical Equipment': (5000, 50000),
            'IT Hardware': (800, 5000),
            'Office Supplies': (50, 500),
            'Construction Materials': (100, 10000),
            'Food Products': (10, 500),
            'Vehicle Purchase': (20000, 100000),
            'Consulting Services': (1000, 50000),
            'Security Equipment': (1000, 20000),
            'Cleaning Services': (500, 10000)
        }
        return np.random.uniform(*base_prices[category])
    
    def generate_record(self, record_id, include_anomaly=False):
        """Generate a single procurement record"""
        vendor = random.choice(self.vendors)
        department = random.choice(self.departments)
        category = random.choice(self.categories)
        location = random.choice(self.locations)
        item = random.choice(self.items)
        
        # Generate dates
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)
        random_days = random.randint(0, 365)
        tender_date = start_date + timedelta(days=random_days)
        delivery_date = tender_date + timedelta(days=random.randint(30, 180))
        
        # Generate description
        template = random.choice(self.description_templates)
        purpose = random.choice(self.purposes)
        description = template.format(
            item=item,
            department=department,
            purpose=purpose,
            location=location
        )
        
        # Generate quantities
        quantity = random.randint(1, 1000)
        
        # Generate price - introduce anomalies for some records
        base_price = self.generate_normal_price(category)
        
        if include_anomaly:
            # Different types of anomalies
            anomaly_type = random.choice(['price_inflation', 'duplicate_desc', 'suspicious_vendor'])
            
            if anomaly_type == 'price_inflation':
                # Inflate price by 3-10x
                inflation_factor = random.uniform(3, 10)
                unit_price = base_price * inflation_factor
                justification = "Urgent requirement with single source"
                
            elif anomaly_type == 'duplicate_desc':
                # Use same description as previous (will be handled in batch)
                unit_price = base_price * random.uniform(0.9, 1.1)
                justification = "Standard procurement process followed"
                
            elif anomaly_type == 'suspicious_vendor':
                # Vendor with suspicious pattern
                if vendor in ['Quick Delivery Inc', 'Global Suppliers Inc.']:
                    unit_price = base_price * random.uniform(2, 5)
                else:
                    unit_price = base_price * random.uniform(0.9, 1.1)
                justification = "Selected through competitive bidding"
        else:
            unit_price = base_price * random.uniform(0.8, 1.2)
            justification = "Competitive bidding process completed"
        
        amount = quantity * unit_price
        
        return {
            'tender_id': f'T{record_id:06d}',
            'vendor_name': vendor,
            'department': department,
            'category': category,
            'description': description,
            'quantity': quantity,
            'unit_price': round(unit_price, 2),
            'total_amount': round(amount, 2),
            'tender_date': tender_date.strftime('%Y-%m-%d'),
            'delivery_date': delivery_date.strftime('%Y-%m-%d'),
            'location': location,
            'justification': justification,
            'procurement_method': random.choice(['Open Tender', 'Limited Tender', 'Single Source', 'Emergency']),
            'contract_duration': random.randint(30, 365),
            'anomaly_type': 'price_inflation' if include_anomaly and anomaly_type == 'price_inflation' else 'normal'
        }
    
    def generate_dataset(self, n_records=200, anomaly_rate=0.15):
        """Generate dataset with anomalies"""
        records = []
        anomaly_count = int(n_records * anomaly_rate)
        
        # Generate some duplicate descriptions for bid-rigging simulation
        duplicate_descriptions = []
        for _ in range(3):
            template = random.choice(self.description_templates)
            duplicate_descriptions.append(
                template.format(
                    item=random.choice(self.items),
                    department=random.choice(self.departments),
                    purpose=random.choice(self.purposes),
                    location=random.choice(self.locations)
                )
            )
        
        for i in range(n_records):
            include_anomaly = i < anomaly_count
            
            record = self.generate_record(i, include_anomaly)
            
            # Force some duplicate descriptions
            if include_anomaly and random.random() < 0.3:
                record['description'] = random.choice(duplicate_descriptions)
            
            records.append(record)
        
        # Add a few exact duplicates for duplicate detection
        for _ in range(3):
            duplicate_record = records[random.randint(0, len(records)-1)].copy()
            duplicate_record['tender_id'] = f'T{len(records):06d}'
            duplicate_record['vendor_name'] = random.choice(self.vendors)
            records.append(duplicate_record)
        
        df = pd.DataFrame(records)
        return df

def save_sample_data():
    """Generate and save sample data"""
    generator = ProcurementDataGenerator()
    df = generator.generate_dataset(n_records=250, anomaly_rate=0.18)
    
    # Save to CSV
    df.to_csv('procurement_data.csv', index=False)
    print(f"Generated {len(df)} records. Saved to 'procurement_data.csv'")
    print(f"Anomalies in dataset: {df['anomaly_type'].value_counts().get('price_inflation', 0)}")
    
    return df

if __name__ == "__main__":
    df = save_sample_data()
    print("\nFirst 5 records:")
    print(df.head())
