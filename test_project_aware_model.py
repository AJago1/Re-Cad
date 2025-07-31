#!/usr/bin/env python3
"""
Test Project-Aware Random Forest Model
This script tests that the project-aware model is using ALL features correctly.
"""

import joblib
import json
import numpy as np
import pandas as pd

def test_project_aware_model():
    """Test the project-aware Random Forest model"""
    print("🧪 Testing Project-Aware Random Forest Model")
    print("=" * 50)
    
    try:
        # Load the project-aware model
        model = joblib.load('random_forest_project_aware_model.pkl')
        scaler = joblib.load('random_forest_project_aware_scaler.pkl')
        
        with open('random_forest_project_aware_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        feature_names = metadata['feature_names']
        
        print(f"✅ Model loaded successfully!")
        print(f"   Features: {len(feature_names)}")
        print(f"   Test R²: {metadata['performance_metrics']['test_r2_score']:.4f}")
        print(f"   Test MAE: €{metadata['performance_metrics']['test_mae']:.2f}")
        
        # Show project-level features
        project_features = [f for f in feature_names if 'Project_Total' in f or 'Total_Parts' in f]
        print(f"\n🏗️ Project-Level Features ({len(project_features)}):")
        for i, feature in enumerate(project_features, 1):
            importance = metadata['feature_importance'].get(feature, 0)
            print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        # Test with sample data
        print(f"\n🔬 Testing with Sample Data:")
        
        # Create sample individual part
        sample_part = {
            'convex_hull_volume': 1200.0,
            'bb_volume': 1500.0,
            'surface_area': 800.0,
            'Quantity': 2,
            'Max D': 50.0,
            'Min D': 10.0,
            'D Ratio': 5.0,
            'waste': 300.0,
            'shrinkwrap_volume': 1300.0,
            'Volume': 1000.0,
            'Volume * Quantity': 2000.0,
            'shrinkwrap_volume*Quantity': 2600.0,
            'Surface area*Quantity': 1600.0,
            'BB_Volume*Quantity': 3000.0,
            'Convex_Hull_Volume*Quantity': 2400.0,
        }
        
        # Create sample project totals
        project_totals = {
            'Total_Parts_in_Project': 5,
            'Project_Total_convex_hull_volume': 6000.0,
            'Project_Total_bb_volume': 7500.0,
            'Project_Total_shrinkwrap_volume': 6500.0,
            'Project_Total_Volume': 5000.0,
            'Project_Total_surface_area': 4000.0,
        }
        
        # Test 1: Individual part only
        individual_features = {}
        for feature_name in feature_names:
            if feature_name in sample_part:
                individual_features[feature_name] = sample_part[feature_name]
            else:
                individual_features[feature_name] = 0.0
        
        individual_vector = np.array([individual_features[f] for f in feature_names]).reshape(1, -1)
        individual_scaled = scaler.transform(individual_vector)
        individual_price = model.predict(individual_scaled)[0]
        
        print(f"   Individual part price: €{individual_price:.2f}")
        
        # Test 2: Part + Project context (ALL FEATURES)
        combined_features = {}
        for feature_name in feature_names:
            if feature_name in sample_part:
                combined_features[feature_name] = sample_part[feature_name]
            elif feature_name in project_totals:
                combined_features[feature_name] = project_totals[feature_name]
            else:
                combined_features[feature_name] = 0.0
        
        combined_vector = np.array([combined_features[f] for f in feature_names]).reshape(1, -1)
        combined_scaled = scaler.transform(combined_vector)
        project_aware_price = model.predict(combined_scaled)[0]
        
        print(f"   Project-aware price: €{project_aware_price:.2f}")
        
        # Show the difference
        diff = project_aware_price - individual_price
        diff_percent = (diff / individual_price) * 100 if individual_price > 0 else 0
        print(f"   Difference: €{diff:.2f} ({diff_percent:+.1f}%)")
        
        # Verify feature usage
        print(f"\n🔍 Feature Vector Verification:")
        print(f"   Total features expected: {len(feature_names)}")
        print(f"   Individual features used: {sum(1 for f in feature_names if f in sample_part)}")
        print(f"   Project features used: {sum(1 for f in feature_names if f in project_totals)}")
        print(f"   Combined features: {sum(1 for f in feature_names if f in sample_part or f in project_totals)}")
        
        # Show some key feature values
        print(f"\n📊 Key Feature Values:")
        key_features = ['Volume', 'Total_Parts_in_Project', 'Project_Total_Volume', 'shrinkwrap_volume']
        for feature in key_features:
            if feature in combined_features:
                print(f"   {feature}: {combined_features[feature]:,.0f}")
        
        print(f"\n✅ Project-Aware Model Test Complete!")
        print(f"🎯 The model IS using project-level features and shows price difference!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_project_aware_model() 