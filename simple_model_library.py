#!/usr/bin/env python3
"""
Simple Model Library for STL Analyzer
====================================

This adds model selection to your pricing GUI.
"""

import os
import json

def scan_available_models():
    """Scan for available pricing models"""
    print("📚 Scanning for available pricing models...")
    
    available_models = {}
    
    # Define model patterns to look for
    model_patterns = [
        {
            'files': ['high_performance_random_forest_model.pkl', 'high_performance_random_forest_metadata.json'],
            'name': '🚀 High Performance RF (R² ~0.949)',
            'type': 'RandomForest',
            'id': 'high_performance'
        },
        {
            'files': ['random_forest_project_aware_with_machine_color.pkl', 'random_forest_project_aware_with_machine_color_scaler.pkl', 'random_forest_project_aware_with_machine_color_metadata.json'],
            'name': '🌲 RF v2.0 (Machine/Color)',
            'type': 'RandomForest',
            'id': 'rf_machine_color'
        },
        {
            'files': ['improved_random_forest_model.pkl', 'improved_random_forest_metadata.json'],
            'name': '📈 Improved RF Model',
            'type': 'RandomForest', 
            'id': 'improved_rf'
        },
        {
            'files': ['random_forest_project_aware_model.pkl', 'random_forest_project_aware_scaler.pkl', 'random_forest_project_aware_metadata.json'],
            'name': '🎯 Project-Aware RF',
            'type': 'RandomForest',
            'id': 'rf_project_aware'
        },
        {
            'files': ['enhanced_lasso_model.pkl', 'enhanced_feature_scaler.pkl', 'enhanced_model_metadata.json'],
            'name': '🎯 Enhanced Lasso',
            'type': 'Lasso',
            'id': 'enhanced_lasso'
        }
    ]
    
    # Check which models are available
    for pattern in model_patterns:
        if all(os.path.exists(f) for f in pattern['files']):
            try:
                # Load metadata to get performance info
                metadata_file = next(f for f in pattern['files'] if f.endswith('_metadata.json'))
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extract performance metrics
                performance = metadata.get('performance', {})
                r2 = performance.get('r2', metadata.get('r2', 0))
                mae = performance.get('mae', metadata.get('mae', 0))
                
                available_models[pattern['id']] = {
                    'name': pattern['name'],
                    'type': pattern['type'],
                    'files': pattern['files'],
                    'metadata': metadata,
                    'r2': r2,
                    'mae': mae,
                    'features': len(metadata.get('features', metadata.get('feature_names', []))),
                    'display_name': f"{pattern['name']} (R²={r2:.3f}, MAE=€{mae:.2f})"
                }
                print(f"   ✅ Found: {pattern['name']} - R²={r2:.3f}, MAE=€{mae:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error loading {pattern['name']}: {e}")
    
    print(f"📚 Found {len(available_models)} available models")
    return available_models

if __name__ == "__main__":
    # Test the model scanner
    models = scan_available_models()
    
    print("\n🎯 SUMMARY OF AVAILABLE MODELS:")
    print("=" * 50)
    
    if models:
        # Sort by R² score (best first)
        sorted_models = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        for i, (model_id, model_info) in enumerate(sorted_models, 1):
            print(f"{i}. {model_info['name']}")
            print(f"   Type: {model_info['type']}")
            print(f"   R² Score: {model_info['r2']:.4f}")
            print(f"   MAE: €{model_info['mae']:.2f}")
            print(f"   Features: {model_info['features']}")
            
            # Check for machine/color support
            feature_names = model_info['metadata'].get('features', model_info['metadata'].get('feature_names', []))
            has_machine = any('machine' in str(f).lower() for f in feature_names)
            has_color = any('color' in str(f).lower() or 'colour' in str(f).lower() for f in feature_names)
            
            if has_machine or has_color:
                support = []
                if has_machine: support.append("🏭 Machine")
                if has_color: support.append("🎨 Color")
                print(f"   Special Features: {', '.join(support)}")
            
            print()
        
        print(f"✅ RECOMMENDATION: Use '{sorted_models[0][1]['name']}' (best R² score)")
        
        # Show the encoding issue for machine/color models
        for model_id, model_info in sorted_models:
            if 'machine' in model_info['name'].lower():
                print(f"\n⚠️  IMPORTANT: {model_info['name']} uses different encoding:")
                encoders = model_info['metadata'].get('encoders', {})
                if encoders:
                    print("   Expected encoding:")
                    for category, values in encoders.items():
                        print(f"   • {category}: {values} → LabelEncoder (0, 1, 2...)")
                    print("   Current GUI uses: One-hot encoding (Machine_P396=1, Machine_Formiga=0)")
                    print("   → This is why machine selection doesn't change prices much!")
                break
    else:
        print("❌ No models found!")
        print("🔍 Make sure model files (.pkl and .json) are in the current directory")
    
    print("\n💡 TO ADD MODEL SELECTION TO GUI:")
    print("1. The combo box is already there (pricing_model_combo)")
    print("2. It just needs to be populated with available models")
    print("3. And connected to proper model loading functions") 