#!/usr/bin/env python3
"""
Retrain Random Forest Model with Machine and Color Features
==========================================================

This script retrains the Random Forest model to include Machine and Color features
from the CSV training data, fixing the issue where these categorical features
were missing from the original model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import os

def load_and_prepare_data():
    """Load CSV data and prepare for training with Machine and Color features"""
    print("📊 Loading training data...")
    df = pd.read_csv('correct totals 3.csv', sep=';', decimal=',')
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"🔍 Columns: {list(df.columns)}")
    
    # Check Machine and Color values
    print(f"\n🏭 Machine values: {df['Machine'].unique()}")
    print(f"🎨 Color values: {df['Colour'].unique()}")
    
    return df

def engineer_features(df):
    """Create all engineered features like the original model"""
    print("🔧 Engineering features...")
    
    # Create copy for modifications
    df_features = df.copy()
    
    # Basic features (these already exist in CSV)
    basic_features = [
        'convex_hull_volume', 'bb_volume', 'surface_area', 'Quantity',
        'Max D', 'Min D', 'D Ratio', 'waste', 'shrinkwrap_volume', 'Volume',
        'Volume * Quantity ', 'shrinkwrap_volume*Quantity', 'Surface area*Quantity',
        'BB_Volume*Quantity', 'Convex_Hull_Volume*Quantity', 'Total_Parts_in_Project',
        'Project_Total_convex_hull_volume', 'Project_Total_bb_volume',
        'Project_Total_shrinkwrap_volume', 'Project_Total_Volume', 'Project_Total_surface_area'
    ]
    
    # Add logarithmic features
    log_base_features = [
        'Volume', 'surface_area', 'shrinkwrap_volume', 'bb_volume', 'convex_hull_volume',
        'Project_Total_Volume', 'Project_Total_surface_area', 'Project_Total_shrinkwrap_volume',
        'Project_Total_bb_volume', 'Project_Total_convex_hull_volume'
    ]
    
    for feature in log_base_features:
        if feature in df_features.columns:
            df_features[f'log_{feature}'] = np.log1p(df_features[feature].fillna(0))
    
    # Add log quantity interactions
    log_quantity_features = [
        'log_Volume * Quantity ', 'log_shrinkwrap_volume*Quantity', 
        'log_Surface area*Quantity', 'log_BB_Volume*Quantity', 'log_Convex_Hull_Volume*Quantity'
    ]
    
    if 'Quantity' in df_features.columns:
        for base_feature in ['Volume', 'surface_area', 'shrinkwrap_volume', 'bb_volume', 'convex_hull_volume']:
            if base_feature in df_features.columns:
                log_feature = f'log_{base_feature}'
                quantity_feature = f'log_{base_feature} * Quantity '
                if log_feature in df_features.columns:
                    df_features[quantity_feature] = df_features[log_feature] * df_features['Quantity']
    
    # Machine encoding (one-hot)
    if 'Machine' in df_features.columns:
        print(f"🏭 Encoding Machine: {df_features['Machine'].unique()}")
        df_features['Machine_EOS_P396'] = (df_features['Machine'].str.contains('P396', na=False)).astype(int)
        df_features['Machine_Formiga'] = (df_features['Machine'].str.contains('Formiga', na=False)).astype(int)
    
    # Color encoding (one-hot for main colors)
    if 'Colour' in df_features.columns:
        print(f"🎨 Encoding Colors: {df_features['Colour'].unique()}")
        df_features['Colour_Black'] = (df_features['Colour'] == 'Black').astype(int)
        df_features['Colour_White'] = (df_features['Colour'] == 'White').astype(int)
        df_features['Colour_Grey'] = (df_features['Colour'] == 'Grey').astype(int)
        df_features['Colour_Orange'] = (df_features['Colour'] == 'Orange').astype(int)
        df_features['Colour_Blue'] = (df_features['Colour'] == 'Blue').astype(int)
        df_features['Colour_Green'] = (df_features['Colour'] == 'Green').astype(int)
        df_features['Colour_Yellow'] = (df_features['Colour'] == 'Yellow').astype(int)
        df_features['Colour_Red'] = (df_features['Colour'] == 'Red').astype(int)
    
    print(f"✅ Feature engineering complete. Total columns: {len(df_features.columns)}")
    return df_features

def select_features_and_target(df):
    """Select final feature set for training"""
    print("🎯 Selecting features for training...")
    
    # Target variable
    target = 'Net price / part'
    
    # Feature selection (exclude problematic features)
    exclude_features = [
        'Project ID', 'Part Name', 'Material', 
        'Net price / part', 'net price * Quantity',  # Target and leakage
        'Machine', 'Colour'  # Original categorical (we use encoded versions)
    ]
    
    # Select all numeric features
    feature_columns = [col for col in df.columns 
                      if col not in exclude_features 
                      and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    print(f"📋 Selected {len(feature_columns)} features for training")
    print(f"🎯 Target variable: {target}")
    
    # Handle missing values
    X = df[feature_columns].fillna(0)
    y = df[target]
    
    # Remove rows with missing target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"✅ Final dataset: {len(X)} samples, {len(feature_columns)} features")
    
    return X, y, feature_columns

def train_random_forest(X, y, feature_names):
    """Train Random Forest with proper hyperparameters"""
    print("🌲 Training Random Forest model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = rf_model.predict(X_train_scaled)
    test_pred = rf_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"📊 Training Results:")
    print(f"   Train R² = {train_r2:.4f}, MAE = €{train_mae:.2f}")
    print(f"   Test  R² = {test_r2:.4f}, MAE = €{test_mae:.2f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Check Machine/Color importance
    machine_color_features = [f for f in feature_names if 'Machine_' in f or 'Colour_' in f]
    if machine_color_features:
        print(f"\n🏭🎨 Machine/Color Feature Importance:")
        mc_importance = feature_importance[feature_importance['feature'].isin(machine_color_features)]
        for idx, row in mc_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return rf_model, scaler, feature_importance

def save_model_and_metadata(model, scaler, feature_names, feature_importance):
    """Save the trained model and metadata"""
    print("💾 Saving model and metadata...")
    
    # Save model and scaler
    joblib.dump(model, 'random_forest_project_aware_with_machine_color.pkl')
    joblib.dump(scaler, 'random_forest_project_aware_with_machine_color_scaler.pkl')
    
    # Create metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'version': '2.0_with_machine_color',
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'includes_machine_color': True,
        'machine_features': [f for f in feature_names if 'Machine_' in f],
        'color_features': [f for f in feature_names if 'Colour_' in f],
        'feature_importance': {
            row['feature']: float(row['importance']) 
            for _, row in feature_importance.iterrows()
        },
        'training_info': {
            'algorithm': 'Random Forest',
            'n_estimators': 100,
            'max_depth': 20,
            'data_source': 'correct totals 3.csv',
            'target_variable': 'Net price / part'
        }
    }
    
    # Save metadata
    with open('random_forest_project_aware_with_machine_color_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved as: random_forest_project_aware_with_machine_color.pkl")
    print(f"✅ Scaler saved as: random_forest_project_aware_with_machine_color_scaler.pkl")  
    print(f"✅ Metadata saved as: random_forest_project_aware_with_machine_color_metadata.json")
    print(f"🎯 Model expects {len(feature_names)} features including Machine and Color")

def main():
    """Main training pipeline"""
    print("🚀 Starting Random Forest Retraining with Machine and Color Features")
    print("=" * 70)
    
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Select features and target
        X, y, feature_names = select_features_and_target(df_features)
        
        # Train model
        model, scaler, feature_importance = train_random_forest(X, y, feature_names)
        
        # Save everything
        save_model_and_metadata(model, scaler, feature_names, feature_importance)
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Random Forest model retrained with Machine and Color features")
        print("🔧 Update your application to use the new model files")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    main() 