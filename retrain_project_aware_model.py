#!/usr/bin/env python3
"""
Retrain Random Forest Model with Project-Level Features
This script retrains the Random Forest model using the full project context
from the correct totals 3.csv file.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the CSV data with project-level features"""
    print("📊 Loading project data from CSV...")
    
    # Load the CSV file with semicolon separator
    df = pd.read_csv('correct totals 3.csv', sep=';')
    print(f"✅ Loaded {len(df)} rows from CSV")
    
    # Display available columns
    print(f"📋 Available columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Define target column - looking for price columns
    target_column = None
    possible_targets = ['Net price / part', 'net price * Quantity', 'Total Price', 'Total_Price', 'price']
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break
    
    if target_column is None:
        print("❌ Could not find price column. Available columns:")
        print(df.columns.tolist())
        return None, None, None
    
    print(f"🎯 Using target column: {target_column}")
    
    # Define feature columns (exclude target and non-numeric columns)
    exclude_columns = [target_column, 'Project ID', 'Part Name', 'Material', 'Machine', 'Colour']
    feature_columns = []
    
    for col in df.columns:
        if col not in exclude_columns:
            try:
                # Try to convert to numeric with more flexible handling
                test_series = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                if not test_series.isna().all():  # If at least some values are numeric
                    feature_columns.append(col)
                    print(f"   ✅ Added numeric column: {col}")
                else:
                    print(f"   ❌ Skipping non-numeric column: {col}")
            except Exception as e:
                print(f"   ❌ Error with column {col}: {e}")
    
    print(f"🔧 Using {len(feature_columns)} feature columns:")
    for i, col in enumerate(feature_columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Prepare features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Convert to numeric, handling any remaining issues
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col].astype(str).str.replace(',', '.'), errors='coerce')
    y = pd.to_numeric(y.astype(str).str.replace(',', '.'), errors='coerce')
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(y.median())
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Remove rows where target is 0 or negative (invalid prices)
    valid_mask = (y > 0) & (~y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"📈 Data shape after cleaning: {X.shape}")
    print(f"💰 Price range: €{y.min():.2f} - €{y.max():.2f}")
    print(f"💰 Average price: €{y.mean():.2f}")
    
    # Show project-level features
    project_features = [col for col in feature_columns if 'Project_Total' in col or 'Total_Parts' in col]
    print(f"🏗️ Project-level features found: {len(project_features)}")
    for feature in project_features:
        print(f"   • {feature}")
    
    return X, y, feature_columns

def create_enhanced_features(X):
    """Create additional engineered features"""
    print("🔧 Creating enhanced features...")
    
    X_enhanced = X.copy()
    
    # Log transformations for volume-related features
    volume_cols = [col for col in X.columns if 'volume' in col.lower() or 'Volume' in col]
    for col in volume_cols:
        if (X[col] > 0).any():
            X_enhanced[f'log_{col}'] = np.log1p(X[col])
    
    # Surface area log transformation
    surface_cols = [col for col in X.columns if 'surface' in col.lower() or 'Surface' in col]
    for col in surface_cols:
        if (X[col] > 0).any():
            X_enhanced[f'log_{col}'] = np.log1p(X[col])
    
    # Ratios and efficiency metrics
    if 'volume' in X.columns and 'surface_area' in X.columns:
        X_enhanced['volume_to_surface_ratio'] = X['volume'] / (X['surface_area'] + 1e-6)
    
    if 'convex_hull_volume' in X.columns and 'volume' in X.columns:
        X_enhanced['convex_hull_efficiency'] = X['volume'] / (X['convex_hull_volume'] + 1e-6)
    
    if 'shrinkwrap_volume' in X.columns and 'volume' in X.columns:
        X_enhanced['shrinkwrap_efficiency'] = X['volume'] / (X['shrinkwrap_volume'] + 1e-6)
    
    # Waste calculations
    if 'bb_volume' in X.columns and 'volume' in X.columns:
        X_enhanced['waste'] = X['bb_volume'] - X['volume']
        X_enhanced['waste_ratio'] = X_enhanced['waste'] / (X['volume'] + 1e-6)
    
    # Dimensional features
    dim_cols = [col for col in X.columns if col in ['x', 'y', 'z', 'Max D', 'Min D']]
    if len(dim_cols) >= 2:
        max_dim = X[dim_cols].max(axis=1)
        min_dim = X[dim_cols].min(axis=1)
        X_enhanced['max_min_ratio_enhanced'] = max_dim / (min_dim + 1e-6)
        X_enhanced['dimension_spread'] = max_dim - min_dim
        X_enhanced['dimension_product'] = X[dim_cols].prod(axis=1)
    
    # Complexity metrics
    if 'surface_area' in X.columns and 'volume' in X.columns:
        X_enhanced['geometric_complexity'] = X['surface_area'] / (X['volume'] ** (2/3) + 1e-6)
    
    print(f"✅ Enhanced features created. New shape: {X_enhanced.shape}")
    return X_enhanced

def train_project_aware_model(X, y, feature_names):
    """Train Random Forest model with project-level features"""
    print("🌲 Training Project-Aware Random Forest Model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='log2',
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    
    print("🔄 Training model...")
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    oob_score = rf_model.oob_score_
    
    print(f"\n🎯 Model Performance:")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print(f"   Training MAE: €{train_mae:.2f}")
    print(f"   Test MAE: €{test_mae:.2f}")
    print(f"   Training RMSE: €{train_rmse:.2f}")
    print(f"   Test RMSE: €{test_rmse:.2f}")
    print(f"   OOB Score: {oob_score:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n🔥 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(sorted_importance.items())[:10], 1):
        print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    # Check for overfitting
    overfitting_risk = "High" if (train_r2 - test_r2) > 0.1 else "Medium" if (train_r2 - test_r2) > 0.05 else "Low"
    print(f"\n⚠️ Overfitting Risk: {overfitting_risk}")
    
    return rf_model, scaler, {
        'train_r2_score': train_r2,
        'test_r2_score': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'oob_score': oob_score,
        'overfitting_risk': overfitting_risk
    }, sorted_importance

def save_project_aware_model(model, scaler, feature_names, performance_metrics, feature_importance):
    """Save the project-aware model and metadata"""
    print("💾 Saving Project-Aware Random Forest Model...")
    
    # Save model and scaler
    joblib.dump(model, 'random_forest_project_aware_model.pkl')
    joblib.dump(scaler, 'random_forest_project_aware_scaler.pkl')
    
    # Create metadata
    metadata = {
        'model_type': 'ProjectAwareRandomForestRegressor',
        'creation_date': datetime.now().isoformat(),
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_parameters': model.get_params(),
        'performance_metrics': performance_metrics,
        'feature_importance': feature_importance,
        'project_features_included': True,
        'description': 'Random Forest model trained with full project-level context including project totals and part relationships'
    }
    
    # Save metadata
    with open('random_forest_project_aware_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Project-Aware Model saved successfully!")
    print(f"   Model file: random_forest_project_aware_model.pkl")
    print(f"   Scaler file: random_forest_project_aware_scaler.pkl")
    print(f"   Metadata file: random_forest_project_aware_metadata.json")
    
    return metadata

def main():
    """Main training pipeline"""
    print("🚀 Starting Project-Aware Random Forest Training")
    print("=" * 60)
    
    # Load and prepare data
    X, y, feature_columns = load_and_prepare_data()
    if X is None:
        return
    
    # Create enhanced features
    X_enhanced = create_enhanced_features(X)
    enhanced_feature_names = list(X_enhanced.columns)
    
    # Train model
    model, scaler, performance_metrics, feature_importance = train_project_aware_model(
        X_enhanced, y, enhanced_feature_names
    )
    
    # Save model
    metadata = save_project_aware_model(
        model, scaler, enhanced_feature_names, performance_metrics, feature_importance
    )
    
    print("\n" + "=" * 60)
    print("🎉 Project-Aware Random Forest Training Complete!")
    print(f"🎯 Final Test R²: {performance_metrics['test_r2_score']:.4f}")
    print(f"💰 Final Test MAE: €{performance_metrics['test_mae']:.2f}")
    print(f"🌲 Features with project context: {len(enhanced_feature_names)}")
    
    # Show project-level features
    project_features = [f for f in enhanced_feature_names if any(keyword in f.lower() for keyword in ['project', 'total_parts', 'total_volume', 'total_surface'])]
    if project_features:
        print(f"\n🏗️ Project-Level Features Included:")
        for feature in project_features:
            importance = feature_importance.get(feature, 0)
            print(f"   • {feature}: {importance:.4f}")
    
    print("\n✅ Ready to use in STL Analyzer!")

if __name__ == "__main__":
    main() 