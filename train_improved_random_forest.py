#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def train_improved_random_forest():
    print("🚀 TRAINING IMPROVED RANDOM FOREST MODEL")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('correct totals 4.csv', sep=';', decimal=',')
    print(f"📈 Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Fix machine names (merge P396 variants)
    df['Machine'] = df['Machine'].str.replace('EOS P396 ', 'EOS P396').str.strip()
    
    # Exclude price-related features (except target)
    price_features = ['Net price/mm3', 'net price * Quantity']  # Keep 'Net price / part' as target
    target = 'Net price / part'
    
    # Get features (exclude price-related and ID columns)
    exclude_cols = price_features + ['Project ID', 'Part Name']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"🎯 Target: {target}")
    print(f"🔧 Using {len(features)} features")
    print(f"❌ Excluded: {price_features}")
    
    # Clean data
    df_clean = df.copy()
    
    # Convert target to numeric
    if df_clean[target].dtype == 'object':
        df_clean[target] = pd.to_numeric(df_clean[target].str.replace(',', '.'), errors='coerce')
    else:
        df_clean[target] = pd.to_numeric(df_clean[target], errors='coerce')
    
    # Handle categorical and numeric features
    categorical_features = ['Machine', 'Colour', 'Material']
    numeric_features = [f for f in features if f not in categorical_features]
    
    # Convert numeric columns
    for feature in numeric_features:
        if df_clean[feature].dtype == 'object':
            df_clean[feature] = pd.to_numeric(df_clean[feature].str.replace(',', '.'), errors='coerce')
    
    # Encode categorical variables and store encoders
    encoders = {}
    for cat_feature in categorical_features:
        if cat_feature in df_clean.columns:
            le = LabelEncoder()
            df_clean[f'{cat_feature}_encoded'] = le.fit_transform(df_clean[cat_feature].fillna('Unknown'))
            encoders[cat_feature] = le
            print(f"🏷️ Encoded {cat_feature}: {list(le.classes_)}")
    
    # Update features list
    encoded_features = [f'{cat}_encoded' for cat in categorical_features if cat in df_clean.columns]
    final_features = numeric_features + encoded_features
    
    # Remove missing target rows and fill feature NAs
    df_clean = df_clean.dropna(subset=[target])
    for feature in final_features:
        if feature in df_clean.columns:
            df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
    
    # Remove outliers using IQR method
    y = df_clean[target].values
    Q1, Q3 = np.percentile(y, [25, 75])
    IQR = Q3 - Q1
    outlier_mask = (y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)
    
    print(f"🗑️ Removing {outlier_mask.sum()} outliers")
    df_clean = df_clean[~outlier_mask]
    
    print(f"✅ Final dataset: {len(df_clean)} rows, {len(final_features)} features")
    
    # Prepare data
    available_features = [f for f in final_features if f in df_clean.columns and not df_clean[f].isna().all()]
    X = df_clean[available_features].values
    y = df_clean[target].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train improved Random Forest model (optimized parameters)
    print(f"\n🔧 Training Improved Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,        # More trees for better performance
        max_depth=20,            # Deeper trees
        min_samples_split=5,     # Better generalization
        min_samples_leaf=2,      # Better generalization
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 Model Performance:")
    print(f"   MAE: €{mae:.2f}")
    print(f"   R²: {r2:.4f}")
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    importance_pairs = list(zip(available_features, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Top 10 Feature Importance:")
    for feature, importance in importance_pairs[:10]:
        print(f"   {feature:35}: {importance:.4f}")
    
    # Save model components
    model_files = {
        'model': 'improved_random_forest_model.pkl',
        'metadata': 'improved_random_forest_metadata.json'
    }
    
    # Save model
    joblib.dump(rf_model, model_files['model'])
    print(f"💾 Saved model: {model_files['model']}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'target': target,
        'features': available_features,
        'categorical_features': categorical_features,
        'encoders': {cat: le.classes_.tolist() for cat, le in encoders.items()},
        'performance': {
            'mae': float(mae),
            'r2': float(r2),
            'n_samples': len(df_clean)
        },
        'feature_importance': {feature: float(importance) for feature, importance in importance_pairs},
        'model_params': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    }
    
    with open(model_files['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {model_files['metadata']}")
    
    print(f"\n🏆 IMPROVED RANDOM FOREST READY!")
    print(f"   Performance: R² = {r2:.4f}, MAE = €{mae:.2f}")
    print(f"   Files: {list(model_files.values())}")
    
    return rf_model, metadata, available_features

if __name__ == "__main__":
    model, metadata, features = train_improved_random_forest()
    print("\n✅ Improved Random Forest training complete!") 