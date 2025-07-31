"""
Random Forest Pricing Model for STL Parts
Enhanced with overfitting prevention and proper validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the training data"""
    print("📊 Loading training data...")
    
    # Load CSV data
    csv_path = "correct totals 3.csv"
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    print(f"✅ Loaded {len(df)} rows of training data")
    
    # Define base features (using actual column names from CSV)
    base_features = [
        'shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'surface_area',
        'bb_volume', 'Max D', 'Min D', 'D Ratio', 'Quantity', 'waste'
    ]
    
    target_column = 'Net price / part'
    
    # Check available features
    available_features = [f for f in base_features if f in df.columns]
    missing_features = [f for f in base_features if f not in df.columns]
    
    print(f"✅ Available features: {len(available_features)}/{len(base_features)}")
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    
    # Convert to numeric and clean data
    for col in available_features + [target_column]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values and extreme outliers
    df_clean = df[available_features + [target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Remove price outliers (beyond 3 standard deviations)
    price_mean = df_clean[target_column].mean()
    price_std = df_clean[target_column].std()
    price_threshold = 3 * price_std
    
    before_outlier_removal = len(df_clean)
    df_clean = df_clean[
        (df_clean[target_column] >= price_mean - price_threshold) & 
        (df_clean[target_column] <= price_mean + price_threshold)
    ]
    after_outlier_removal = len(df_clean)
    
    print(f"🧹 Removed {before_outlier_removal - after_outlier_removal} price outliers")
    print(f"✅ Clean dataset: {len(df_clean)} samples")
    
    # Prepare features and target
    X = df_clean[available_features]
    y = df_clean[target_column]
    
    print(f"📈 Price range: €{y.min():.2f} - €{y.max():.2f} (mean: €{y.mean():.2f})")
    
    return X, y, available_features

def create_enhanced_features(X):
    """Create additional engineered features for Random Forest"""
    print("🔧 Creating enhanced features...")
    
    X_enhanced = X.copy()
    
    # Geometric ratios and relationships
    X_enhanced['volume_to_surface_ratio'] = X['Volume'] / np.maximum(X['surface_area'], 1)
    X_enhanced['convex_hull_efficiency'] = X['Volume'] / np.maximum(X['convex_hull_volume'], 1)
    X_enhanced['shrinkwrap_efficiency'] = X['Volume'] / np.maximum(X['shrinkwrap_volume'], 1)
    X_enhanced['waste_ratio'] = X['waste'] / np.maximum(X['bb_volume'], 1)
    
    # Dimensional analysis
    X_enhanced['max_min_ratio_enhanced'] = X['Max D'] / np.maximum(X['Min D'], 0.1)
    X_enhanced['dimension_spread'] = X['Max D'] - X['Min D']
    X_enhanced['dimension_product'] = X['Max D'] * X['Min D']
    X_enhanced['d_ratio_squared'] = X['D Ratio'] ** 2
    
    # Volume-based features
    X_enhanced['volume_per_unit'] = X['Volume'] / np.maximum(X['Quantity'], 1)
    X_enhanced['surface_per_unit'] = X['surface_area'] / np.maximum(X['Quantity'], 1)
    
    # Complexity indicators
    X_enhanced['geometric_complexity'] = (X['surface_area'] ** 2) / np.maximum(X['Volume'], 1)
    X_enhanced['dimensional_complexity'] = (X['Max D'] * X['Min D']) / np.maximum(X['Volume'], 1)
    X_enhanced['shape_complexity'] = X['D Ratio'] * X_enhanced['waste_ratio']
    
    # Logarithmic features for scale effects
    for col in ['Volume', 'surface_area', 'convex_hull_volume', 'shrinkwrap_volume']:
        if col in X.columns:
            X_enhanced[f'log_{col}'] = np.log1p(X[col])
    
    # Quantity-dependent features
    X_enhanced['total_volume'] = X['Volume'] * X['Quantity']
    X_enhanced['total_surface'] = X['surface_area'] * X['Quantity']
    
    print(f"✅ Enhanced features: {len(X_enhanced.columns)} total features")
    
    return X_enhanced

def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest hyperparameters with cross-validation"""
    print("🎯 Optimizing Random Forest hyperparameters...")
    
    # Define parameter grid - conservative to prevent overfitting
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True],
    }
    
    # Base Random Forest
    rf_base = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )
    
    # Grid search with cross-validation
    print("🔍 Performing grid search with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"✅ Best cross-validation R² score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Comprehensive model evaluation"""
    print("\n📊 Model Evaluation:")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Test metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Out-of-bag score
    oob_score = getattr(model, 'oob_score_', None)
    
    print(f"📈 Training Performance:")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   MAE: €{train_mae:.2f}")
    print(f"   RMSE: €{train_rmse:.2f}")
    
    print(f"📉 Test Performance:")
    print(f"   R² Score: {test_r2:.4f}")
    print(f"   MAE: €{test_mae:.2f}")
    print(f"   RMSE: €{test_rmse:.2f}")
    
    if oob_score:
        print(f"🎯 Out-of-Bag Score: {oob_score:.4f}")
    
    # Check for overfitting
    r2_diff = train_r2 - test_r2
    mae_diff = test_mae - train_mae
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   R² difference (train - test): {r2_diff:.4f}")
    print(f"   MAE difference (test - train): €{mae_diff:.2f}")
    
    if r2_diff > 0.1:
        print("   ⚠️ Potential overfitting detected (R² gap > 0.1)")
    elif r2_diff > 0.05:
        print("   ⚡ Mild overfitting (R² gap > 0.05)")
    else:
        print("   ✅ Good generalization (R² gap < 0.05)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔥 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'oob_score': oob_score,
        'feature_importance': feature_importance,
        'overfitting_risk': 'High' if r2_diff > 0.1 else 'Medium' if r2_diff > 0.05 else 'Low'
    }

def save_model_and_metadata(model, scaler, feature_names, metrics, model_params):
    """Save the trained model and metadata"""
    print("\n💾 Saving model and metadata...")
    
    # Save model
    with open('random_forest_pricing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('random_forest_feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'creation_date': datetime.now().isoformat(),
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_parameters': model_params,
        'performance_metrics': {
            'train_r2_score': float(metrics['train_r2']),
            'test_r2_score': float(metrics['test_r2']),
            'train_mae': float(metrics['train_mae']),
            'test_mae': float(metrics['test_mae']),
            'train_rmse': float(metrics['train_rmse']),
            'test_rmse': float(metrics['test_rmse']),
            'oob_score': float(metrics['oob_score']) if metrics['oob_score'] else None,
            'overfitting_risk': metrics['overfitting_risk']
        },
        'feature_importance': {
            row['feature']: float(row['importance']) 
            for _, row in metrics['feature_importance'].iterrows()
        }
    }
    
    with open('random_forest_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model saved successfully!")
    print("   📁 random_forest_pricing_model.pkl")
    print("   📁 random_forest_feature_scaler.pkl") 
    print("   📁 random_forest_model_metadata.json")

def main():
    """Main training pipeline"""
    print("🌲 Random Forest Pricing Model Training")
    print("=" * 50)
    
    # Load and prepare data
    X, y, base_features = load_and_prepare_data()
    
    # Create enhanced features
    X_enhanced = create_enhanced_features(X)
    
    # Split data (70/30 as requested)
    print(f"\n📊 Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, 
        test_size=0.3, 
        random_state=42
    )
    
    print(f"✅ Training set: {len(X_train)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    
    # Scale features
    print(f"\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_enhanced.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_enhanced.columns, index=X_test.index)
    
    # Optimize Random Forest
    best_rf = optimize_random_forest(X_train_scaled, y_train)
    
    # Evaluate model
    metrics = evaluate_model(
        best_rf, X_train_scaled, X_test_scaled, 
        y_train, y_test, X_enhanced.columns.tolist()
    )
    
    # Save model and metadata
    model_params = best_rf.get_params()
    save_model_and_metadata(
        best_rf, scaler, X_enhanced.columns.tolist(), 
        metrics, model_params
    )
    
    print(f"\n🎯 Final Model Summary:")
    print(f"   Model Type: Random Forest Regressor")
    print(f"   Test R² Score: {metrics['test_r2']:.4f}")
    print(f"   Test MAE: €{metrics['test_mae']:.2f}")
    print(f"   Overfitting Risk: {metrics['overfitting_risk']}")
    print(f"   Number of Trees: {model_params['n_estimators']}")
    print(f"   Max Depth: {model_params['max_depth']}")
    
    return best_rf, scaler, metrics

if __name__ == "__main__":
    model, scaler, metrics = main() 