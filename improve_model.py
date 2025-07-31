import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def improve_model():
    # Load data
    df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (removed false prices).csv', sep=';')
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns
    numeric_columns = [
        'convex_hull_volume', 'bb_volume', 'surface_area', 'Volume',
        'Full_Project_Shrinkwrap_Volume_PA2200', 'Full_Project_Convex_Hull_Volume_PA2200',
        'Full_Project_BB_Volume_PA2200', 'Full_Project_Surface_Area_PA2200', 'Full_Project_Volume_PA2200',
        'Net price / part', 'Quantity', 'Max D', 'Min D', 'D Ratio',
        'waste', 'convexity_ratio', 'shrinkwrap_volume', 'shrinkwrap_ratio', 'waste_ratio'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Prepare features and target
    target_col = 'Net price / part'
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # STRATEGY 1: Remove outliers (focus on main business range)
    print("STRATEGY 1: DATA ANALYSIS")
    print("="*50)
    
    # Since false prices are already removed, let's be less aggressive with outlier removal
    # Keep broader range since data is already cleaned
    mask_business = (df_clean[target_col] >= 2) & (df_clean[target_col] <= 5000)  # Broader range
    df_business = df_clean[mask_business].copy()
    print(f"Original cleaned data: {len(df_clean)} samples")
    print(f"Final business range (€2-5000): {len(df_business)} samples")
    print(f"Removed extreme outliers: {len(df_clean) - len(df_business)} samples")
    
    # Features
    geometric_features = [
        'convex_hull_volume', 'bb_volume', 'surface_area', 'Volume',
        'Max D', 'Min D', 'D Ratio',
        'waste', 'convexity_ratio', 'shrinkwrap_volume', 'shrinkwrap_ratio', 'waste_ratio'
    ]
    
    project_features = [
        'Full_Project_Shrinkwrap_Volume_PA2200', 'Full_Project_Convex_Hull_Volume_PA2200',
        'Full_Project_BB_Volume_PA2200', 'Full_Project_Surface_Area_PA2200', 'Full_Project_Volume_PA2200'
    ]
    
    production_features = ['Quantity']
    all_features = geometric_features + project_features + production_features
    available_features = [col for col in all_features if col in df_business.columns]
    
    X = df_business[available_features].copy()
    y = df_business[target_col].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    
    # STRATEGY 2: Feature Engineering
    print(f"\nSTRATEGY 2: FEATURE ENGINEERING")
    print("="*50)
    
    # Add derived features
    X['volume_per_surface'] = X['Volume'] / (X['surface_area'] + 0.001)  # Avoid division by zero
    X['efficiency_ratio'] = X['Volume'] / (X['shrinkwrap_volume'] + 0.001)
    X['size_ratio'] = X['Max D'] / (X['Min D'] + 0.001)
    X['project_share'] = X['Volume'] / (X['Full_Project_Volume_PA2200'] + 0.001)
    X['density_metric'] = X['Volume'] / (X['bb_volume'] + 0.001)
    
    print(f"Added 5 engineered features")
    print(f"Total features: {len(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # STRATEGY 3: Try Multiple Models
    print(f"\nSTRATEGY 3: MODEL COMPARISON")
    print("="*50)
    
    models = {
        'Ridge Regression': (Ridge(alpha=1.0), True),  # Needs scaling
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42), False),  # No scaling needed
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=42), False)  # No scaling needed
    }
    
    results = {}
    
    for name, (model, needs_scaling) in models.items():
        if needs_scaling:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Fit model
        model.fit(X_train_processed, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_processed)
        y_pred_test = model.predict(X_test_processed)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        # MAPE for business range (all values >= 1)
        mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mae': mae,
            'mape': mape,
            'predictions': y_pred_test
        }
        
        print(f"\n{name}:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  MAPE: {mape:.1f}%")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['mape'])
    best_result = results[best_model_name]
    
    print(f"\n🎯 BEST MODEL: {best_model_name}")
    print(f"   MAPE: {best_result['mape']:.1f}%")
    print(f"   R²: {best_result['test_r2']:.4f}")
    
    # STRATEGY 4: Feature Importance Analysis
    print(f"\nSTRATEGY 4: FEATURE IMPORTANCE (Best Model)")
    print("="*50)
    
    if hasattr(best_result['model'], 'feature_importances_'):
        # Tree-based model
        importances = best_result['model'].feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:<30} {row['Importance']:.4f}")
    
    # STRATEGY 5: Error Analysis
    print(f"\nSTRATEGY 5: ERROR ANALYSIS")
    print("="*50)
    
    y_pred_best = best_result['predictions']
    
    # Error by price range
    price_ranges = [
        (1, 10, "€1-10"),
        (10, 50, "€10-50"), 
        (50, 100, "€50-100"),
        (100, 500, "€100-500"),
        (500, 1000, "€500-1000")
    ]
    
    print("MAPE by Price Range:")
    for min_price, max_price, label in price_ranges:
        mask = (y_test >= min_price) & (y_test < max_price)
        if mask.sum() > 5:  # At least 5 samples
            range_mape = np.mean(np.abs((y_test[mask] - y_pred_best[mask]) / y_test[mask])) * 100
            print(f"  {label}: {range_mape:.1f}% (n={mask.sum()})")
    
    # Show best predictions
    print(f"\nBEST PREDICTIONS (lowest % error):")
    errors = np.abs((y_test - y_pred_best) / y_test) * 100
    best_indices = np.argsort(errors)[:10]
    
    print("Actual    Predicted  Error%")
    print("-"*30)
    for i in best_indices:
        actual = y_test.iloc[i]
        pred = y_pred_best[i]
        error_pct = errors.iloc[i]
        print(f"{actual:8.2f}  {pred:8.2f}   {error_pct:6.1f}%")
    
    return best_result

if __name__ == "__main__":
    improve_model() 