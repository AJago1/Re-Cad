import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

def investigate_mape():
    # Load data
    df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 2.0.csv', sep=';')
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
    available_features = [col for col in all_features if col in df_clean.columns]
    
    X = df_clean[available_features].copy()
    y = df_clean[target_col].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    
    print("DATA INVESTIGATION")
    print("="*50)
    
    print(f"Target variable (Net price / part) statistics:")
    print(f"  Count: {len(y)}")
    print(f"  Min: {y.min():.4f}")
    print(f"  Max: {y.max():.4f}")
    print(f"  Mean: {y.mean():.4f}")
    print(f"  Median: {y.median():.4f}")
    print(f"  Std: {y.std():.4f}")
    
    # Check for zero values
    zero_count = (y == 0).sum()
    print(f"  Zero values: {zero_count} ({zero_count/len(y)*100:.1f}%)")
    
    # Check for very small values
    small_values = (y > 0) & (y < 1)
    small_count = small_values.sum()
    print(f"  Values between 0-1: {small_count} ({small_count/len(y)*100:.1f}%)")
    
    # Value ranges
    print(f"\nValue distribution:")
    print(f"  0: {(y == 0).sum()}")
    print(f"  0-1: {((y > 0) & (y <= 1)).sum()}")
    print(f"  1-10: {((y > 1) & (y <= 10)).sum()}")
    print(f"  10-100: {((y > 10) & (y <= 100)).sum()}")
    print(f"  100-1000: {((y > 100) & (y <= 1000)).sum()}")
    print(f"  >1000: {(y > 1000).sum()}")
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate various error metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Calculate MAPE excluding zero values
    mask_non_zero = y_test != 0
    if mask_non_zero.sum() > 0:
        mape_non_zero = np.mean(np.abs((y_test[mask_non_zero] - y_pred_test[mask_non_zero]) / y_test[mask_non_zero])) * 100
        print(f"  MAPE (excluding zeros): {mape_non_zero:.2f}%")
        print(f"  Non-zero test samples: {mask_non_zero.sum()}/{len(y_test)}")
    
    # Calculate MAPE excluding very small values (< 1)
    mask_reasonable = y_test >= 1
    if mask_reasonable.sum() > 0:
        mape_reasonable = np.mean(np.abs((y_test[mask_reasonable] - y_pred_test[mask_reasonable]) / y_test[mask_reasonable])) * 100
        print(f"  MAPE (excluding < 1): {mape_reasonable:.2f}%")
        print(f"  Reasonable test samples: {mask_reasonable.sum()}/{len(y_test)}")
    
    # Calculate MAPE excluding very small values (< 10)
    mask_larger = y_test >= 10
    if mask_larger.sum() > 0:
        mape_larger = np.mean(np.abs((y_test[mask_larger] - y_pred_test[mask_larger]) / y_test[mask_larger])) * 100
        print(f"  MAPE (excluding < 10): {mape_larger:.2f}%")
        print(f"  Larger test samples: {mask_larger.sum()}/{len(y_test)}")
    
    # Show some actual vs predicted examples
    print(f"\nACTUAL vs PREDICTED examples (first 10 test samples):")
    print("Actual    Predicted  Error%")
    print("-"*30)
    for i in range(min(10, len(y_test))):
        actual = y_test.iloc[i]
        pred = y_pred_test[i]
        if actual != 0:
            error_pct = abs(actual - pred) / actual * 100
        else:
            error_pct = float('inf')
        print(f"{actual:8.2f}  {pred:8.2f}   {error_pct:6.1f}%")

if __name__ == "__main__":
    investigate_mape() 