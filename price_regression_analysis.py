import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the data for regression analysis"""
    
    # Load the edited CSV
    df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED .csv', sep=';')
    
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns, handling European decimal format
    numeric_columns = [
        'convex_hull_volume', 'bb_volume', 'surface_area', 'volume',
        'Full_Project_Shrinkwrap_Volume_PA2200', 'Full_Project_Convex_Hull_Volume_PA2200',
        'Full_Project_BB_Volume_PA2200', 'Full_Project_Surface_Area_PA2200', 'Full_Project_Volume_PA2200',
        'Price/volume (cm3)', 'Quantity', 'x', 'y', 'z', 'Max D', 'Min D', 'D Ratio', 'Middle D',
        'waste', 'convexity_ratio', 'shrinkwrap_volume', 'shrinkwrap_ratio', 'waste_ratio', 'Production Time'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    return df

def prepare_features_and_target(df):
    """Prepare features and target variable for regression"""
    
    # Target variable
    target_col = 'Price/volume (cm3)'
    
    # Remove rows where target is missing
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"After removing missing target values: {len(df_clean)} rows")
    
    # Define feature categories
    geometric_features = [
        'convex_hull_volume', 'bb_volume', 'surface_area', 'volume',
        'x', 'y', 'z', 'Max D', 'Min D', 'D Ratio', 'Middle D',
        'waste', 'convexity_ratio', 'shrinkwrap_volume', 'shrinkwrap_ratio', 'waste_ratio'
    ]
    
    project_features = [
        'Full_Project_Shrinkwrap_Volume_PA2200', 'Full_Project_Convex_Hull_Volume_PA2200',
        'Full_Project_BB_Volume_PA2200', 'Full_Project_Surface_Area_PA2200', 'Full_Project_Volume_PA2200'
    ]
    
    production_features = ['Quantity', 'Production Time']
    
    # Combine all numeric features
    all_features = geometric_features + project_features + production_features
    
    # Keep only features that exist in the dataframe
    available_features = [col for col in all_features if col in df_clean.columns]
    
    print(f"Available features for regression: {len(available_features)}")
    
    # Create feature matrix
    X = df_clean[available_features].copy()
    y = df_clean[target_col].copy()
    
    # Handle missing values in features - fill with median or drop columns with too many NaNs
    for col in X.columns:
        nan_count = X[col].isnull().sum()
        if nan_count > 0:
            if nan_count == len(X):  # If all values are NaN, drop the column
                print(f"Dropping column {col} - all values are NaN")
                X = X.drop(columns=[col])
            elif nan_count > len(X) * 0.5:  # If more than 50% are NaN, drop the column
                print(f"Dropping column {col} - {nan_count}/{len(X)} values are NaN (>{(nan_count/len(X)*100):.1f}%)")
                X = X.drop(columns=[col])
            else:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"Filled {nan_count} missing values in {col} with median: {median_val:.2f}")
    
    print(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Target variable range: {y.min():.4f} - {y.max():.4f}")
    
    # Final check for any remaining NaN values
    total_nans = X.isnull().sum().sum()
    if total_nans > 0:
        print(f"WARNING: {total_nans} NaN values still remain - filling with 0")
        X.fillna(0, inplace=True)
    
    return X, y, list(X.columns), geometric_features, project_features, production_features

def run_regression_analysis(X, y, feature_names):
    """Run comprehensive regression analysis"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.01)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS RESULTS")
    print("="*60)
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'coefficients': model.coef_ if hasattr(model, 'coef_') else None
        }
        
        print(f"\n{name}:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  Train MAE:  {train_mae:.4f}")
        print(f"  Test MAE:   {test_mae:.4f}")
    
    return results, scaler, X_train, X_test, y_train, y_test

def analyze_feature_importance(results, feature_names, scaler):
    """Analyze and display feature importance"""
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get best model (highest test R²)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name} (Test R² = {results[best_model_name]['test_r2']:.4f})")
    
    # Feature importance for linear models
    if hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\nTop 15 Most Important Features (by absolute coefficient):")
        print("-" * 70)
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
            direction = "↑" if row['Coefficient'] > 0 else "↓"
            print(f"{i:2d}. {row['Feature']:<35} {direction} {row['Coefficient']:>10.6f}")
        
        # Separate analysis by feature type
        return feature_importance, best_model_name
    
    return None, best_model_name

def analyze_by_feature_groups(feature_importance, geometric_features, project_features, production_features):
    """Analyze importance by feature groups"""
    
    if feature_importance is None:
        return
    
    print("\n" + "="*60)
    print("FEATURE GROUP ANALYSIS")
    print("="*60)
    
    groups = {
        'Geometric Features': geometric_features,
        'Project-Level Features': project_features,
        'Production Features': production_features
    }
    
    for group_name, group_features in groups.items():
        group_data = feature_importance[feature_importance['Feature'].isin(group_features)]
        if len(group_data) > 0:
            avg_importance = group_data['Abs_Coefficient'].mean()
            max_importance = group_data['Abs_Coefficient'].max()
            top_feature = group_data.iloc[0] if len(group_data) > 0 else None
            
            print(f"\n{group_name}:")
            print(f"  Features in group: {len(group_data)}")
            print(f"  Average importance: {avg_importance:.6f}")
            print(f"  Max importance: {max_importance:.6f}")
            if top_feature is not None:
                direction = "↑" if top_feature['Coefficient'] > 0 else "↓"
                print(f"  Top feature: {top_feature['Feature']} {direction} {top_feature['Coefficient']:.6f}")

def correlation_analysis(X, y):
    """Analyze correlations with target variable"""
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS WITH PRICE/VOLUME")
    print("="*60)
    
    # Calculate correlations
    correlations = []
    for col in X.columns:
        corr, p_value = stats.pearsonr(X[col], y)
        correlations.append({
            'Feature': col,
            'Correlation': corr,
            'P_Value': p_value,
            'Abs_Correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    print(f"\nTop 15 Features by Correlation with Price/Volume:")
    print("-" * 65)
    for i, (_, row) in enumerate(corr_df.head(15).iterrows(), 1):
        significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
        direction = "+" if row['Correlation'] > 0 else "-"
        print(f"{i:2d}. {row['Feature']:<35} {direction} {abs(row['Correlation']):>6.4f} {significance}")
    
    return corr_df

def generate_insights(results, feature_importance, corr_df, best_model_name):
    """Generate business insights from the analysis"""
    
    print("\n" + "="*60)
    print("KEY BUSINESS INSIGHTS")
    print("="*60)
    
    best_r2 = results[best_model_name]['test_r2']
    
    print(f"\n1. MODEL PERFORMANCE:")
    print(f"   - Best model: {best_model_name}")
    print(f"   - Explains {best_r2:.1%} of price variation")
    if best_r2 < 0.5:
        print(f"   - ⚠️  Low explanatory power - other factors not captured may be important")
    elif best_r2 < 0.7:
        print(f"   - ✅ Moderate explanatory power - key factors identified")
    else:
        print(f"   - 🎯 High explanatory power - model captures most price drivers")
    
    if feature_importance is not None:
        print(f"\n2. TOP PRICE DRIVERS:")
        top_5 = feature_importance.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"   {i}. {row['Feature']} - {direction} price")
    
    if len(corr_df) > 0:
        print(f"\n3. STRONGEST CORRELATIONS:")
        strong_corrs = corr_df[corr_df['Abs_Correlation'] > 0.3].head(3)
        for _, row in strong_corrs.iterrows():
            direction = "positively" if row['Correlation'] > 0 else "negatively"
            print(f"   - {row['Feature']} is {direction} correlated (r={row['Correlation']:.3f})")

def main():
    """Main analysis function"""
    
    print("🔍 STARTING PRICE REGRESSION ANALYSIS FOR PA2200 PARTS")
    print("="*80)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        X, y, feature_names, geometric_features, project_features, production_features = prepare_features_and_target(df)
        
        # Run regression analysis
        results, scaler, X_train, X_test, y_train, y_test = run_regression_analysis(X, y, feature_names)
        
        # Analyze feature importance
        feature_importance, best_model_name = analyze_feature_importance(results, feature_names, scaler)
        
        # Analyze by feature groups
        analyze_by_feature_groups(feature_importance, geometric_features, project_features, production_features)
        
        # Correlation analysis
        corr_df = correlation_analysis(X, y)
        
        # Generate insights
        generate_insights(results, feature_importance, corr_df, best_model_name)
        
        # Save detailed results
        if feature_importance is not None:
            feature_importance.to_csv('feature_importance_analysis.csv', index=False)
            print(f"\n📊 Detailed feature importance saved to: feature_importance_analysis.csv")
        
        corr_df.to_csv('correlation_analysis.csv', index=False)
        print(f"📊 Correlation analysis saved to: correlation_analysis.csv")
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 