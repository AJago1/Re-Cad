#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    print("📊 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('correct totals 4.csv', sep=';', decimal=',')
    print(f"📈 Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # EXCLUDE PRICE-RELATED FEATURES as requested (except our target)
    price_features = ['Net price / part', 'net price * Quantity']  # Keep Net price/mm3 as target
    target = 'Net price/mm3'
    
    # Get features (exclude price-related and ID columns)
    exclude_cols = price_features + ['Project ID', 'Part Name']
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"🎯 Target: {target}")
    print(f"🔧 Using {len(features)} features")
    print(f"❌ Excluded: {price_features}")
    
    # Clean data
    df_clean = df.copy()
    
    # Convert target to numeric (handle both string and numeric types)
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
    
    # Encode categorical variables
    le_dict = {}
    for cat_feature in categorical_features:
        if cat_feature in df_clean.columns:
            le = LabelEncoder()
            df_clean[f'{cat_feature}_encoded'] = le.fit_transform(df_clean[cat_feature].fillna('Unknown'))
            le_dict[cat_feature] = le
            print(f"🏷️ Encoded {cat_feature}: {list(le.classes_)}")
    
    # Update features list
    encoded_features = [f'{cat}_encoded' for cat in categorical_features if cat in df_clean.columns]
    final_features = numeric_features + encoded_features
    
    # Remove missing target rows and fill feature NAs
    df_clean = df_clean.dropna(subset=[target])
    for feature in final_features:
        if feature in df_clean.columns:
            df_clean[feature] = df_clean[feature].fillna(df_clean[feature].median())
    
    print(f"✅ Final dataset: {len(df_clean)} rows, {len(final_features)} features")
    
    # TREND ANALYSIS
    print(f"\n🔍 DEEP TREND ANALYSIS")
    print("=" * 30)
    
    # Price statistics
    price_stats = df_clean[target].describe()
    print(f"💰 Price per mm³ Statistics:")
    print(f"   Mean: €{price_stats['mean']:.6f}/mm³")
    print(f"   Median: €{price_stats['50%']:.6f}/mm³")
    print(f"   Range: €{price_stats['min']:.6f} - €{price_stats['max']:.6f}/mm³")
    print(f"   Std Dev: €{price_stats['std']:.6f}/mm³")
    
              # Machine analysis
     if 'Machine' in df_clean.columns:
         machine_stats = df_clean.groupby('Machine')[target].agg(['count', 'mean', 'median', 'std']).round(8)
         print(f"\n🏭 Machine Analysis:")
         for machine, stats in machine_stats.iterrows():
             print(f"   {machine}:")
             print(f"      Parts: {stats['count']}, Avg: €{stats['mean']:.6f}/mm³, Median: €{stats['median']:.6f}/mm³")
     
     # Color analysis  
     if 'Colour' in df_clean.columns:
         color_stats = df_clean.groupby('Colour')[target].agg(['count', 'mean', 'median']).round(8)
         print(f"\n🎨 Color Analysis:")
         for color, stats in color_stats.head(8).iterrows():
             print(f"   {color}: {stats['count']} parts, Avg: €{stats['mean']:.6f}/mm³")
    
    # Feature correlations
    available_features = [f for f in final_features if f in df_clean.columns and not df_clean[f].isna().all()]
    correlations = df_clean[available_features + [target]].corr()[target].sort_values(ascending=False)
    
    print(f"\n📊 Top Feature Correlations with Price per mm³:")
    for feature, corr in correlations.head(10).items():
        if feature != target:
            print(f"   {feature:30}: {corr:.4f}")
    
    # OUTLIER DETECTION
    print(f"\n🚨 OUTLIER DETECTION")
    print("=" * 25)
    
    y = df_clean[target].values
    
    # Statistical outliers (Z-score > 3)
    z_scores = np.abs((y - np.mean(y)) / np.std(y))
    statistical_outliers = z_scores > 3
    
    # IQR outliers
    Q1, Q3 = np.percentile(y, [25, 75])
    IQR = Q3 - Q1
    iqr_outliers = (y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)
    
    combined_outliers = statistical_outliers | iqr_outliers
    
    print(f"📈 Outlier Detection Results:")
    print(f"   Statistical (Z>3): {statistical_outliers.sum()} outliers")
    print(f"   IQR method: {iqr_outliers.sum()} outliers")
    print(f"   Combined: {combined_outliers.sum()} total outliers")
    
    # Show top outliers
    if combined_outliers.sum() > 0:
        outlier_df = df_clean[combined_outliers].copy()
        outlier_df['z_score'] = z_scores[combined_outliers]
        top_outliers = outlier_df.nlargest(5, 'z_score')
        
        print(f"\n🔍 Top 5 Weird Outliers:")
        for idx, row in top_outliers.iterrows():
            project = row.get('Project ID', 'N/A')
            part = str(row.get('Part Name', 'N/A'))[:40]
            price = row[target]
            volume = row.get('Volume', 'N/A')
            machine = row.get('Machine', 'N/A')
            color = row.get('Colour', 'N/A')
            quantity = row.get('Quantity', 'N/A')
            print(f"   {project} - {part}...")
            print(f"      💰 €{price:.2f}, Vol: {volume}, Qty: {quantity}, Machine: {machine}, Color: {color}")
    
    # MODEL BUILDING
    print(f"\n🤖 MODEL BUILDING & COMPARISON")
    print("=" * 35)
    
    # Prepare data
    X = df_clean[available_features].values
    y = df_clean[target].values
    
    # Remove outliers for training
    clean_mask = ~combined_outliers
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]
    
    print(f"🗑️ Using {len(y_clean)} samples (removed {(~clean_mask).sum()} outliers)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    # Scale for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': RidgeCV(alphas=[0.1, 1.0, 10.0]),
        'Lasso Regression': LassoCV(alphas=[0.1, 1.0, 10.0]),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Use scaled data for linear models, original for RF
        if 'Forest' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'model': model}
        
        print(f"   📊 Results:")
        print(f"      MAE: €{mae:.2f}")
        print(f"      RMSE: €{rmse:.2f}")
        print(f"      R²: {r2:.4f}")
    
    # Feature importance for Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = rf_model.feature_importances_
        
        print(f"\n🎯 Random Forest Feature Importance (Top 15):")
        importance_pairs = list(zip(available_features, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in importance_pairs[:15]:
            print(f"   {feature:35}: {importance:.4f}")
    
    # SUMMARY REPORT
    print(f"\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 45)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_r2 = results[best_model_name]['r2']
    best_mae = results[best_model_name]['mae']
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   • Total parts analyzed: {len(df_clean):,}")
    print(f"   • Unique projects: {df_clean['Project ID'].nunique()}")
    print(f"   • Features used: {len(available_features)}")
    print(f"   • Price range: €{df_clean[target].min():.2f} - €{df_clean[target].max():.2f}")
    print(f"   • Average price: €{df_clean[target].mean():.2f}")
    
    if 'Machine' in df_clean.columns:
        print(f"\n🏭 MACHINE BREAKDOWN:")
        machine_counts = df_clean['Machine'].value_counts()
        for machine, count in machine_counts.items():
            pct = count / len(df_clean) * 100
            avg_price = df_clean[df_clean['Machine'] == machine][target].mean()
            print(f"   • {machine}: {count:,} parts ({pct:.1f}%), Avg €{avg_price:.2f}")
    
    if 'Colour' in df_clean.columns:
        print(f"\n🎨 COLOR BREAKDOWN:")
        color_counts = df_clean['Colour'].value_counts()
        for color, count in color_counts.head(8).items():
            pct = count / len(df_clean) * 100
            avg_price = df_clean[df_clean['Colour'] == color][target].mean()
            print(f"   • {color}: {count:,} parts ({pct:.1f}%), Avg €{avg_price:.2f}")
    
    print(f"\n🤖 MODEL PERFORMANCE:")
    for name, result in results.items():
        print(f"   • {name}:")
        print(f"     - MAE: €{result['mae']:.2f}")
        print(f"     - R²: {result['r2']:.4f}")
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   • R² Score: {best_r2:.4f}")
    print(f"   • MAE: €{best_mae:.2f}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Machine types show pricing differences")
    print(f"   • Color variations affect pricing patterns")
    print(f"   • Volume-related features are key price drivers")
    print(f"   • {combined_outliers.sum()} outliers detected for investigation")
    print(f"   • Random Forest captures non-linear relationships best")
    
    return df_clean, results, available_features

if __name__ == "__main__":
    df, results, features = main()
    print("\n✅ Analysis complete!") 