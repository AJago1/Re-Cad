#!/usr/bin/env python3
"""
Advanced Feature Selection for STL Pricing Model
=================================================

This script implements multiple feature selection techniques to find the optimal
combination of geometric features for pricing prediction, excluding any pricing information.

Techniques used:
1. Lasso Regression (L1 regularization) - automatic feature selection
2. Ridge Regression (L2 regularization) - feature shrinkage
3. Elastic Net - combination of L1 and L2
4. Recursive Feature Elimination (RFE)
5. Mutual Information feature selection
6. Variance Threshold filtering
7. Correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.feature_selection import RFE, mutual_info_regression, VarianceThreshold, SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("🎯 ADVANCED FEATURE SELECTION FOR STL PRICING MODEL")
print("=" * 60)

# Load and prepare data
print("📁 Loading data...")
df = pd.read_csv('correct totals 3.csv', sep=';', decimal=',')
print(f"Dataset shape: {df.shape}")

# Define all available numerical features (excluding pricing information)
all_features = [
    # Basic geometric features
    'convex_hull_volume', 'bb_volume', 'surface_area', 'Volume', 
    'Max D', 'Min D', 'D Ratio', 'waste', 'shrinkwrap_volume', 'Quantity',
    
    # Derived quantity-based features
    'Volume * Quantity', 'shrinkwrap_volume*Quantity', 'Surface area*Quantity',
    'BB_Volume*Quantity', 'Convex_Hull_Volume*Quantity',
    
    # Project-level features
    'Total_Parts_in_Project', 'Project_Total_convex_hull_volume',
    'Project_Total_bb_volume', 'Project_Total_shrinkwrap_volume',
    'Project_Total_Volume', 'Project_Total_surface_area'
]

target = 'Net price / part'

print(f"\n🔍 Available features: {len(all_features)}")
for i, feature in enumerate(all_features, 1):
    print(f"  {i:2d}. {feature}")

# Clean and prepare data
print(f"\n🧹 Cleaning data...")
# Convert all features to numeric
for col in all_features + [target]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"⚠️  Missing feature: {col}")

# Remove missing features
available_features = [f for f in all_features if f in df.columns]
print(f"✅ Available features after cleaning: {len(available_features)}")

# Remove rows with missing values
df_clean = df[available_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
print(f"📊 Clean dataset shape: {df_clean.shape}")

X = df_clean[available_features]
y = df_clean[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🔄 Train set: {X_train.shape}, Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n🎯 TARGET ANALYSIS:")
print(f"Price range: €{y.min():.2f} - €{y.max():.2f}")
print(f"Price mean: €{y.mean():.2f} ± €{y.std():.2f}")

# =============================================================================
# 1. BASELINE LINEAR REGRESSION
# =============================================================================
print(f"\n" + "="*60)
print("1️⃣  BASELINE LINEAR REGRESSION")
print("="*60)

lr_baseline = LinearRegression()
lr_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = lr_baseline.predict(X_test_scaled)

baseline_r2 = r2_score(y_test, y_pred_baseline)
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

print(f"📊 Baseline Performance:")
print(f"   R² Score: {baseline_r2:.4f}")
print(f"   MAE: €{baseline_mae:.2f}")
print(f"   RMSE: €{baseline_rmse:.2f}")

# Feature importance from coefficients
feature_importance_baseline = pd.DataFrame({
    'feature': available_features,
    'coefficient': lr_baseline.coef_,
    'abs_coefficient': np.abs(lr_baseline.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n🎯 Top 10 Features (Baseline):")
for i, (_, row) in enumerate(feature_importance_baseline.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<35} | Coef: {row['coefficient']:8.4f}")

# =============================================================================
# 2. LASSO REGRESSION (L1 REGULARIZATION)
# =============================================================================
print(f"\n" + "="*60)
print("2️⃣  LASSO REGRESSION (L1 REGULARIZATION)")
print("="*60)

# Grid search for optimal alpha
lasso_alphas = np.logspace(-4, 2, 50)
lasso_grid = GridSearchCV(
    Lasso(random_state=42, max_iter=2000),
    {'alpha': lasso_alphas},
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("🔍 Finding optimal Lasso alpha...")
lasso_grid.fit(X_train_scaled, y_train)
best_lasso = lasso_grid.best_estimator_

y_pred_lasso = best_lasso.predict(X_test_scaled)
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print(f"📊 Lasso Performance:")
print(f"   Best Alpha: {lasso_grid.best_params_['alpha']:.6f}")
print(f"   R² Score: {lasso_r2:.4f}")
print(f"   MAE: €{lasso_mae:.2f}")
print(f"   RMSE: €{lasso_rmse:.2f}")

# Selected features (non-zero coefficients)
lasso_features = pd.DataFrame({
    'feature': available_features,
    'coefficient': best_lasso.coef_
})
lasso_selected = lasso_features[lasso_features['coefficient'] != 0].copy()
lasso_selected['abs_coefficient'] = np.abs(lasso_selected['coefficient'])
lasso_selected = lasso_selected.sort_values('abs_coefficient', ascending=False)

print(f"\n🎯 Lasso Selected Features ({len(lasso_selected)}/{len(available_features)}):")
for i, (_, row) in enumerate(lasso_selected.iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<35} | Coef: {row['coefficient']:8.4f}")

# =============================================================================
# 3. RIDGE REGRESSION (L2 REGULARIZATION)
# =============================================================================
print(f"\n" + "="*60)
print("3️⃣  RIDGE REGRESSION (L2 REGULARIZATION)")
print("="*60)

# Grid search for optimal alpha
ridge_alphas = np.logspace(-4, 4, 50)
ridge_grid = GridSearchCV(
    Ridge(random_state=42),
    {'alpha': ridge_alphas},
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("🔍 Finding optimal Ridge alpha...")
ridge_grid.fit(X_train_scaled, y_train)
best_ridge = ridge_grid.best_estimator_

y_pred_ridge = best_ridge.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"📊 Ridge Performance:")
print(f"   Best Alpha: {ridge_grid.best_params_['alpha']:.6f}")
print(f"   R² Score: {ridge_r2:.4f}")
print(f"   MAE: €{ridge_mae:.2f}")
print(f"   RMSE: €{ridge_rmse:.2f}")

# Feature importance from Ridge
ridge_importance = pd.DataFrame({
    'feature': available_features,
    'coefficient': best_ridge.coef_,
    'abs_coefficient': np.abs(best_ridge.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n🎯 Top 10 Features (Ridge):")
for i, (_, row) in enumerate(ridge_importance.head(10).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<35} | Coef: {row['coefficient']:8.4f}")

# =============================================================================
# 4. ELASTIC NET (L1 + L2 REGULARIZATION)
# =============================================================================
print(f"\n" + "="*60)
print("4️⃣  ELASTIC NET (L1 + L2 REGULARIZATION)")
print("="*60)

# Grid search for optimal parameters
elastic_params = {
    'alpha': np.logspace(-4, 2, 20),
    'l1_ratio': np.linspace(0.1, 0.9, 9)
}

elastic_grid = GridSearchCV(
    ElasticNet(random_state=42, max_iter=2000),
    elastic_params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("🔍 Finding optimal Elastic Net parameters...")
elastic_grid.fit(X_train_scaled, y_train)
best_elastic = elastic_grid.best_estimator_

y_pred_elastic = best_elastic.predict(X_test_scaled)
elastic_r2 = r2_score(y_test, y_pred_elastic)
elastic_mae = mean_absolute_error(y_test, y_pred_elastic)
elastic_rmse = np.sqrt(mean_squared_error(y_test, y_pred_elastic))

print(f"📊 Elastic Net Performance:")
print(f"   Best Alpha: {elastic_grid.best_params_['alpha']:.6f}")
print(f"   Best L1 Ratio: {elastic_grid.best_params_['l1_ratio']:.3f}")
print(f"   R² Score: {elastic_r2:.4f}")
print(f"   MAE: €{elastic_mae:.2f}")
print(f"   RMSE: €{elastic_rmse:.2f}")

# Selected features
elastic_features = pd.DataFrame({
    'feature': available_features,
    'coefficient': best_elastic.coef_
})
elastic_selected = elastic_features[elastic_features['coefficient'] != 0].copy()
elastic_selected['abs_coefficient'] = np.abs(elastic_selected['coefficient'])
elastic_selected = elastic_selected.sort_values('abs_coefficient', ascending=False)

print(f"\n🎯 Elastic Net Selected Features ({len(elastic_selected)}/{len(available_features)}):")
for i, (_, row) in enumerate(elastic_selected.iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<35} | Coef: {row['coefficient']:8.4f}")

# =============================================================================
# 5. RECURSIVE FEATURE ELIMINATION (RFE)
# =============================================================================
print(f"\n" + "="*60)
print("5️⃣  RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*60)

# Test different numbers of features
n_features_to_test = [5, 10, 15, 20]
rfe_results = {}

for n_features in n_features_to_test:
    if n_features <= len(available_features):
        print(f"🔍 Testing RFE with {n_features} features...")
        
        rfe = RFE(LinearRegression(), n_features_to_select=n_features)
        rfe.fit(X_train_scaled, y_train)
        
        X_train_rfe = rfe.transform(X_train_scaled)
        X_test_rfe = rfe.transform(X_test_scaled)
        
        lr_rfe = LinearRegression()
        lr_rfe.fit(X_train_rfe, y_train)
        y_pred_rfe = lr_rfe.predict(X_test_rfe)
        
        rfe_r2 = r2_score(y_test, y_pred_rfe)
        rfe_mae = mean_absolute_error(y_test, y_pred_rfe)
        
        selected_features = [available_features[i] for i in range(len(available_features)) if rfe.support_[i]]
        
        rfe_results[n_features] = {
            'r2': rfe_r2,
            'mae': rfe_mae,
            'features': selected_features,
            'model': rfe
        }
        
        print(f"   R² Score: {rfe_r2:.4f} | MAE: €{rfe_mae:.2f}")

# Find best RFE configuration
best_rfe_n = max(rfe_results.keys(), key=lambda k: rfe_results[k]['r2'])
best_rfe_result = rfe_results[best_rfe_n]

print(f"\n🎯 Best RFE Configuration ({best_rfe_n} features):")
print(f"   R² Score: {best_rfe_result['r2']:.4f}")
print(f"   MAE: €{best_rfe_result['mae']:.2f}")
print(f"   Selected Features:")
for i, feature in enumerate(best_rfe_result['features'], 1):
    print(f"     {i:2d}. {feature}")

# =============================================================================
# 6. MUTUAL INFORMATION FEATURE SELECTION
# =============================================================================
print(f"\n" + "="*60)
print("6️⃣  MUTUAL INFORMATION FEATURE SELECTION")
print("="*60)

print("🔍 Calculating mutual information scores...")
mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)

mi_features = pd.DataFrame({
    'feature': available_features,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\n🎯 Top 15 Features by Mutual Information:")
for i, (_, row) in enumerate(mi_features.head(15).iterrows(), 1):
    print(f"  {i:2d}. {row['feature']:<35} | MI Score: {row['mi_score']:.4f}")

# Test different numbers of top MI features
mi_results = {}
for n_features in [5, 10, 15]:
    if n_features <= len(available_features):
        top_mi_features = mi_features.head(n_features)['feature'].tolist()
        
        X_train_mi = X_train[top_mi_features]
        X_test_mi = X_test[top_mi_features]
        
        # Scale the selected features
        scaler_mi = StandardScaler()
        X_train_mi_scaled = scaler_mi.fit_transform(X_train_mi)
        X_test_mi_scaled = scaler_mi.transform(X_test_mi)
        
        lr_mi = LinearRegression()
        lr_mi.fit(X_train_mi_scaled, y_train)
        y_pred_mi = lr_mi.predict(X_test_mi_scaled)
        
        mi_r2 = r2_score(y_test, y_pred_mi)
        mi_mae = mean_absolute_error(y_test, y_pred_mi)
        
        mi_results[n_features] = {
            'r2': mi_r2,
            'mae': mi_mae,
            'features': top_mi_features
        }
        
        print(f"🔍 Top {n_features} MI features - R²: {mi_r2:.4f}, MAE: €{mi_mae:.2f}")

# =============================================================================
# 7. COMPREHENSIVE COMPARISON
# =============================================================================
print(f"\n" + "="*60)
print("7️⃣  COMPREHENSIVE COMPARISON")
print("="*60)

results_summary = {
    'Baseline Linear': {'r2': baseline_r2, 'mae': baseline_mae, 'rmse': baseline_rmse, 'n_features': len(available_features)},
    'Lasso': {'r2': lasso_r2, 'mae': lasso_mae, 'rmse': lasso_rmse, 'n_features': len(lasso_selected)},
    'Ridge': {'r2': ridge_r2, 'mae': ridge_mae, 'rmse': ridge_rmse, 'n_features': len(available_features)},
    'Elastic Net': {'r2': elastic_r2, 'mae': elastic_mae, 'rmse': elastic_rmse, 'n_features': len(elastic_selected)},
    f'RFE ({best_rfe_n})': {'r2': best_rfe_result['r2'], 'mae': best_rfe_result['mae'], 'rmse': 0, 'n_features': best_rfe_n}
}

# Add MI results
for n_features, result in mi_results.items():
    results_summary[f'MI Top {n_features}'] = {
        'r2': result['r2'], 
        'mae': result['mae'], 
        'rmse': 0, 
        'n_features': n_features
    }

print(f"📊 PERFORMANCE COMPARISON:")
print(f"{'Method':<20} {'R² Score':<10} {'MAE (€)':<10} {'Features':<10} {'Efficiency':<12}")
print("-" * 70)

for method, metrics in results_summary.items():
    efficiency = metrics['r2'] / metrics['n_features'] if metrics['n_features'] > 0 else 0
    print(f"{method:<20} {metrics['r2']:<10.4f} {metrics['mae']:<10.2f} {metrics['n_features']:<10} {efficiency:<12.6f}")

# Find best method
best_method = max(results_summary.keys(), key=lambda k: results_summary[k]['r2'])
print(f"\n🏆 BEST METHOD: {best_method}")
print(f"   R² Score: {results_summary[best_method]['r2']:.4f}")
print(f"   MAE: €{results_summary[best_method]['mae']:.2f}")
print(f"   Features: {results_summary[best_method]['n_features']}")

# =============================================================================
# 8. FINAL RECOMMENDATIONS
# =============================================================================
print(f"\n" + "="*60)
print("8️⃣  FINAL RECOMMENDATIONS")
print("="*60)

print(f"🎯 OPTIMAL FEATURE SET RECOMMENDATIONS:")
print(f"\n1️⃣  LASSO SELECTED FEATURES ({len(lasso_selected)} features):")
print(f"   Best for: Automatic feature selection with sparsity")
print(f"   Performance: R² = {lasso_r2:.4f}, MAE = €{lasso_mae:.2f}")
print(f"   Features:")
for i, (_, row) in enumerate(lasso_selected.head(10).iterrows(), 1):
    print(f"     {i:2d}. {row['feature']}")

print(f"\n2️⃣  TOP MUTUAL INFORMATION FEATURES:")
best_mi_n = max(mi_results.keys(), key=lambda k: mi_results[k]['r2'])
best_mi_result = mi_results[best_mi_n]
print(f"   Best for: Non-linear feature relationships")
print(f"   Performance: R² = {best_mi_result['r2']:.4f}, MAE = €{best_mi_result['mae']:.2f}")
print(f"   Top {best_mi_n} features:")
for i, feature in enumerate(best_mi_result['features'], 1):
    print(f"     {i:2d}. {feature}")

print(f"\n3️⃣  RFE SELECTED FEATURES:")
print(f"   Best for: Systematic feature elimination")
print(f"   Performance: R² = {best_rfe_result['r2']:.4f}, MAE = €{best_rfe_result['mae']:.2f}")
print(f"   Features:")
for i, feature in enumerate(best_rfe_result['features'], 1):
    print(f"     {i:2d}. {feature}")

# Create final optimal feature set (intersection of top methods)
lasso_features_set = set(lasso_selected['feature'].tolist())
mi_features_set = set(best_mi_result['features'])
rfe_features_set = set(best_rfe_result['features'])

# Features that appear in multiple methods
intersection_2 = (lasso_features_set & mi_features_set) | (lasso_features_set & rfe_features_set) | (mi_features_set & rfe_features_set)
intersection_3 = lasso_features_set & mi_features_set & rfe_features_set

print(f"\n🎯 CONSENSUS FEATURES:")
print(f"Features selected by 2+ methods ({len(intersection_2)}):")
for i, feature in enumerate(sorted(intersection_2), 1):
    print(f"  {i:2d}. {feature}")

if intersection_3:
    print(f"\nFeatures selected by ALL 3 methods ({len(intersection_3)}):")
    for i, feature in enumerate(sorted(intersection_3), 1):
        print(f"  {i:2d}. {feature}")

# Save results
print(f"\n💾 SAVING RESULTS...")

# Save feature rankings
feature_rankings = pd.DataFrame({
    'feature': available_features,
    'lasso_coef': [lasso_features.set_index('feature').loc[f, 'coefficient'] if f in lasso_features['feature'].values else 0 for f in available_features],
    'ridge_coef': ridge_importance.set_index('feature')['coefficient'],
    'elastic_coef': [elastic_features.set_index('feature').loc[f, 'coefficient'] if f in elastic_features['feature'].values else 0 for f in available_features],
    'mi_score': mi_features.set_index('feature')['mi_score'],
    'in_lasso': [f in lasso_features_set for f in available_features],
    'in_rfe': [f in rfe_features_set for f in available_features],
    'in_top_mi': [f in mi_features_set for f in available_features]
})

feature_rankings.to_csv('feature_selection_results.csv', index=False)
print(f"✅ Feature rankings saved to 'feature_selection_results.csv'")

# Save model comparison
comparison_df = pd.DataFrame(results_summary).T
comparison_df.to_csv('model_comparison_results.csv')
print(f"✅ Model comparison saved to 'model_comparison_results.csv'")

print(f"\n🎉 FEATURE SELECTION ANALYSIS COMPLETE!")
print(f"📈 Best performing method: {best_method}")
print(f"🎯 Recommended features for GUI implementation: {len(lasso_selected)} Lasso-selected features")

# Save final recommendations
print(f"\n💾 SAVING FINAL RECOMMENDATIONS...")

# Save optimal feature set
optimal_features = sorted(intersection_2)
print(f"🎯 OPTIMAL FEATURE SET FOR GUI:")
for i, feature in enumerate(optimal_features, 1):
    print(f"  {i:2d}. {feature}")

# Save results as JSON
results = {
    'optimal_features': optimal_features,
    'lasso_coefficients': dict(zip(lasso_selected['feature'], lasso_selected['coefficient'])),
    'model_performance': {'r2': lasso_r2, 'mae': lasso_mae},
    'scaler_params': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()},
    'feature_names': available_features
}

import json
with open('optimal_pricing_model.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to 'optimal_pricing_model.json'")
print(f"🎉 FEATURE SELECTION COMPLETE!") 