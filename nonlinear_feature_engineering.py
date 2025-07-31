"""
Advanced Non-Linear Feature Engineering for STL Pricing Model
Implements comprehensive derived features to improve pricing accuracy across all part sizes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 ADVANCED NON-LINEAR FEATURE ENGINEERING")
print("=" * 60)

# Load the dataset
print("📊 Loading dataset...")
df = pd.read_csv('correct totals 3.csv', sep=';', decimal=',')
print(f"Dataset shape: {df.shape}")

# Clean and prepare base features
print("\n🔧 Preparing base features...")
base_features = [
    'shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'Min D', 'Max D', 
    'D Ratio', 'waste', 'surface_area', 'bb_volume', 'Quantity'
]

target_column = 'Net price / part'

# Convert to numeric and clean
for col in base_features + [target_column]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove invalid data
df_clean = df[base_features + [target_column]].replace([np.inf, -np.inf], np.nan).dropna()
df_clean = df_clean[df_clean[target_column] > 0]  # Remove zero/negative prices

print(f"Clean dataset: {len(df_clean)} samples")

def create_nonlinear_features(df):
    """Create comprehensive non-linear derived features"""
    features_df = df.copy()
    
    print("🎯 Creating non-linear derived features...")
    
    # Extract base values
    volume = features_df['Volume']
    surface_area = features_df['surface_area']
    shrinkwrap_vol = features_df['shrinkwrap_volume']
    convex_hull_vol = features_df['convex_hull_volume']
    bb_volume = features_df['bb_volume']
    max_dim = features_df['Max D']
    min_dim = features_df['Min D']
    waste = features_df['waste']
    quantity = features_df['Quantity']
    
    # 1. SIZE-DEPENDENT FEATURES
    print("   • Size-dependent features...")
    features_df['size_factor'] = volume ** (1/3)  # Cubic root scaling
    features_df['micro_part_penalty'] = 1 / (1 + volume/1000)  # High for small, low for large
    features_df['large_part_efficiency'] = volume / (volume + 10000)  # Efficiency bonus for larger
    features_df['size_category_micro'] = (volume < 1000).astype(int)
    features_df['size_category_small'] = ((volume >= 1000) & (volume < 10000)).astype(int)
    features_df['size_category_medium'] = ((volume >= 10000) & (volume < 50000)).astype(int)
    features_df['size_category_large'] = (volume >= 50000).astype(int)
    
    # 2. LOGARITHMIC TRANSFORMATIONS
    print("   • Logarithmic transformations...")
    features_df['log_volume'] = np.log1p(volume)
    features_df['log_surface_area'] = np.log1p(surface_area)
    features_df['log_shrinkwrap_volume'] = np.log1p(shrinkwrap_vol)
    features_df['log_convex_hull_volume'] = np.log1p(convex_hull_vol)
    features_df['log_bb_volume'] = np.log1p(bb_volume)
    
    # 3. POWER TRANSFORMATIONS
    print("   • Power transformations...")
    features_df['volume_sqrt'] = volume ** 0.5
    features_df['volume_cbrt'] = volume ** (1/3)
    features_df['surface_area_sqrt'] = surface_area ** 0.5
    features_df['shrinkwrap_sqrt'] = shrinkwrap_vol ** 0.5
    
    # 4. COMPLEXITY-SIZE INTERACTIONS
    print("   • Complexity-size interactions...")
    features_df['complexity_size_ratio'] = (surface_area / np.maximum(volume, 1)) * features_df['micro_part_penalty']
    features_df['dimension_instability'] = (max_dim / np.maximum(min_dim, 0.1)) * features_df['micro_part_penalty']
    features_df['waste_complexity'] = (waste / np.maximum(volume, 1)) * features_df['micro_part_penalty']
    features_df['surface_volume_interaction'] = (surface_area * volume) ** 0.5
    features_df['complexity_volume_interaction'] = (surface_area / np.maximum(volume, 1)) * features_df['log_volume']
    
    # 5. ECONOMIC BEHAVIOR FEATURES
    print("   • Economic behavior features...")
    features_df['fixed_cost_impact'] = 1000 / np.maximum(volume, 100)  # Fixed costs hit small parts harder
    features_df['handling_complexity'] = 1 / (shrinkwrap_vol ** 0.3)
    features_df['material_waste_curve'] = (waste / np.maximum(bb_volume, 1)) * (1 + features_df['micro_part_penalty'])
    features_df['support_material_factor'] = ((bb_volume - volume) / np.maximum(volume, 1)) * features_df['micro_part_penalty']
    
    # 6. MANUFACTURING EFFICIENCY FEATURES
    print("   • Manufacturing efficiency features...")
    features_df['scale_efficiency'] = quantity * (volume ** 0.3)
    features_df['batch_efficiency'] = quantity / (1 + 1000/np.maximum(volume, 1))
    features_df['manufacturing_challenge'] = (surface_area / np.maximum(volume, 1)) / (volume ** 0.2)
    features_df['precision_requirement'] = (max_dim / np.maximum(min_dim, 0.1)) / (volume ** 0.1)
    
    # 7. MATERIAL UTILIZATION FEATURES
    print("   • Material utilization features...")
    features_df['material_density'] = volume / np.maximum(bb_volume, 1)
    features_df['packing_efficiency'] = shrinkwrap_vol / np.maximum(bb_volume, 1)
    features_df['shape_complexity'] = surface_area / (volume ** (2/3))  # Surface area to volume^(2/3) ratio
    features_df['convexity_efficiency'] = volume / np.maximum(convex_hull_vol, 1)
    
    # 8. DIMENSIONAL ANALYSIS FEATURES
    print("   • Dimensional analysis features...")
    features_df['aspect_ratio_penalty'] = np.maximum(max_dim/np.maximum(min_dim, 0.1) - 1, 0) ** 0.5
    features_df['dimensional_balance'] = min_dim * max_dim / np.maximum(volume, 1)
    features_df['slenderness_ratio'] = max_dim / (volume ** (1/3))
    
    # 9. QUANTITY-DEPENDENT FEATURES
    print("   • Quantity-dependent features...")
    features_df['quantity_volume_interaction'] = quantity * features_df['log_volume']
    features_df['quantity_complexity_interaction'] = quantity * (surface_area / np.maximum(volume, 1))
    features_df['quantity_efficiency'] = np.log1p(quantity) * features_df['large_part_efficiency']
    
    # 10. ADVANCED INTERACTION TERMS
    print("   • Advanced interaction terms...")
    features_df['volume_surface_complexity'] = (volume * surface_area * (max_dim/np.maximum(min_dim, 0.1))) ** (1/3)
    features_df['geometric_mean_dims'] = (max_dim * min_dim) ** 0.5
    features_df['harmonic_mean_efficiency'] = 2 / (1/np.maximum(volume, 1) + 1/np.maximum(surface_area, 1))
    
    print(f"   ✅ Created {len(features_df.columns) - len(df.columns)} new derived features")
    return features_df

# Create enhanced dataset with non-linear features
df_enhanced = create_nonlinear_features(df_clean)

# Get all feature columns (exclude target)
feature_columns = [col for col in df_enhanced.columns if col != target_column]
print(f"\n📈 Total features available: {len(feature_columns)}")

# Prepare data for modeling
X = df_enhanced[feature_columns]
y = df_enhanced[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n🎯 Training set: {len(X_train)} samples")
print(f"🎯 Test set: {len(X_test)} samples")

# Scale features
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use LassoCV to find optimal alpha with cross-validation
print("\n🔍 Finding optimal Lasso alpha with cross-validation...")
alphas = np.logspace(-4, 1, 50)  # Test alphas from 0.0001 to 10
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=3000)
lasso_cv.fit(X_train_scaled, y_train)

optimal_alpha = lasso_cv.alpha_
print(f"✅ Optimal alpha: {optimal_alpha:.6f}")

# Train final model with optimal alpha
print("\n🏗️ Training enhanced Lasso model...")
lasso_enhanced = Lasso(alpha=optimal_alpha, random_state=42, max_iter=3000)
lasso_enhanced.fit(X_train_scaled, y_train)

# Evaluate model
y_pred_train = lasso_enhanced.predict(X_train_scaled)
y_pred_test = lasso_enhanced.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n📊 ENHANCED MODEL PERFORMANCE:")
print(f"   Training R²: {train_r2:.4f}")
print(f"   Test R²: {test_r2:.4f}")
print(f"   Training MAE: €{train_mae:.2f}")
print(f"   Test MAE: €{test_mae:.2f}")

# Get feature importance (non-zero coefficients)
feature_importance = []
for feature, coef in zip(feature_columns, lasso_enhanced.coef_):
    if abs(coef) > 1e-6:  # Only include non-zero coefficients
        feature_importance.append((feature, abs(coef), coef))

feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"\n🎯 SELECTED FEATURES: {len(feature_importance)} out of {len(feature_columns)}")
print("\nTop 15 Most Important Features:")
for i, (feature, abs_coef, coef) in enumerate(feature_importance[:15]):
    direction = "↗️" if coef > 0 else "↘️"
    print(f"   {i+1:2d}. {feature:<35} {direction} {abs_coef:8.3f} (coef: {coef:8.3f})")

# Analyze performance by size category
print(f"\n📈 PERFORMANCE BY SIZE CATEGORY:")
size_categories = [
    ("Micro (< 1000 mm³)", df_enhanced['Volume'] < 1000),
    ("Small (1000-10000 mm³)", (df_enhanced['Volume'] >= 1000) & (df_enhanced['Volume'] < 10000)),
    ("Medium (10000-50000 mm³)", (df_enhanced['Volume'] >= 10000) & (df_enhanced['Volume'] < 50000)),
    ("Large (≥ 50000 mm³)", df_enhanced['Volume'] >= 50000)
]

# Get predictions for full dataset
X_full_scaled = scaler.transform(X)
y_pred_full = lasso_enhanced.predict(X_full_scaled)

for category_name, mask in size_categories:
    if mask.sum() > 0:
        y_true_cat = y[mask]
        y_pred_cat = y_pred_full[mask]
        cat_r2 = r2_score(y_true_cat, y_pred_cat)
        cat_mae = mean_absolute_error(y_true_cat, y_pred_cat)
        avg_price = y_true_cat.mean()
        print(f"   {category_name:<25} R²: {cat_r2:.3f}, MAE: €{cat_mae:.2f}, Avg Price: €{avg_price:.2f} ({mask.sum()} parts)")

# Save enhanced model
print(f"\n💾 Saving enhanced model...")
import pickle
import json

# Save model and scaler
with open('enhanced_lasso_model.pkl', 'wb') as f:
    pickle.dump(lasso_enhanced, f)

with open('enhanced_feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save model metadata
selected_features = [feature for feature, _, _ in feature_importance]
model_metadata = {
    'model_type': 'Enhanced Lasso with Non-Linear Features',
    'optimal_alpha': optimal_alpha,
    'selected_features': selected_features,
    'feature_coefficients': {feature: coef for feature, _, coef in feature_importance},
    'performance': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    },
    'feature_count': len(selected_features),
    'total_features_tested': len(feature_columns),
    'training_samples': len(X_train)
}

with open('enhanced_model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"✅ Enhanced model saved!")
print(f"   Model file: enhanced_lasso_model.pkl")
print(f"   Scaler file: enhanced_feature_scaler.pkl") 
print(f"   Metadata file: enhanced_model_metadata.json")

# Compare with original model performance
print(f"\n📊 COMPARISON WITH ORIGINAL MODEL:")
print(f"   Original R²: ~0.863 → Enhanced R²: {test_r2:.3f} ({((test_r2-0.863)/0.863*100):+.1f}%)")
print(f"   Original MAE: ~€20.24 → Enhanced MAE: €{test_mae:.2f} ({((test_mae-20.24)/20.24*100):+.1f}%)")
print(f"   Original Features: 12 → Enhanced Features: {len(selected_features)} ({len(selected_features)-12:+d})")

print(f"\n🎉 ENHANCED NON-LINEAR FEATURE ENGINEERING COMPLETE!")
print(f"   Ready for integration into GUI pricing system") 