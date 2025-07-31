import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🎯 PROJECT SCALE PRICING ANALYSIS")
print("=" * 60)
print("Understanding how price changes with project size & scale")

# Load dataset with all features
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (COMPLETE with ALL features).csv', sep=';', decimal=',')
print(f'Dataset shape: {df.shape}')

# FEATURES REQUESTED BY USER
project_scale_features = [
    # Original volumes
    'shrinkwrap_volume',
    'convex_hull_volume', 
    'bb_volume',
    'Volume',
    
    # Dimensions
    'Max D',
    'Min D',
    
    # Scale indicators
    'Quantity',
    'surface_area',
    
    # Full project volumes (project-level aggregated data)
    'Full_Project_Shrinkwrap_Volume_PA2200',
    'Full_Project_BB_Volume_PA2200',
    'Full_Project_Surface_Area_PA2200', 
    'Full_Project_Volume_PA2200',
    
    # Best derived feature
    'Size_Factor'
]

target = 'Net price / part'

print(f'\n📊 PROJECT SCALE FEATURES:')
for i, feature in enumerate(project_scale_features, 1):
    print(f'{i:2d}. {feature}')

# Convert to numeric
for col in project_scale_features + [target]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# CREATE SIZE_FACTOR (it was missing from the complete dataset)
print('Creating Size_Factor derived feature...')
df['Size_Factor'] = np.cbrt(df['shrinkwrap_volume']) * np.sqrt(df['surface_area'])
print('✓ Size_Factor = ∛(shrinkwrap_volume) × √(surface_area)')

# Clean data
all_features = project_scale_features + [target]
df_clean = df[all_features].dropna()
print(f'\nClean dataset shape: {df_clean.shape}')

# Prepare features and target
X = df_clean[project_scale_features]
y = df_clean[target]

print(f'\nFeatures shape: {X.shape}')
print(f'Target shape: {y.shape}')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for better interpretation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'\n🎯 MODEL PERFORMANCE:')
print(f'R² Score: {r2:.4f} ({r2*100:.1f}%)')
print(f'Mean Absolute Error: €{mae:.2f}')

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': project_scale_features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f'\n🔍 FEATURE IMPORTANCE (Standardized Coefficients):')
print('=' * 70)
print(f'{"Rank":<4} {"Feature":<35} {"Coefficient":<12} {"Impact":<10}')
print('-' * 70)

for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
    direction = '↑ Increases' if row['coefficient'] > 0 else '↓ Decreases'
    coefficient = row['coefficient']
    feature_name = row['feature'][:34]
    
    print(f'{i:<4} {feature_name:<35} {coefficient:>+10.4f} {direction}')

# PROJECT SCALE ANALYSIS - Focus on scale-related features
print(f'\n🏭 PROJECT SCALE IMPACT ANALYSIS:')
print('=' * 50)

scale_features = {
    'Quantity': 'Part quantity (economies of scale)',
    'Full_Project_Volume_PA2200': 'Total project volume',
    'Full_Project_Shrinkwrap_Volume_PA2200': 'Total project shrinkwrap volume',
    'Full_Project_BB_Volume_PA2200': 'Total project bounding box volume',
    'Full_Project_Surface_Area_PA2200': 'Total project surface area'
}

print('Scale Feature Analysis:')
for feature, description in scale_features.items():
    if feature in feature_importance['feature'].values:
        coef = feature_importance[feature_importance['feature'] == feature]['coefficient'].iloc[0]
        direction = 'REDUCES' if coef < 0 else 'INCREASES'
        abs_impact = abs(coef)
        
        print(f'\n• {feature}:')
        print(f'  Description: {description}')
        print(f'  Coefficient: {coef:+.4f}')
        print(f'  Impact: {direction} price (strength: {abs_impact:.4f})')

# QUANTITY ANALYSIS - Specific focus on economies of scale
quantity_coef = feature_importance[feature_importance['feature'] == 'Quantity']['coefficient'].iloc[0]
print(f'\n💰 QUANTITY/ECONOMIES OF SCALE ANALYSIS:')
print(f'Quantity coefficient: {quantity_coef:+.4f}')

if quantity_coef < 0:
    print(f'✅ ECONOMIES OF SCALE CONFIRMED!')
    print(f'   • Each additional unit REDUCES price per part')
    print(f'   • Standardized impact: {abs(quantity_coef):.4f}')
else:
    print(f'❌ No economies of scale detected')
    print(f'   • Higher quantities actually INCREASE price per part')

# PROJECT SIZE ANALYSIS
project_volume_coef = feature_importance[feature_importance['feature'] == 'Full_Project_Volume_PA2200']['coefficient'].iloc[0]
print(f'\n📏 PROJECT SIZE IMPACT:')
print(f'Project volume coefficient: {project_volume_coef:+.4f}')

if project_volume_coef < 0:
    print(f'✅ BIGGER PROJECTS = LOWER PRICES!')
    print(f'   • Larger project volumes reduce price per part')
    print(f'   • Standardized impact: {abs(project_volume_coef):.4f}')
else:
    print(f'❌ Bigger projects actually increase prices')

# PRACTICAL PRICE IMPACT CALCULATION
print(f'\n💡 PRACTICAL PRICE IMPACT EXAMPLES:')
print('=' * 40)

# Get feature means and scales for real-world interpretation
feature_means = scaler.mean_
feature_scales = scaler.scale_

quantity_idx = project_scale_features.index('Quantity')
quantity_mean = feature_means[quantity_idx] 
quantity_scale = feature_scales[quantity_idx]

# Calculate real price impact of doubling quantity
quantity_change = (2 * quantity_mean - quantity_mean) / quantity_scale  # Standardized change
price_impact = quantity_change * quantity_coef

print(f'If Quantity DOUBLES (from {quantity_mean:.1f} to {2*quantity_mean:.1f}):')
print(f'  Price change: €{price_impact:+.2f} per part')

# Project volume impact
if 'Full_Project_Volume_PA2200' in project_scale_features:
    proj_vol_idx = project_scale_features.index('Full_Project_Volume_PA2200')
    proj_vol_mean = feature_means[proj_vol_idx]
    proj_vol_scale = feature_scales[proj_vol_idx]
    
    proj_vol_change = (2 * proj_vol_mean - proj_vol_mean) / proj_vol_scale
    proj_price_impact = proj_vol_change * project_volume_coef
    
    print(f'\nIf Project Volume DOUBLES (from {proj_vol_mean:.0f} to {2*proj_vol_mean:.0f}):')
    print(f'  Price change: €{price_impact:+.2f} per part')

# TOP PRICE DRIVERS
print(f'\n🎯 TOP 5 PRICE DRIVERS:')
top_5 = feature_importance.head(5)
for i, (_, row) in enumerate(top_5.iterrows(), 1):
    direction = '↑' if row['coefficient'] > 0 else '↓'
    print(f'{i}. {row["feature"]} {direction} (impact: {row["abs_coefficient"]:.4f})')

# SAVE RESULTS
results = {
    'model_performance': {
        'r2_score': float(r2),
        'mae': float(mae),
        'intercept': float(model.intercept_)
    },
    'feature_coefficients': {
        feature: float(coef) for feature, coef in zip(project_scale_features, model.coef_)
    },
    'scale_analysis': {
        'quantity_coefficient': float(quantity_coef),
        'project_volume_coefficient': float(project_volume_coef),
        'economies_of_scale': bool(quantity_coef < 0),
        'bigger_projects_cheaper': bool(project_volume_coef < 0)
    },
    'feature_scaling': {
        'means': feature_means.tolist(),
        'scales': feature_scales.tolist()
    }
}

import json
with open('project_scale_pricing_model.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n💾 Results saved to: project_scale_pricing_model.json')

print(f'\n🚀 SUMMARY:')
print(f'Model explains {r2*100:.1f}% of price variation')
print(f'Average prediction error: ±€{mae:.2f}')

if quantity_coef < 0:
    print(f'✅ Confirmed: Higher quantities = Lower prices (economies of scale)')
else:
    print(f'❌ No economies of scale detected')

if project_volume_coef < 0:
    print(f'✅ Confirmed: Bigger projects = Lower prices')
else:
    print(f'❌ Bigger projects don\'t reduce prices') 