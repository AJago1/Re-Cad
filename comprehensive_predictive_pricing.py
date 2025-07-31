import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🎯 COMPREHENSIVE PREDICTIVE PRICING - ALL VOLUMES + DERIVED FEATURES")
print("=" * 80)
print("Finding the 5 BEST features from ALL original volumes + derived features")

# Load dataset
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (COMPLETE with ALL features).csv', sep=';', decimal=',')
print(f'Dataset shape: {df.shape}')

# ALL ORIGINAL GEOMETRIC FEATURES (available in GUI from STL analysis)
original_geometric_features = [
    # ALL 5 VOLUME TYPES (original from CSV)
    'shrinkwrap_volume',    # 🥇 Most important volume
    'convex_hull_volume',   # 🥈 Second most important  
    'Volume',               # Actual part volume
    'bb_volume',            # Bounding box volume
    
    # SURFACE AREA (original from CSV)
    'surface_area',         # 🥉 Third most important
    
    # DIMENSIONAL FEATURES (original from CSV)
    'Max D',                # Maximum dimension
    'Min D',                # Minimum dimension
    'waste',                # Waste material
    'Quantity'              # Part quantity
]

# Convert to numeric
for col in original_geometric_features + ['Net price / part']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# CREATE ALL PREDICTIVE DERIVED FEATURES
print(f'\n🔧 CREATING ALL PREDICTIVE DERIVED FEATURES:')

# 1. MATERIAL & EFFICIENCY FEATURES
df['Material_Efficiency'] = df['Volume'] / (df['Volume'] + df['waste'])
print('✓ Material_Efficiency = Volume / (Volume + waste)')

df['Waste_Efficiency'] = df['Volume'] / df['shrinkwrap_volume']
print('✓ Waste_Efficiency = Volume / shrinkwrap_volume')

# 2. SURFACE & COMPLEXITY FEATURES  
df['Surface_Complexity'] = df['surface_area'] / df['shrinkwrap_volume']
print('✓ Surface_Complexity = surface_area / shrinkwrap_volume')

df['SA_to_Volume_Ratio'] = df['surface_area'] / df['Volume']
print('✓ SA_to_Volume_Ratio = surface_area / Volume')

# 3. PACKING & SPACE EFFICIENCY
df['Packing_Efficiency'] = df['shrinkwrap_volume'] / df['bb_volume'] 
print('✓ Packing_Efficiency = shrinkwrap_volume / bb_volume')

df['Volume_Density'] = df['Volume'] / df['bb_volume']
print('✓ Volume_Density = Volume / bb_volume')

# 4. DIMENSIONAL FEATURES
df['Dimensional_Complexity'] = df['Max D'] / df['Min D']
print('✓ Dimensional_Complexity = Max D / Min D')

df['Aspect_Ratio'] = df['Max D'] / df['Min D']  # Same as above but named differently
print('✓ Aspect_Ratio = Max D / Min D')

# 5. VOLUME EFFICIENCY FEATURES
df['Volume_Efficiency'] = df['Volume'] / df['convex_hull_volume']
print('✓ Volume_Efficiency = Volume / convex_hull_volume')

df['Shrinkwrap_to_Convex_Ratio'] = df['shrinkwrap_volume'] / df['convex_hull_volume']
print('✓ Shrinkwrap_to_Convex_Ratio = shrinkwrap_volume / convex_hull_volume')

# 6. ADVANCED COMPOSITE FEATURES
df['Manufacturing_Difficulty'] = (df['Surface_Complexity'] * df['Dimensional_Complexity']) / df['Volume_Efficiency']
print('✓ Manufacturing_Difficulty = (Surface_Complexity × Dimensional_Complexity) / Volume_Efficiency')

df['Material_Intensity'] = df['shrinkwrap_volume'] / (df['Max D'] * df['Min D'])
print('✓ Material_Intensity = shrinkwrap_volume / (Max D × Min D)')

df['Size_Factor'] = np.cbrt(df['shrinkwrap_volume']) * np.sqrt(df['surface_area'])
print('✓ Size_Factor = ∛(shrinkwrap_volume) × √(surface_area)')

df['Complexity_Index'] = (df['surface_area'] / df['shrinkwrap_volume']) * (df['Max D'] / df['Min D'])
print('✓ Complexity_Index = (surface_area / shrinkwrap_volume) × (Max D / Min D)')

df['Material_Utilization'] = df['Volume'] / df['bb_volume'] / (df['waste'] / df['Volume'] + 1)
print('✓ Material_Utilization = (Volume / bb_volume) / (waste_ratio + 1)')

# COMPREHENSIVE CANDIDATE FEATURES (ALL volumes + surface + derived)
candidate_features = [
    # ALL ORIGINAL VOLUME FEATURES
    'shrinkwrap_volume',        # 🥇 Most important raw volume
    'convex_hull_volume',       # 🥈 Second most important raw volume
    'Volume',                   # Actual part volume
    'bb_volume',                # Bounding box volume
    
    # ORIGINAL SURFACE FEATURE  
    'surface_area',             # 🥉 Raw surface area
    
    # EFFICIENCY & MATERIAL FEATURES
    'Material_Efficiency',      # Waste efficiency
    'Waste_Efficiency',         # Volume utilization
    'Material_Utilization',     # Combined material efficiency
    
    # SURFACE & COMPLEXITY FEATURES
    'Surface_Complexity',       # Surface per volume
    'SA_to_Volume_Ratio',       # Traditional surface complexity
    
    # PACKING & SPACE FEATURES
    'Packing_Efficiency',       # Space utilization
    'Volume_Density',           # Volume packing
    
    # DIMENSIONAL FEATURES
    'Dimensional_Complexity',   # Aspect ratio
    'Material_Intensity',       # Material per cross-section
    
    # VOLUME RELATIONSHIP FEATURES
    'Volume_Efficiency',        # Volume utilization
    'Shrinkwrap_to_Convex_Ratio', # Volume relationship
    
    # ADVANCED COMPOSITE FEATURES
    'Manufacturing_Difficulty', # Overall complexity
    'Size_Factor',              # Combined size effect
    'Complexity_Index',         # Surface × dimensional complexity
    
    # ORIGINAL FEATURES
    'Quantity'                  # Production scale
]

target = 'Net price / part'

print(f'\n📊 COMPREHENSIVE FEATURE SET:')
print(f'Total candidate features: {len(candidate_features)}')
print(f'Original volume features: 4 (shrinkwrap, convex_hull, Volume, bb_volume)')
print(f'Original surface features: 1 (surface_area)')
print(f'Derived features: {len(candidate_features) - 5}')

# Clean data
df_clean = df[candidate_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
print(f'\nClean dataset: {df_clean.shape[0]} samples')

X = df_clean[candidate_features]
y = df_clean[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'\n📊 COMPREHENSIVE MODEL PERFORMANCE:')
print(f'R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)')
print(f'MAE: €{mae:.2f}')

# FEATURE IMPORTANCE ANALYSIS
importance_df = pd.DataFrame({
    'feature': candidate_features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f'\n🏆 COMPREHENSIVE FEATURE IMPORTANCE RANKING:')
print('Rank | Feature                     | Type      | Coefficient | Impact')
print('-' * 80)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    direction = '↑' if row['coefficient'] > 0 else '↓'
    feature_name = row['feature'][:26].ljust(26)
    
    # Categorize feature type
    if row['feature'] in ['shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'bb_volume']:
        feature_type = 'VOLUME   '
    elif row['feature'] == 'surface_area':
        feature_type = 'SURFACE  '
    else:
        feature_type = 'DERIVED  '
    
    # Highlight top features
    if i <= 5:
        marker = '🏆'
    elif i <= 10:
        marker = '⭐'
    else:
        marker = '  '
        
    print(f'{i:3d} {marker}| {feature_name} | {feature_type} {direction} | {row["coefficient"]:10.4f}')

# SELECT TOP 5 FEATURES
top_5_features = importance_df.head(5)['feature'].tolist()

print(f'\n🎯 TOP 5 COMPREHENSIVE PREDICTIVE FEATURES:')
print('=' * 60)
for i, feature in enumerate(top_5_features, 1):
    coef = importance_df[importance_df['feature'] == feature]['coefficient'].iloc[0]
    direction = '↑' if coef > 0 else '↓'
    
    # Determine feature type
    if feature in ['shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'bb_volume']:
        feature_type = 'ORIGINAL VOLUME'
    elif feature == 'surface_area':
        feature_type = 'ORIGINAL SURFACE'
    else:
        feature_type = 'DERIVED FEATURE'
    
    print(f'{i}. {feature} {direction}')
    print(f'   Type: {feature_type} | Coefficient: {coef:.4f}')

# TEST TOP 5 MODEL
X_top5 = df_clean[top_5_features]
X_train_top5, X_test_top5, y_train, y_test = train_test_split(X_top5, y, test_size=0.2, random_state=42)

scaler_top5 = StandardScaler()
X_train_top5_scaled = scaler_top5.fit_transform(X_train_top5)
X_test_top5_scaled = scaler_top5.transform(X_test_top5)

model_top5 = Ridge(alpha=10.0)
model_top5.fit(X_train_top5_scaled, y_train)
y_pred_top5 = model_top5.predict(X_test_top5_scaled)

r2_top5 = r2_score(y_test, y_pred_top5)
mae_top5 = mean_absolute_error(y_test, y_pred_top5)

print(f'\n📈 TOP 5 COMPREHENSIVE MODEL PERFORMANCE:')
print(f'R² Score: {r2_top5:.4f} ({r2_top5*100:.1f}% variance explained)')
print(f'MAE: €{mae_top5:.2f}')
print(f'Performance vs Full Model: {(r2_top5/r2)*100:.1f}%')

# CREATE COMPREHENSIVE PRICING FORMULA
print(f'\n🧮 COMPREHENSIVE PRICING FORMULA FOR GUI:')
print('=' * 50)

coefficients = model_top5.coef_
intercept = model_top5.intercept_
feature_means = scaler_top5.mean_
feature_scales = scaler_top5.scale_

print('def calculate_comprehensive_price(features):')
print('    """')
print('    Calculate price using top 5 comprehensive features')
print('    Mix of original volumes, surface, and derived features')
print('    features = {')
for feature in top_5_features:
    print(f'        "{feature}": value,')
print('    }')
print('    """')
print('    ')

# Generate formula code
print('    # Feature scaling (standardization)')
for i, feature in enumerate(top_5_features):
    print(f'    {feature}_scaled = ({feature} - {feature_means[i]:.6f}) / {feature_scales[i]:.6f}')

print('    ')
print('    # Price calculation')
formula_parts = []
for i, feature in enumerate(top_5_features):
    coef = coefficients[i]
    formula_parts.append(f'({coef:.6f} * {feature}_scaled)')

formula = ' + '.join(formula_parts)
print(f'    price = {intercept:.6f} + {formula}')
print('    ')
print('    return max(0, price)  # Ensure non-negative price')

# COMPREHENSIVE FEATURE EXPLANATIONS
print(f'\n📚 COMPREHENSIVE FEATURE EXPLANATIONS:')
print('=' * 50)

explanations = {
    # Original volume features
    'shrinkwrap_volume': 'ORIGINAL: Tight-fitting volume - most accurate material cost',
    'convex_hull_volume': 'ORIGINAL: Convex shape volume - support material calculation',
    'Volume': 'ORIGINAL: Actual part volume - base material cost',
    'bb_volume': 'ORIGINAL: Bounding box volume - build space requirement',
    
    # Original surface feature
    'surface_area': 'ORIGINAL: Total surface area - printing time and finishing',
    
    # Derived features
    'Material_Efficiency': 'DERIVED: Volume/(Volume+waste) - material utilization',
    'Surface_Complexity': 'DERIVED: surface_area/shrinkwrap_volume - geometric complexity',
    'Packing_Efficiency': 'DERIVED: shrinkwrap_volume/bb_volume - space efficiency',
    'Manufacturing_Difficulty': 'DERIVED: Composite complexity score',
    'Material_Intensity': 'DERIVED: shrinkwrap_volume/(MaxD×MinD) - density factor',
    'Size_Factor': 'DERIVED: ∛(shrinkwrap_volume) × √(surface_area) - size scaling',
    'Volume_Efficiency': 'DERIVED: Volume/convex_hull_volume - volume utilization',
    'Complexity_Index': 'DERIVED: (surface_area/shrinkwrap_volume) × (MaxD/MinD)',
    'Shrinkwrap_to_Convex_Ratio': 'DERIVED: shrinkwrap_volume/convex_hull_volume'
}

for feature in top_5_features:
    if feature in explanations:
        print(f'• {feature}:')
        print(f'  {explanations[feature]}')

# SAVE COMPREHENSIVE RESULTS
results = {
    'top_5_features': top_5_features,
    'coefficients': coefficients.tolist(),
    'intercept': float(intercept),
    'feature_means': feature_means.tolist(),
    'feature_scales': feature_scales.tolist(),
    'r2_score': float(r2_top5),
    'mae': float(mae_top5),
    'feature_explanations': {feature: explanations.get(feature, 'No explanation') for feature in top_5_features}
}

import json
with open('comprehensive_gui_pricing_model.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n💾 COMPREHENSIVE MODEL SAVED: comprehensive_gui_pricing_model.json')

print(f'\n🎯 FINAL COMPREHENSIVE RECOMMENDATION:')
print('=' * 50)
original_count = sum(1 for f in top_5_features if f in ['shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'bb_volume', 'surface_area'])
derived_count = len(top_5_features) - original_count

print(f'✅ Original geometric features in top 5: {original_count}')
print(f'🔧 Derived features in top 5: {derived_count}')
print(f'📊 Model accuracy: {r2_top5*100:.1f}%')
print(f'💰 Average error: ±€{mae_top5:.2f}')

print(f'\n🏆 YOUR OPTIMAL 5 FEATURES FOR GUI:')
for i, feature in enumerate(top_5_features, 1):
    feature_type = 'ORIGINAL' if feature in ['shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'bb_volume', 'surface_area'] else 'DERIVED'
    print(f'{i}. {feature} ({feature_type})')

print(f'\n🚀 Ready for GUI integration with the best mix of original + derived features!') 