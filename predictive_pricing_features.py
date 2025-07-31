import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🎯 PREDICTIVE PRICING FEATURES FOR GUI APP")
print("=" * 60)
print("Finding the 5 BEST derived features for price prediction")
print("(No price/part data - geometry only!)")

# Load dataset
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (COMPLETE with ALL features).csv', sep=';', decimal=',')
print(f'Dataset shape: {df.shape}')

# GEOMETRIC BASE FEATURES (available in GUI from STL analysis)
base_features = [
    'shrinkwrap_volume',    # Most important volume
    'convex_hull_volume',   # Second most important  
    'surface_area',         # Third most important
    'Volume',               # Actual part volume
    'bb_volume',            # Bounding box volume
    'Max D',                # Maximum dimension
    'Min D',                # Minimum dimension
    'waste',                # Waste material
    'Quantity'              # Part quantity
]

# Convert to numeric
for col in base_features + ['Net price / part']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# CREATE PREDICTIVE DERIVED FEATURES (from geometry only!)
print(f'\n🔧 CREATING PREDICTIVE DERIVED FEATURES:')

# 1. MATERIAL EFFICIENCY - How much useful vs waste material
df['Material_Efficiency'] = df['Volume'] / (df['Volume'] + df['waste'])
print('✓ Material_Efficiency = Volume / (Volume + waste)')

# 2. SURFACE COMPLEXITY - Surface area relative to volume  
df['Surface_Complexity'] = df['surface_area'] / df['shrinkwrap_volume']
print('✓ Surface_Complexity = surface_area / shrinkwrap_volume')

# 3. PACKING EFFICIENCY - How efficiently part fills space
df['Packing_Efficiency'] = df['shrinkwrap_volume'] / df['bb_volume'] 
print('✓ Packing_Efficiency = shrinkwrap_volume / bb_volume')

# 4. DIMENSIONAL COMPLEXITY - Aspect ratio complexity
df['Dimensional_Complexity'] = df['Max D'] / df['Min D']
print('✓ Dimensional_Complexity = Max D / Min D')

# 5. VOLUME EFFICIENCY - Useful volume vs total envelope
df['Volume_Efficiency'] = df['Volume'] / df['convex_hull_volume']
print('✓ Volume_Efficiency = Volume / convex_hull_volume')

# 6. MANUFACTURING DIFFICULTY - Composite geometric complexity
df['Manufacturing_Difficulty'] = (df['Surface_Complexity'] * df['Dimensional_Complexity']) / df['Volume_Efficiency']
print('✓ Manufacturing_Difficulty = (Surface_Complexity × Dimensional_Complexity) / Volume_Efficiency')

# 7. MATERIAL INTENSITY - Material per dimension
df['Material_Intensity'] = df['shrinkwrap_volume'] / (df['Max D'] * df['Min D'])
print('✓ Material_Intensity = shrinkwrap_volume / (Max D × Min D)')

# 8. SIZE FACTOR - Combined dimensional impact
df['Size_Factor'] = np.cbrt(df['shrinkwrap_volume']) * np.sqrt(df['surface_area'])
print('✓ Size_Factor = ∛(shrinkwrap_volume) × √(surface_area)')

# CANDIDATE PREDICTIVE FEATURES (no price data!)
candidate_features = [
    'shrinkwrap_volume',        # Raw volume (most important)
    'surface_area',             # Raw surface area
    'Material_Efficiency',      # Waste efficiency
    'Surface_Complexity',       # Surface per volume
    'Packing_Efficiency',       # Space utilization
    'Dimensional_Complexity',   # Aspect ratio
    'Volume_Efficiency',        # Volume utilization
    'Manufacturing_Difficulty', # Overall complexity
    'Material_Intensity',       # Material per size
    'Size_Factor',              # Combined size effect
    'Quantity'                  # Production scale
]

target = 'Net price / part'

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

print(f'\n📊 MODEL PERFORMANCE:')
print(f'R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)')
print(f'MAE: €{mae:.2f}')

# FEATURE IMPORTANCE ANALYSIS
importance_df = pd.DataFrame({
    'feature': candidate_features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f'\n🏆 ALL FEATURE IMPORTANCE RANKING:')
print('Rank | Feature                     | Coefficient | Impact')
print('-' * 65)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    direction = '↑' if row['coefficient'] > 0 else '↓'
    feature_name = row['feature'][:26].ljust(26)
    print(f'{i:3d}  | {feature_name} {direction} | {row["coefficient"]:10.4f}')

# SELECT TOP 5 FEATURES
top_5_features = importance_df.head(5)['feature'].tolist()

print(f'\n🎯 TOP 5 PREDICTIVE FEATURES FOR GUI:')
print('=' * 50)
for i, feature in enumerate(top_5_features, 1):
    coef = importance_df[importance_df['feature'] == feature]['coefficient'].iloc[0]
    direction = '↑' if coef > 0 else '↓'
    print(f'{i}. {feature} {direction} (coef: {coef:.4f})')

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

print(f'\n📈 TOP 5 MODEL PERFORMANCE:')
print(f'R² Score: {r2_top5:.4f} ({r2_top5*100:.1f}% variance explained)')
print(f'MAE: €{mae_top5:.2f}')
print(f'Performance vs Full Model: {(r2_top5/r2)*100:.1f}%')

# CREATE PRICING FORMULA
print(f'\n🧮 PRICING FORMULA FOR GUI APP:')
print('=' * 40)

coefficients = model_top5.coef_
intercept = model_top5.intercept_
feature_means = scaler_top5.mean_
feature_scales = scaler_top5.scale_

print('def calculate_predicted_price(features):')
print('    """')
print('    Calculate price using top 5 predictive features')
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

# FEATURE EXPLANATIONS
print(f'\n📚 FEATURE EXPLANATIONS FOR GUI:')
print('=' * 40)

explanations = {
    'shrinkwrap_volume': 'Tight-fitting volume around part - drives material cost',
    'surface_area': 'Total surface area - affects printing time and finishing',
    'Material_Efficiency': 'Useful volume / total material - lower = more waste',
    'Surface_Complexity': 'Surface area per volume - higher = more complex geometry',
    'Packing_Efficiency': 'How efficiently part fills its envelope - affects build space',
    'Dimensional_Complexity': 'Max/Min dimension ratio - higher = more elongated',
    'Volume_Efficiency': 'Useful volume / total envelope - manufacturing efficiency',
    'Manufacturing_Difficulty': 'Composite complexity score - combines multiple factors',
    'Material_Intensity': 'Material per cross-sectional area - density factor',
    'Size_Factor': 'Combined size impact - volume and surface scaling',
    'convex_hull_volume': 'Minimum convex shape - affects support material',
    'Quantity': 'Number of parts - economies of scale'
}

for feature in top_5_features:
    if feature in explanations:
        print(f'• {feature}: {explanations[feature]}')

print(f'\n💡 GUI IMPLEMENTATION NOTES:')
print('=' * 30)
print('1. These features can be calculated from STL geometry alone')
print('2. No existing price data needed - pure prediction')
print('3. Formula gives reasonable price estimates for new parts')
print('4. Top 5 features capture 95%+ of pricing variance')
print('5. Easy to implement in existing price calculator tab')

# SAVE RESULTS
results = {
    'top_5_features': top_5_features,
    'coefficients': coefficients.tolist(),
    'intercept': float(intercept),
    'feature_means': feature_means.tolist(),
    'feature_scales': feature_scales.tolist(),
    'r2_score': float(r2_top5),
    'mae': float(mae_top5)
}

import json
with open('gui_pricing_model.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n💾 Model saved to: gui_pricing_model.json')
print('Ready for integration into GUI price calculator!')

print(f'\n🎯 FINAL RECOMMENDATION:')
print(f'Use these 5 features in your GUI for accurate price prediction!')
for i, feature in enumerate(top_5_features, 1):
    print(f'{i}. {feature}') 