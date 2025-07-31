import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print('🔍 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS')
print('Including shrinkwrap_volume and convex_hull_volume!')
print('=' * 60)

# Load dataset
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (with derived features).csv', sep=';', decimal=',')
print(f'Dataset shape: {df.shape}')

# ALL GEOMETRIC FEATURES - including the ones I missed!
all_features = [
    # ORIGINAL GEOMETRIC FEATURES
    'Volume', 'surface_area', 'bb_volume', 'convex_hull_volume', 'shrinkwrap_volume',
    'Quantity', 'Max D', 'Min D', 'D Ratio', 'waste', 'waste_ratio', 'shrinkwrap_ratio',
    
    # DERIVED FEATURES  
    'SA_to_Volume_Ratio', 'Volume_Density', 'Material_Efficiency', 'Complexity_Score',
    'Packing_Efficiency', 'Waste_per_Volume', 'Waste_Efficiency', 'Volume_to_SA_Ratio'
]

target = 'Net price / part'

# Clean data
print('\nCleaning data and checking feature availability...')
for col in all_features + [target]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f'⚠️  Feature missing: {col}')

# Check which features we actually have
available_features = [f for f in all_features if f in df.columns]
print(f'Available features: {len(available_features)}/{len(all_features)}')

# Remove NaN and infinite values  
df_clean = df[available_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
print(f'Clean dataset shape: {df_clean.shape}')

X = df_clean[available_features]
y = df_clean[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge regression (best from previous analysis)
model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'\n🏆 MODEL PERFORMANCE:')
print(f'R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)')
print(f'MAE: €{mae:.2f}')

# COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS
print(f'\n📈 COMPLETE FEATURE IMPORTANCE RANKING:')
print('=' * 60)

importance_df = pd.DataFrame({
    'feature': available_features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print('Rank | Feature                    | Coefficient |   Impact')
print('-' * 60)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    direction = '↑' if row['coefficient'] > 0 else '↓'
    feature_name = row['feature'][:25].ljust(25)
    
    # Highlight the key geometric features
    if row['feature'] in ['surface_area', 'shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'bb_volume']:
        marker = '🔥' if i <= 5 else '⭐'
    else:
        marker = '  '
        
    print(f'{i:3d}  | {marker} {feature_name} {direction} | {row["coefficient"]:10.4f}')

# SPECIFIC COMPARISON: SURFACE AREA vs SHRINKWRAP vs CONVEX HULL
print(f'\n🎯 KEY GEOMETRIC FEATURES COMPARISON:')
print('=' * 50)

key_features = ['surface_area', 'shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'bb_volume']
key_comparison = importance_df[importance_df['feature'].isin(key_features)].copy()

print('Feature               | Coefficient | Rank | Relative Importance')
print('-' * 65)
for _, row in key_comparison.iterrows():
    direction = '↑' if row['coefficient'] > 0 else '↓'
    rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
    relative_importance = row['abs_coefficient'] / importance_df['abs_coefficient'].max() * 100
    
    feature_name = row['feature'][:18].ljust(18)
    print(f'{feature_name} {direction} | {row["coefficient"]:10.4f} | {rank:4d} | {relative_importance:6.1f}%')

# CORRELATION ANALYSIS with target
print(f'\n📊 CORRELATION WITH PRICE:')
print('=' * 40)
correlations = []
for feature in key_features:
    if feature in df_clean.columns:
        corr = df_clean[feature].corr(df_clean[target])
        correlations.append((feature, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print('Feature               | Correlation with Price')
print('-' * 45)
for feature, corr in correlations:
    feature_name = feature[:18].ljust(18)
    print(f'{feature_name} | {corr:8.4f}')

# STATISTICAL SUMMARY of key features
print(f'\n📋 FEATURE STATISTICS:')
print('=' * 40)
stats_features = ['surface_area', 'shrinkwrap_volume', 'convex_hull_volume']
for feature in stats_features:
    if feature in df_clean.columns:
        mean_val = df_clean[feature].mean()
        std_val = df_clean[feature].std()
        print(f'{feature[:20]:20}: Mean={mean_val:10.2f}, Std={std_val:10.2f}')

print(f'\n🤔 INTERPRETATION:')
print('-' * 30)
print('• Higher coefficient = stronger impact on pricing')
print('• Correlation shows direct linear relationship with price')
print('• Both metrics together reveal true feature importance')

# Check if shrinkwrap and convex hull are actually important
shrink_rank = importance_df[importance_df['feature'] == 'shrinkwrap_volume'].index[0] + 1 if 'shrinkwrap_volume' in importance_df['feature'].values else 'N/A'
convex_rank = importance_df[importance_df['feature'] == 'convex_hull_volume'].index[0] + 1 if 'convex_hull_volume' in importance_df['feature'].values else 'N/A'
surface_rank = importance_df[importance_df['feature'] == 'surface_area'].index[0] + 1 if 'surface_area' in importance_df['feature'].values else 'N/A'

print(f'\n✅ FINAL VERDICT:')
print('-' * 20)
print(f'Surface Area rank: #{surface_rank}')
print(f'Shrinkwrap Volume rank: #{shrink_rank}')
print(f'Convex Hull Volume rank: #{convex_rank}')
print('\nThe user was RIGHT to question this! Let me see the true ranking...') 