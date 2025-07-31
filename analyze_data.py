import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('correct totals 1.csv', sep=';')

# Convert price to numeric (handle European decimal format)
df['price'] = pd.to_numeric(df['Net price / part'].str.replace(',', '.'), errors='coerce')

# Convert other numeric columns 
numeric_cols = ['convex_hull_volume', 'bb_volume', 'surface_area', 'Quantity', 
                'Max D', 'Min D', 'D Ratio', 'waste', 'shrinkwrap_volume', 
                'Volume', 'Total_Parts_in_Project', 'Project_Total_convex_hull_volume',
                'Project_Total_bb_volume', 'Project_Total_shrinkwrap_volume', 
                'Project_Total_Volume', 'Project_Total_surface_area']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print("=== DATA OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Unique projects: {df['Project ID'].nunique()}")
print(f"Price range: €{df['price'].min():.2f} to €{df['price'].max():.2f}")
print(f"Median price: €{df['price'].median():.2f}")
print(f"Parts per project range: {df['Total_Parts_in_Project'].min()} to {df['Total_Parts_in_Project'].max()}")

print("\n=== BULK PRICING ANALYSIS ===")
# Group by quantity bins to see bulk effects
df['qty_bin'] = pd.cut(df['Quantity'], bins=[0, 1, 5, 10, 25, 50, 100, 500, np.inf], 
                       labels=['1', '2-5', '6-10', '11-25', '26-50', '51-100', '101-500', '500+'])
bulk_stats = df.groupby('qty_bin')['price'].agg(['count', 'mean', 'median', 'std']).round(2)
print(bulk_stats)

print("\n=== PROJECT SIZE EFFECTS ===")
# Analyze if bigger projects get discounts
df['project_size_bin'] = pd.cut(df['Total_Parts_in_Project'], 
                               bins=[0, 5, 15, 50, 150, np.inf],
                               labels=['1-5', '6-15', '16-50', '51-150', '150+'])
project_stats = df.groupby('project_size_bin')['price'].agg(['count', 'mean', 'median']).round(2)
print(project_stats)

print("\n=== GEOMETRIC CORRELATIONS WITH PRICE ===")
geometric_features = ['convex_hull_volume', 'bb_volume', 'surface_area', 'Volume', 'waste', 'D Ratio']
correlations = df[geometric_features + ['price']].corr()['price'].sort_values(ascending=False)
print(correlations)

print("\n=== DERIVED FEATURES ANALYSIS ===")
# Create derived features
df['volume_efficiency'] = df['Volume'] / df['bb_volume']  # How efficiently space is used
df['surface_to_volume'] = df['surface_area'] / df['Volume']  # Surface complexity
df['waste_ratio'] = df['waste'] / df['Volume']  # Material waste efficiency
df['price_per_mm3'] = df['price'] / df['Volume']  # Price density
df['volume_density'] = df['Volume'] / df['convex_hull_volume']  # Part density
df['project_volume_share'] = df['Volume'] / df['Project_Total_Volume']  # Share of project volume

# Calculate part-level vs project-level ratios
df['vs_project_avg_volume'] = df['Volume'] / (df['Project_Total_Volume'] / df['Total_Parts_in_Project'])

derived_features = ['volume_efficiency', 'surface_to_volume', 'waste_ratio', 
                   'volume_density', 'project_volume_share', 'vs_project_avg_volume']

print("Derived feature correlations with price:")
derived_corr = df[derived_features + ['price']].corr()['price'].sort_values(ascending=False)
print(derived_corr)

print("\n=== MACHINE & MATERIAL ANALYSIS ===")
machine_stats = df.groupby('Machine')['price'].agg(['count', 'mean', 'median']).round(2)
print("By Machine:")
print(machine_stats)

material_stats = df.groupby('Material')['price'].agg(['count', 'mean']).round(2)
print("\nBy Material:")
print(material_stats)

print("\n=== OUTLIER ANALYSIS ===")
# Find parts with unusual price/volume ratios
df['price_per_volume_zscore'] = np.abs(stats.zscore(df['price_per_mm3'].fillna(0)))
outliers = df[df['price_per_volume_zscore'] > 2][['Project ID', 'Part Name', 'price', 'Volume', 'price_per_mm3']]
print(f"High price/volume outliers (Z-score > 2): {len(outliers)} parts")
if len(outliers) > 0:
    print(outliers.head(10))

print("\n=== FEATURE IMPORTANCE FOR LINEAR MODEL ===")
# Prepare feature matrix for correlation analysis
feature_cols = ['Volume', 'surface_area', 'convex_hull_volume', 'bb_volume', 
               'D Ratio', 'Quantity', 'Total_Parts_in_Project'] + derived_features

feature_matrix = df[feature_cols + ['price']].dropna()
feature_correlations = feature_matrix.corr()['price'].abs().sort_values(ascending=False)
print("Top features by absolute correlation with price:")
print(feature_correlations.head(15))

# Check for multicollinearity
print("\n=== MULTICOLLINEARITY CHECK ===")
corr_matrix = feature_matrix[feature_cols].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("Highly correlated feature pairs (|r| > 0.8):")
for pair in high_corr_pairs:
    print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

print("\n=== RECOMMENDATIONS ===")
print("1. DERIVED FEATURES TO ADD:")
print("   - volume_efficiency (Volume/bb_volume)")
print("   - surface_to_volume (surface_area/Volume) - complexity metric")
print("   - waste_ratio (waste/Volume) - efficiency metric")
print("   - project_volume_share - for bulk effect modeling")
print("   - log(Volume) - for non-linear volume effects")

print("\n2. BULK DISCOUNT MODELING:")
print("   - Add quantity bins as categorical features")
print("   - Add project size effect (Total_Parts_in_Project)")
print("   - Consider interaction terms: Volume * Quantity")

print("\n3. BIGGEST PREDICTION CHALLENGES:")
print("   - Very small parts (high surface/volume ratio)")  
print("   - Single-piece orders vs bulk orders")
print("   - Complex geometries with high waste ratios")
print("   - Different machine capabilities affecting pricing") 