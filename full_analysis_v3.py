import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load new data
df = pd.read_csv('correct totals 3.csv', sep=';')

# Convert price columns to numeric (handle European decimal format)
df['price'] = pd.to_numeric(df['Net price / part'].str.replace(',', '.'), errors='coerce')
df['total_price'] = pd.to_numeric(df['net price * Quantity'].str.replace(',', '.'), errors='coerce')

# Convert all numeric columns
numeric_cols = ['convex_hull_volume', 'bb_volume', 'surface_area', 'Quantity', 
                'Max D', 'Min D', 'D Ratio', 'waste', 'shrinkwrap_volume', 'Volume',
                'Volume * Quantity ', 'shrinkwrap_volume*Quantity', 'Surface area*Quantity',
                'BB_Volume*Quantity', 'Convex_Hull_Volume*Quantity', 'Total_Parts_in_Project',
                'Project_Total_convex_hull_volume', 'Project_Total_bb_volume', 
                'Project_Total_shrinkwrap_volume', 'Project_Total_Volume', 'Project_Total_surface_area']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print("=" * 70)
print("COMPREHENSIVE SLS 3D PRINTING PRICE ANALYSIS - VERSION 3")
print("=" * 70)

print(f"\n=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Unique projects: {df['Project ID'].nunique()}")
print(f"Price range: €{df['price'].min():.2f} to €{df['price'].max():.2f}")
print(f"Median price per part: €{df['price'].median():.2f}")
print(f"Total order values: €{df['total_price'].min():.2f} to €{df['total_price'].max():.2f}")
print(f"Median total order: €{df['total_price'].median():.2f}")

print(f"\n=== NEW FEATURE: TOTAL ORDER VALUE ===")
# Analyze the new total price feature
df['avg_part_price_check'] = df['total_price'] / df['Quantity']
price_diff = abs(df['price'] - df['avg_part_price_check']).max()
print(f"Price consistency check (should be ~0): {price_diff:.6f}")

# Order value analysis
df['order_size_category'] = pd.cut(df['total_price'], 
                                   bins=[0, 10, 50, 200, 1000, np.inf],
                                   labels=['Micro (€0-10)', 'Small (€10-50)', 'Medium (€50-200)', 
                                          'Large (€200-1K)', 'Huge (€1K+)'])

order_analysis = df.groupby('order_size_category')['price'].agg(['count', 'mean', 'median']).round(2)
print(f"\nPricing by total order value:")
print(order_analysis)

print(f"\n=== BULK PRICING ANALYSIS ===")
# Detailed bulk analysis
df['qty_bin'] = pd.cut(df['Quantity'], bins=[0, 1, 5, 10, 25, 50, 100, 500, np.inf], 
                       labels=['1', '2-5', '6-10', '11-25', '26-50', '51-100', '101-500', '500+'])

bulk_stats = df.groupby('qty_bin').agg({
    'price': ['count', 'mean', 'median', 'std'],
    'total_price': ['mean', 'median']
}).round(2)
bulk_stats.columns = ['count', 'avg_price', 'med_price', 'std_price', 'avg_total', 'med_total']
print(bulk_stats)

# Calculate bulk discount rates
single_price = df[df['Quantity'] == 1]['price'].median()
print(f"\nBulk discount analysis (vs single piece median €{single_price:.2f}):")
for qty_range in ['2-5', '6-10', '26-50', '101-500', '500+']:
    if qty_range in bulk_stats.index:
        current_price = bulk_stats.loc[qty_range, 'med_price']
        discount = ((single_price - current_price) / single_price) * 100
        print(f"  {qty_range} pieces: €{current_price:.2f} ({discount:+.0f}% vs single)")

print(f"\n=== PROJECT SIZE EFFECT ANALYSIS ===")
# Project size effect
df['project_size_bin'] = pd.cut(df['Total_Parts_in_Project'], 
                               bins=[0, 5, 15, 50, 150, 500, np.inf],
                               labels=['1-5', '6-15', '16-50', '51-150', '151-500', '500+'])

project_stats = df.groupby('project_size_bin').agg({
    'price': ['count', 'mean', 'median'],
    'total_price': 'mean'
}).round(2)
project_stats.columns = ['count', 'avg_price', 'med_price', 'avg_total_order']
print(project_stats)

print(f"\n=== MACHINE ANALYSIS ===")
machine_stats = df.groupby('Machine').agg({
    'price': ['count', 'mean', 'median'],
    'Volume': 'mean',
    'total_price': 'mean'
}).round(2)
machine_stats.columns = ['count', 'avg_price', 'med_price', 'avg_volume', 'avg_total']
print(machine_stats)

print(f"\n=== FEATURE CORRELATION ANALYSIS ===")
# All available features for correlation
base_features = ['Volume', 'surface_area', 'convex_hull_volume', 'bb_volume', 'waste', 
                'D Ratio', 'Quantity', 'Total_Parts_in_Project']

interaction_features = ['Volume * Quantity ', 'Surface area*Quantity', 'BB_Volume*Quantity', 
                       'Convex_Hull_Volume*Quantity', 'shrinkwrap_volume*Quantity']

price_features = ['total_price']

all_features = base_features + interaction_features + price_features

# Calculate correlations
feature_correlations = df[all_features + ['price']].corr()['price'].abs().sort_values(ascending=False)
print("Top 15 features by correlation with price per part:")
print(feature_correlations.head(15))

print(f"\n=== DERIVED FEATURES ANALYSIS ===")
# Create comprehensive derived features
df['volume_efficiency'] = df['Volume'] / df['bb_volume']
df['surface_to_volume'] = df['surface_area'] / df['Volume']
df['waste_ratio'] = df['waste'] / df['Volume']
df['price_per_mm3'] = df['price'] / df['Volume']
df['volume_density'] = df['Volume'] / df['convex_hull_volume']
df['project_volume_share'] = df['Volume'] / df['Project_Total_Volume']

# New derived features based on total price
df['total_value_per_mm3'] = df['total_price'] / df['Volume * Quantity ']
df['bulk_efficiency'] = df['Volume * Quantity '] / df['total_price']  # mm3 per euro
df['order_complexity'] = df['Surface area*Quantity'] / df['Volume * Quantity ']  # surface/volume for whole order
df['machine_utilization'] = df['Volume * Quantity '] / df['BB_Volume*Quantity']  # space efficiency
df['project_dominance'] = df['total_price'] / (df['Project_Total_Volume'] * df['price_per_mm3'].median())  # how much of project value

# Logarithmic features
df['log_volume'] = np.log1p(df['Volume'])
df['log_total_volume'] = np.log1p(df['Volume * Quantity '])
df['log_quantity'] = np.log1p(df['Quantity'])
df['log_total_price'] = np.log1p(df['total_price'])

derived_features = ['volume_efficiency', 'surface_to_volume', 'waste_ratio', 'volume_density',
                   'project_volume_share', 'total_value_per_mm3', 'bulk_efficiency', 
                   'order_complexity', 'machine_utilization', 'project_dominance',
                   'log_volume', 'log_total_volume', 'log_quantity', 'log_total_price']

print("Derived feature correlations with price per part:")
derived_corr = df[derived_features + ['price']].corr()['price'].abs().sort_values(ascending=False)
for feat, corr in derived_corr.items():
    if feat != 'price':
        print(f"  {feat}: {corr:.4f}")

print(f"\n=== PREDICTION PROBLEM AREAS ===")

# 1. Tiny vs Large parts
df['size_category'] = pd.cut(df['Volume'], bins=[0, 1000, 10000, 100000, np.inf],
                            labels=['Tiny (<1K mm³)', 'Small (1-10K)', 'Medium (10-100K)', 'Large (100K+)'])

size_analysis = df.groupby('size_category').agg({
    'price': ['count', 'mean', 'std'],
    'price_per_mm3': ['mean', 'std']
}).round(4)
print("Pricing by part size (where model will struggle):")
print(size_analysis)

# 2. Single vs bulk complexity
single_parts = df[df['Quantity'] == 1]
bulk_parts = df[df['Quantity'] >= 50]

print(f"\n=== SINGLE vs BULK PREDICTION CHALLENGES ===")
print(f"Single parts (n={len(single_parts)}):")
print(f"  Price range: €{single_parts['price'].min():.2f} - €{single_parts['price'].max():.2f}")
print(f"  Price/mm³ range: {single_parts['price_per_mm3'].min():.6f} - {single_parts['price_per_mm3'].max():.6f}")
print(f"  Best predictor: {single_parts[all_features + ['price']].corr()['price'].abs().sort_values(ascending=False).index[1]} (r={single_parts[all_features + ['price']].corr()['price'].abs().sort_values(ascending=False).iloc[1]:.3f})")

print(f"\nBulk parts (n={len(bulk_parts)}):")
print(f"  Price range: €{bulk_parts['price'].min():.2f} - €{bulk_parts['price'].max():.2f}")
print(f"  Price/mm³ range: {bulk_parts['price_per_mm3'].min():.6f} - {bulk_parts['price_per_mm3'].max():.6f}")
if len(bulk_parts) > 10:
    print(f"  Best predictor: {bulk_parts[all_features + ['price']].corr()['price'].abs().sort_values(ascending=False).index[1]} (r={bulk_parts[all_features + ['price']].corr()['price'].abs().sort_values(ascending=False).iloc[1]:.3f})")

# 3. Outlier analysis
df['price_zscore'] = np.abs(stats.zscore(df['price_per_mm3'].fillna(0)))
outliers = df[df['price_zscore'] > 2.5]
print(f"\n=== EXTREME OUTLIERS (Z-score > 2.5) ===")
print(f"Found {len(outliers)} outlier parts:")
if len(outliers) > 0:
    outlier_sample = outliers[['Project ID', 'Part Name', 'price', 'Volume', 'Quantity', 'price_per_mm3']].head(5)
    print(outlier_sample.to_string())

print(f"\n=== MULTICOLLINEARITY ISSUES ===")
# Check for highly correlated features
all_model_features = base_features + interaction_features + derived_features
feature_matrix = df[all_model_features].dropna()
corr_matrix = feature_matrix.corr()

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("Highly correlated feature pairs (|r| > 0.85) - REMOVE ONE FROM EACH PAIR:")
for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")

print(f"\n=== FINAL RECOMMENDATIONS ===")
print("🎯 BEST FEATURES FOR LINEAR MODEL:")
top_features = feature_correlations.head(8)
for i, (feat, corr) in enumerate(top_features.items(), 1):
    if feat != 'price':
        print(f"  {i}. {feat}: {corr:.4f}")

print(f"\n🚨 CRITICAL ISSUES TO ADDRESS:")
print("1. BULK DISCOUNTS: 50-90% price drops with quantity - try separate models by quantity bins")
print("2. PROJECT SIZE EFFECT: Large projects get additional discounts")  
print("3. MACHINE DIFFERENCES: P396 vs Formiga have different cost structures")
print("4. TINY PARTS: High surface/volume ratio causes extreme price/mm³ variations")
print("5. MULTICOLLINEARITY: Many geometry features are 85%+ correlated")

print(f"\n💡 SUGGESTED NEW DERIVED FEATURES:")
print("1. quantity_tier = ['single', 'small_batch', 'medium_batch', 'bulk'] (categorical)")
print("2. machine_volume_interaction = Volume * (Machine=='P396')")
print("3. complexity_penalty = surface_area / (Volume^0.67)  # Surface scales differently than volume")
print("4. bulk_discount_factor = 1 / (1 + log(Quantity))")
print("5. project_economy_of_scale = Total_Parts_in_Project / Volume")
print("6. material_efficiency = Volume / (convex_hull_volume + waste)")

print(f"\n🔧 MODEL STRATEGY RECOMMENDATIONS:")
print("1. ENSEMBLE APPROACH: Separate models for quantity bins (1, 2-10, 11-50, 50+)")
print("2. TWO-STAGE MODEL: First predict if bulk discount applies, then predict price")
print("3. REGULARIZATION: Use Ridge/Lasso regression due to multicollinearity")
print("4. FEATURE SELECTION: Remove highly correlated features (keep best predictor from each pair)")

# Summary statistics
print(f"\n📊 FINAL STATISTICS:")
print(f"  Best single predictor: {feature_correlations.index[1]} (r={feature_correlations.iloc[1]:.3f})")
print(f"  Worst bulk discount: {bulk_stats['med_price'].min():.2f}€ vs {single_price:.2f}€ single part")
print(f"  Price prediction range: {df['price'].std() / df['price'].mean() * 100:.0f}% coefficient of variation")
print(f"  Critical features (r>0.5): {len(feature_correlations[feature_correlations > 0.5]) - 1}") 