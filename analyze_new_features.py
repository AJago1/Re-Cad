import pandas as pd
import numpy as np
from scipy import stats

# Load new data
df = pd.read_csv('correct totals 2.csv', sep=';')

# Convert price to numeric (handle European decimal format)
df['price'] = pd.to_numeric(df['Net price / part'].str.replace(',', '.'), errors='coerce')

# Convert other numeric columns including the new ones
new_features = ['Volume * Quantity', 'shrinkwrap_volume*Quantity', 'Surface area*Quantity', 
                'BB_Volume*Quantity', 'Convex_Hull_Volume*Quantity']

basic_features = ['Volume', 'surface_area', 'convex_hull_volume', 'bb_volume', 'Quantity', 
                  'Total_Parts_in_Project', 'shrinkwrap_volume']

all_features = basic_features + new_features

for col in all_features:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print("=== NEW INTERACTION FEATURES ANALYSIS ===")
print("\nNew features added:")
for feature in new_features:
    print(f"- {feature}")

print("\n=== CORRELATION WITH PRICE ===")
# Calculate correlations with price for all features
feature_correlations = df[all_features + ['price']].corr()['price'].abs().sort_values(ascending=False)
print("Feature correlations with price (absolute values):")
print(feature_correlations.head(15))

print("\n=== COMPARING BASE vs INTERACTION FEATURES ===")
# Compare individual features vs their quantity interactions
comparisons = [
    ('Volume', 'Volume * Quantity'),
    ('surface_area', 'Surface area*Quantity'), 
    ('bb_volume', 'BB_Volume*Quantity'),
    ('convex_hull_volume', 'Convex_Hull_Volume*Quantity'),
    ('shrinkwrap_volume', 'shrinkwrap_volume*Quantity')
]

for base, interaction in comparisons:
    base_corr = abs(df[base].corr(df['price']))
    interaction_corr = abs(df[interaction].corr(df['price']))
    improvement = ((interaction_corr - base_corr) / base_corr) * 100
    print(f"\n{base}:")
    print(f"  Base correlation: {base_corr:.4f}")
    print(f"  Interaction correlation: {interaction_corr:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")

print("\n=== BULK QUANTITY PATTERNS ===")
# Analyze how Volume*Quantity performs across different quantity bins
df['qty_bin'] = pd.cut(df['Quantity'], bins=[0, 1, 5, 10, 25, 50, 100, 500, np.inf], 
                       labels=['1', '2-5', '6-10', '11-25', '26-50', '51-100', '101-500', '500+'])

# Look at price per volume*quantity ratio across bins
df['price_per_vol_qty'] = df['price'] / df['Volume * Quantity']

print("\nPrice per (Volume*Quantity) by quantity bin:")
bulk_analysis = df.groupby('qty_bin')['price_per_vol_qty'].agg(['count', 'mean', 'median']).round(6)
print(bulk_analysis)

print("\n=== MULTICOLLINEARITY CHECK ===")
# Check correlations between new features and existing ones
corr_matrix = df[all_features].corr()
high_corr_new = []

for new_feat in new_features:
    for base_feat in basic_features:
        if abs(corr_matrix.loc[new_feat, base_feat]) > 0.8:
            high_corr_new.append((new_feat, base_feat, corr_matrix.loc[new_feat, base_feat]))

print("High correlations between new interaction features and base features:")
for feat1, feat2, corr in high_corr_new:
    print(f"{feat1} <-> {feat2}: {corr:.3f}")

print("\n=== DERIVED VOLUME EFFICIENCY FEATURES ===")
# Test some additional derived features
df['volume_efficiency'] = df['Volume'] / df['bb_volume']
df['surface_to_volume'] = df['surface_area'] / df['Volume']
df['complexity_score'] = df['Surface area*Quantity'] / df['Volume * Quantity']  # Surface/volume but accounting for quantity
df['material_utilization'] = df['Volume * Quantity'] / df['BB_Volume*Quantity']

efficiency_features = ['volume_efficiency', 'surface_to_volume', 'complexity_score', 'material_utilization']

print("\nEfficiency feature correlations with price:")
for feat in efficiency_features:
    corr = abs(df[feat].corr(df['price']))
    print(f"{feat}: {corr:.4f}")

print("\n=== KEY INSIGHTS ===")
volume_qty_corr = abs(df['Volume * Quantity'].corr(df['price']))
volume_corr = abs(df['Volume'].corr(df['price']))
print(f"1. Volume*Quantity correlation ({volume_qty_corr:.4f}) vs Volume alone ({volume_corr:.4f})")

surface_qty_corr = abs(df['Surface area*Quantity'].corr(df['price']))
surface_corr = abs(df['surface_area'].corr(df['price']))
print(f"2. Surface*Quantity correlation ({surface_qty_corr:.4f}) vs Surface alone ({surface_corr:.4f})")

# Check if interaction features help with bulk vs single item prediction
single_items = df[df['Quantity'] == 1]
bulk_items = df[df['Quantity'] > 50]

print(f"\n3. BULK vs SINGLE PREDICTION:")
print(f"   Single items (n={len(single_items)}): Volume*Qty correlation = {abs(single_items['Volume * Quantity'].corr(single_items['price'])):.4f}")
print(f"   Bulk items (n={len(bulk_items)}): Volume*Qty correlation = {abs(bulk_items['Volume * Quantity'].corr(bulk_items['price'])):.4f}")

print("\n=== RECOMMENDATIONS ===")
print("1. BEST NEW FEATURES:")
top_new_features = feature_correlations[feature_correlations.index.isin(new_features)].head(3)
for feat, corr in top_new_features.items():
    print(f"   - {feat}: {corr:.4f}")

print("\n2. FEATURE ENGINEERING SUCCESS:")
if volume_qty_corr > volume_corr:
    print(f"   ✓ Volume*Quantity improves prediction by {((volume_qty_corr/volume_corr-1)*100):+.1f}%")
else:
    print(f"   ✗ Volume*Quantity doesn't improve over Volume alone")

print("\n3. NEXT STEPS:")
print("   - Try log(Volume*Quantity) for non-linear scaling effects")  
print("   - Add Machine*Volume interaction (different machines price differently)")
print("   - Try Surface*Quantity/Volume*Quantity for complexity-adjusted bulk pricing") 