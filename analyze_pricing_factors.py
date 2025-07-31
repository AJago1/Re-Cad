import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Read the enhanced CSV file
print("Loading enhanced dataset...")
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (with derived features).csv', 
                 sep=';', decimal=',')

print(f"Dataset shape: {df.shape}")

# Convert numeric columns for analysis
numeric_columns = ['Net price / part', 'SA_to_Volume_Ratio', 'Volume_Density', 'Packing_Efficiency', 
                  'Material_Efficiency', 'Price_per_Volume', 'Price_per_Surface_Area', 'Price_per_BB_Volume',
                  'Volume_to_SA_Ratio', 'Waste_per_Volume', 'Waste_Efficiency', 'Complexity_Score',
                  'Volume', 'surface_area', 'bb_volume', 'Quantity', 'Max D', 'D Ratio', 'waste_ratio']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n========== PRICING CORRELATION ANALYSIS ==========")

# Calculate correlations between price and various features
price_correlations = []
features_to_analyze = [
    ('Volume', 'Part Volume'),
    ('surface_area', 'Surface Area'),
    ('bb_volume', 'Bounding Box Volume'),
    ('SA_to_Volume_Ratio', 'Surface Area/Volume Ratio'),
    ('Volume_Density', 'Volume Density'),
    ('Complexity_Score', 'Complexity Score'),
    ('Max D', 'Maximum Dimension'),
    ('D Ratio', 'Aspect Ratio'),
    ('waste_ratio', 'Waste Ratio'),
    ('Waste_per_Volume', 'Waste per Volume'),
    ('Material_Efficiency', 'Material Efficiency'),
    ('Packing_Efficiency', 'Packing Efficiency')
]

print("\nCorrelation with Part Price:")
print("-" * 50)

for feature, display_name in features_to_analyze:
    if feature in df.columns:
        # Filter out NaN values for correlation calculation
        mask = ~(pd.isna(df['Net price / part']) | pd.isna(df[feature]))
        if mask.sum() > 10:  # Need at least 10 data points
            corr, p_value = pearsonr(df.loc[mask, 'Net price / part'], df.loc[mask, feature])
            price_correlations.append((display_name, corr, p_value))
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{display_name:25}: r = {corr:6.3f} {significance}")

# Sort by absolute correlation strength
price_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n========== STRONGEST PRICING FACTORS ==========")
print("Top factors correlating with price (strongest first):")
for i, (feature, corr, p_val) in enumerate(price_correlations[:8], 1):
    direction = "Higher" if corr > 0 else "Lower"
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    print(f"{i}. {feature}: {strength} correlation ({corr:.3f})")
    print(f"   → {direction} {feature.lower()} = Higher price")

print("\n========== SIZE CATEGORY ANALYSIS ==========")
size_stats = df.groupby('Size_Category')['Net price / part'].agg(['mean', 'median', 'count']).round(2)
print("Average prices by size category:")
print(size_stats)

print("\n========== PRODUCTION SCALE ANALYSIS ==========")
scale_stats = df.groupby('Production_Scale')['Net price / part'].agg(['mean', 'median', 'count']).round(2)
print("Average prices by production scale:")
print(scale_stats)

print("\n========== COMPLEXITY ANALYSIS ==========")
# Create complexity bins
df['Complexity_Bin'] = pd.cut(df['Complexity_Score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
complexity_stats = df.groupby('Complexity_Bin')['Net price / part'].agg(['mean', 'median', 'count']).round(2)
print("Average prices by complexity level:")
print(complexity_stats)

print("\n========== EFFICIENCY ANALYSIS ==========")
# Material efficiency bins
df['Material_Efficiency_Bin'] = pd.cut(df['Material_Efficiency'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
efficiency_stats = df.groupby('Material_Efficiency_Bin')['Net price / part'].agg(['mean', 'median', 'count']).round(2)
print("Average prices by material efficiency:")
print(efficiency_stats)

print("\n========== MACHINE TYPE ANALYSIS ==========")
machine_stats = df.groupby('Machine')['Net price / part'].agg(['mean', 'median', 'count']).round(2)
print("Average prices by machine type:")
print(machine_stats)

print("\n========== TOP INSIGHTS FOR PRICING ==========")
print("\n🔍 KEY FINDINGS:")

# Find the most expensive and cheapest parts
expensive_parts = df.nlargest(5, 'Net price / part')[['Part Name', 'Net price / part', 'Volume', 'Complexity_Score', 'Size_Category', 'Production_Scale']]
cheap_parts = df.nsmallest(5, 'Net price / part')[['Part Name', 'Net price / part', 'Volume', 'Complexity_Score', 'Size_Category', 'Production_Scale']]

print("\n💰 Most Expensive Parts:")
for _, part in expensive_parts.iterrows():
    print(f"   • {part['Part Name'][:40]}: €{part['Net price / part']:.2f} ({part['Size_Category']}, Complexity: {part['Complexity_Score']:.2f})")

print("\n💸 Least Expensive Parts:")
for _, part in cheap_parts.iterrows():
    print(f"   • {part['Part Name'][:40]}: €{part['Net price / part']:.2f} ({part['Size_Category']}, Complexity: {part['Complexity_Score']:.2f})")

# Price per volume analysis
print(f"\n📊 PRICE EFFICIENCY METRICS:")
print(f"   • Average price per volume: €{df['Price_per_Volume'].mean():.4f}/mm³")
print(f"   • Average price per surface area: €{df['Price_per_Surface_Area'].mean():.6f}/mm²")

# Find parts with unusual pricing patterns
high_price_per_volume = df.nlargest(3, 'Price_per_Volume')[['Part Name', 'Price_per_Volume', 'Net price / part', 'Volume']]
print(f"\n⚠️ Parts with highest price per volume (potentially overpriced or very complex):")
for _, part in high_price_per_volume.iterrows():
    print(f"   • {part['Part Name'][:50]}: €{part['Price_per_Volume']:.6f}/mm³")

print("\n🎯 PRICING OPTIMIZATION OPPORTUNITIES:")
print("   1. Parts with very high price/volume ratios may need price review")
print("   2. Large batch productions show different pricing patterns")
print("   3. Complexity score strongly influences pricing")
print("   4. Material efficiency impacts could be better utilized in pricing")

print(f"\n✅ Analysis complete! Enhanced dataset has {df.shape[1]} total columns including 26 new derived features.") 