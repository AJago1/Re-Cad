import pandas as pd
import numpy as np

# Read the CSV file with proper handling of semicolon separators and comma decimals
print("Reading CSV file...")
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (removed false prices).csv', 
                 sep=';', decimal=',')

print(f"Original data shape: {df.shape}")

# Convert numeric columns properly
numeric_columns = ['Net price / part', 'convex_hull_volume', 'bb_volume', 'surface_area', 
                  'Quantity', 'Max D', 'Min D', 'D Ratio', 'waste', 'shrinkwrap_volume', 
                  'shrinkwrap_ratio', 'waste_ratio', 'Volume', 'Full_Project_Shrinkwrap_Volume_PA2200',
                  'Full_Project_BB_Volume_PA2200', 'Full_Project_Surface_Area_PA2200', 'Full_Project_Volume_PA2200']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Adding derived features...")

# ========== EFFICIENCY & DENSITY FEATURES ==========
# 1. Surface Area to Volume Ratio - higher ratio = more complex/detailed parts
df['SA_to_Volume_Ratio'] = df['surface_area'] / df['Volume'].replace(0, np.nan)

# 2. Volume Density (Volume/Bounding Box Volume) - how solid the part is
df['Volume_Density'] = df['Volume'] / df['bb_volume'].replace(0, np.nan)

# 3. Packing Efficiency (Shrinkwrap Volume / Bounding Box Volume)
df['Packing_Efficiency'] = df['shrinkwrap_volume'] / df['bb_volume'].replace(0, np.nan)

# 4. Material Efficiency (Volume / Shrinkwrap Volume) - how much of the print envelope is actual part
df['Material_Efficiency'] = df['Volume'] / df['shrinkwrap_volume'].replace(0, np.nan)

# ========== PRICING FEATURES ==========
# 5. Price per unit Volume
df['Price_per_Volume'] = df['Net price / part'] / df['Volume'].replace(0, np.nan)

# 6. Price per unit Surface Area
df['Price_per_Surface_Area'] = df['Net price / part'] / df['surface_area'].replace(0, np.nan)

# 7. Price per unit Bounding Box Volume (size-based pricing indicator)
df['Price_per_BB_Volume'] = df['Net price / part'] / df['bb_volume'].replace(0, np.nan)

# 8. Total Part Value (Price × Quantity)
df['Total_Part_Value'] = df['Net price / part'] * df['Quantity']

# ========== SIZE & COMPLEXITY FEATURES ==========
# 9. Volume to Surface Area Ratio (inverse of SA/Volume - bigger values = chunkier parts)
df['Volume_to_SA_Ratio'] = df['Volume'] / df['surface_area'].replace(0, np.nan)

# 10. Size Category based on Max Dimension
def categorize_size(max_d):
    if pd.isna(max_d):
        return 'Unknown'
    elif max_d < 30:
        return 'Small'
    elif max_d < 100:
        return 'Medium'
    elif max_d < 200:
        return 'Large'
    else:
        return 'XLarge'

df['Size_Category'] = df['Max D'].apply(categorize_size)

# 11. Aspect Ratio Category
def categorize_aspect_ratio(d_ratio):
    if pd.isna(d_ratio):
        return 'Unknown'
    elif d_ratio < 2:
        return 'Compact'
    elif d_ratio < 5:
        return 'Elongated'
    else:
        return 'Very_Elongated'

df['Aspect_Ratio_Category'] = df['D Ratio'].apply(categorize_aspect_ratio)

# ========== WASTE & EFFICIENCY ANALYSIS ==========
# 12. Waste per unit Volume
df['Waste_per_Volume'] = df['waste'] / df['Volume'].replace(0, np.nan)

# 13. Total Material Usage (Volume + Waste)
df['Total_Material_Usage'] = df['Volume'] + df['waste']

# 14. Waste Efficiency (lower is better)
df['Waste_Efficiency'] = df['Volume'] / df['Total_Material_Usage'].replace(0, np.nan)

# ========== PRODUCTION SCALE FEATURES ==========
# 15. Production Scale Category
def categorize_production_scale(quantity):
    if pd.isna(quantity):
        return 'Unknown'
    elif quantity == 1:
        return 'Prototype'
    elif quantity <= 10:
        return 'Small_Batch'
    elif quantity <= 50:
        return 'Medium_Batch'
    else:
        return 'Large_Batch'

df['Production_Scale'] = df['Quantity'].apply(categorize_production_scale)

# ========== COMPLEXITY INDICATORS ==========
# 16. Complexity Score (combination of multiple factors)
# Normalize key metrics to 0-1 scale first
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

df['Normalized_SA_Volume_Ratio'] = normalize_column(df['SA_to_Volume_Ratio'].fillna(0))
df['Normalized_Waste_Ratio'] = normalize_column(df['waste_ratio'].fillna(0))
df['Normalized_D_Ratio'] = normalize_column(df['D Ratio'].fillna(0))

# Complexity score (higher = more complex)
df['Complexity_Score'] = (df['Normalized_SA_Volume_Ratio'] * 0.4 + 
                         df['Normalized_Waste_Ratio'] * 0.3 + 
                         df['Normalized_D_Ratio'] * 0.3)

# Drop temporary normalized columns
df.drop(['Normalized_SA_Volume_Ratio', 'Normalized_Waste_Ratio', 'Normalized_D_Ratio'], axis=1, inplace=True)

# ========== PROJECT-LEVEL ANALYSIS ==========
# 17. Project Value Analysis
project_stats = df.groupby('Project ID').agg({
    'Total_Part_Value': 'sum',
    'Net price / part': ['mean', 'std', 'min', 'max'],
    'Volume': 'sum',
    'Quantity': 'sum',
    'Part Name': 'count'
}).round(6)

# Flatten column names
project_stats.columns = ['Total_Project_Value', 'Avg_Part_Price', 'Price_Std_Dev', 'Min_Part_Price', 
                        'Max_Part_Price', 'Total_Project_Volume', 'Total_Project_Quantity', 'Part_Count']

# Merge back to main dataframe
df = df.merge(project_stats, on='Project ID', how='left')

# 18. Part's share of project value
df['Part_Value_Share'] = df['Total_Part_Value'] / df['Total_Project_Value'].replace(0, np.nan)

# 19. Price position within project (is this part expensive/cheap relative to others in same project?)
df['Price_Position_in_Project'] = (df['Net price / part'] - df['Avg_Part_Price']) / df['Price_Std_Dev'].replace(0, np.nan)

# ========== CONVERT BACK TO COMMA DECIMAL FORMAT ==========
print("Converting numeric columns back to comma decimal format...")

# List of all numeric columns including the new derived features
all_numeric_columns = numeric_columns + [
    'SA_to_Volume_Ratio', 'Volume_Density', 'Packing_Efficiency', 'Material_Efficiency',
    'Price_per_Volume', 'Price_per_Surface_Area', 'Price_per_BB_Volume', 'Total_Part_Value',
    'Volume_to_SA_Ratio', 'Waste_per_Volume', 'Total_Material_Usage', 'Waste_Efficiency',
    'Complexity_Score', 'Total_Project_Value', 'Avg_Part_Price', 'Price_Std_Dev', 
    'Min_Part_Price', 'Max_Part_Price', 'Total_Project_Volume', 'Total_Project_Quantity',
    'Part_Count', 'Part_Value_Share', 'Price_Position_in_Project'
]

# Convert to comma decimal format
for col in all_numeric_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.6f}".replace('.', ',') if pd.notna(x) else x)

# Save the enhanced dataset
output_filename = 'Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (with derived features).csv'
df.to_csv(output_filename, sep=';', index=False, decimal=',')

print(f"\nEnhanced dataset saved as: {output_filename}")
print(f"New data shape: {df.shape}")

# Print summary of new features
print("\n========== NEW DERIVED FEATURES ADDED ==========")
print("EFFICIENCY & DENSITY:")
print("  - SA_to_Volume_Ratio: Surface area complexity indicator")
print("  - Volume_Density: How solid/hollow the part is")
print("  - Packing_Efficiency: Print space utilization")
print("  - Material_Efficiency: Actual part vs print envelope")

print("\nPRICING ANALYSIS:")
print("  - Price_per_Volume: Cost efficiency per unit volume")
print("  - Price_per_Surface_Area: Cost per unit surface area")
print("  - Price_per_BB_Volume: Size-based pricing indicator")
print("  - Total_Part_Value: Price × Quantity")

print("\nSIZE & COMPLEXITY:")
print("  - Volume_to_SA_Ratio: Chunkiness indicator")
print("  - Size_Category: Small/Medium/Large/XLarge")
print("  - Aspect_Ratio_Category: Compact/Elongated/Very_Elongated")
print("  - Complexity_Score: Combined complexity indicator (0-1)")

print("\nWASTE ANALYSIS:")
print("  - Waste_per_Volume: Waste efficiency indicator")
print("  - Total_Material_Usage: Volume + Waste")
print("  - Waste_Efficiency: Material utilization")

print("\nPRODUCTION SCALE:")
print("  - Production_Scale: Prototype/Small/Medium/Large_Batch")

print("\nPROJECT-LEVEL INSIGHTS:")
print("  - Total_Project_Value: Sum of all part values in project")
print("  - Avg_Part_Price: Average part price in project")
print("  - Price_Std_Dev: Price variability within project")
print("  - Part_Value_Share: This part's % of total project value")
print("  - Price_Position_in_Project: Price relative to project average")

print(f"\nTotal features added: {df.shape[1] - 22} new columns")
print("Ready for advanced pricing analysis!") 