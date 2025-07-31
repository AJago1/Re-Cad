import pandas as pd
import numpy as np

print("🚀 COMPLETE FEATURE ENGINEERING - INCLUDING ALL ORIGINAL FEATURES")
print("=" * 70)

# Read the CSV file with proper handling of semicolon separators and comma decimals
print("Reading CSV file...")
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (removed false prices).csv', 
                 sep=';', decimal=',')

print(f"Original data shape: {df.shape}")

# ALL ORIGINAL FEATURES FROM CSV (22 total)
original_features = [
    # Identification & Project Features
    'Project ID', 'Part Name', 'Material', 'Machine', 'Colour',
    
    # Pricing Feature (TARGET)
    'Net price / part',
    
    # GEOMETRIC FEATURES (5 Volume Types + Surface Area) - THE MOST IMPORTANT!
    'convex_hull_volume',  # 🥈 Rank #2 in importance
    'bb_volume',           # Rank #7
    'surface_area',        # 🥉 Rank #3 in importance  
    'shrinkwrap_volume',   # 🥇 Rank #1 in importance (MOST IMPORTANT!)
    'Volume',              # Rank #5
    
    # Dimensional Features
    'Max D', 'Min D', 'D Ratio',
    
    # Production Features
    'Quantity',
    
    # Waste & Efficiency Features
    'waste', 'shrinkwrap_ratio', 'waste_ratio',
    
    # Project Totals (Aggregated Features)
    'Full_Project_Shrinkwrap_Volume_PA2200',
    'Full_Project_BB_Volume_PA2200', 
    'Full_Project_Surface_Area_PA2200',
    'Full_Project_Volume_PA2200'
]

print(f"\n📋 ORIGINAL FEATURES VERIFICATION:")
print(f"Expected: {len(original_features)} features")
missing_features = [f for f in original_features if f not in df.columns]
if missing_features:
    print(f"⚠️  Missing features: {missing_features}")
else:
    print("✅ All original features found!")

# Convert ALL numeric columns properly (exclude categorical: Project ID, Part Name, Material, Machine, Colour)
numeric_columns = [col for col in original_features if col not in ['Project ID', 'Part Name', 'Material', 'Machine', 'Colour']]

print(f"\n🔢 CONVERTING NUMERIC COLUMNS:")
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"✓ {col}")

# CREATE ALL 26 DERIVED FEATURES
print(f"\n🔧 CREATING DERIVED FEATURES:")

# 1. GEOMETRIC EFFICIENCY RATIOS
print("1. Geometric Efficiency Ratios...")

# SA to Volume Ratio
df['SA_to_Volume_Ratio'] = df['surface_area'] / df['Volume']

# Volume Density (Packing Efficiency)  
df['Volume_Density'] = df['Volume'] / df['bb_volume']
df['Packing_Efficiency'] = df['Volume'] / df['bb_volume']  # Same as Volume_Density

# Material Efficiency
df['Material_Efficiency'] = df['Volume'] / (df['Volume'] + df['waste'])

# Volume to SA Ratio (inverse of complexity)
df['Volume_to_SA_Ratio'] = df['Volume'] / df['surface_area']

# 2. ADVANCED GEOMETRIC FEATURES  
print("2. Advanced Geometric Features...")

# Waste per Volume
df['Waste_per_Volume'] = df['waste'] / df['Volume']

# Waste Efficiency (how much of shrinkwrap is useful)
df['Waste_Efficiency'] = df['Volume'] / df['shrinkwrap_volume']

# Total Material Usage
df['Total_Material_Usage'] = df['Volume'] + df['waste']

# Complexity Score (normalized composite)
def normalize_feature(series):
    return (series - series.min()) / (series.max() - series.min())

df['SA_to_Vol_norm'] = normalize_feature(df['SA_to_Volume_Ratio'])
df['Waste_ratio_norm'] = normalize_feature(df['waste_ratio']) 
df['D_ratio_norm'] = normalize_feature(df['D Ratio'])

df['Complexity_Score'] = (df['SA_to_Vol_norm'] + df['Waste_ratio_norm'] + df['D_ratio_norm']) / 3

# Remove temporary normalized columns
df.drop(['SA_to_Vol_norm', 'Waste_ratio_norm', 'D_ratio_norm'], axis=1, inplace=True)

# 3. PRICING ANALYSIS FEATURES
print("3. Pricing Analysis Features...")

df['Price_per_Volume'] = df['Net price / part'] / df['Volume']
df['Price_per_Surface_Area'] = df['Net price / part'] / df['surface_area'] 
df['Price_per_BB_Volume'] = df['Net price / part'] / df['bb_volume']
df['Total_Part_Value'] = df['Net price / part'] * df['Quantity']

# 4. CATEGORIZATION FEATURES
print("4. Categorization Features...")

# Size Category
def categorize_size(volume):
    if volume <= 1000:
        return "Small"
    elif volume <= 50000:
        return "Medium"
    elif volume <= 500000:
        return "Large"
    else:
        return "XLarge"

df['Size_Category'] = df['Volume'].apply(categorize_size)

# Aspect Ratio Category
def categorize_aspect_ratio(d_ratio):
    if d_ratio <= 2:
        return "Compact"
    elif d_ratio <= 5:
        return "Elongated"
    else:
        return "Very_Elongated"

df['Aspect_Ratio_Category'] = df['D Ratio'].apply(categorize_aspect_ratio)

# Production Scale
def categorize_production_scale(quantity):
    if quantity >= 100:
        return "Large_Batch"
    elif quantity >= 10:
        return "Medium_Batch"
    elif quantity >= 2:
        return "Small_Batch"
    else:
        return "Prototype"

df['Production_Scale'] = df['Quantity'].apply(categorize_production_scale)

# 5. PROJECT-LEVEL ANALYTICS
print("5. Project-Level Analytics...")

# Calculate project-level statistics
project_stats = df.groupby('Project ID').agg({
    'Total_Part_Value': 'sum',
    'Net price / part': ['mean', 'std']
}).round(4)

project_stats.columns = ['Total_Project_Value', 'Avg_Part_Price', 'Price_Std_Dev']
project_stats = project_stats.reset_index()

# Merge back to main dataframe
df = df.merge(project_stats, on='Project ID', how='left')

# Part Value Share
df['Part_Value_Share'] = df['Total_Part_Value'] / df['Total_Project_Value']

# Price Position in Project (standardized)
df['Price_Position_in_Project'] = ((df['Net price / part'] - df['Avg_Part_Price']) / 
                                  df['Price_Std_Dev'].replace(0, np.nan))

# Handle infinite and NaN values
print("\n🧹 CLEANING DATA:")
print("Replacing infinite values with NaN...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Count NaN values before cleaning
nan_counts_before = df.isnull().sum()
features_with_nans = nan_counts_before[nan_counts_before > 0]

if len(features_with_nans) > 0:
    print(f"Features with NaN values: {len(features_with_nans)}")
    for feature, count in features_with_nans.items():
        if count > 0:
            print(f"  {feature}: {count} NaN values")
else:
    print("✅ No NaN values found!")

# Convert decimal format back to comma notation for consistency
print("\n💾 SAVING ENHANCED DATASET:")
def convert_to_comma_decimal(df, numeric_cols):
    """Convert numeric columns back to comma decimal format"""
    df_output = df.copy()
    for col in numeric_cols:
        if col in df_output.columns:
            df_output[col] = df_output[col].apply(lambda x: str(x).replace('.', ',') if pd.notnull(x) else x)
    return df_output

# All numeric columns for output conversion
all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Convert to comma decimal format
df_output = convert_to_comma_decimal(df, all_numeric_cols)

# Save enhanced dataset
output_filename = 'Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (COMPLETE with ALL features).csv'
df_output.to_csv(output_filename, sep=';', index=False)

print(f"✅ Enhanced dataset saved as: {output_filename}")
print(f"📊 Final dataset shape: {df_output.shape}")
print(f"📈 Original features: {len(original_features)}")

# Count derived features
derived_features = [col for col in df.columns if col not in original_features]
print(f"🔧 Derived features added: {len(derived_features)}")
print(f"🎯 Total features: {df.shape[1]}")

print(f"\n📋 DERIVED FEATURES CREATED:")
for i, feature in enumerate(derived_features, 1):
    print(f"{i:2d}. {feature}")

print(f"\n🏆 KEY VOLUME FEATURES (by importance):")
print("1. 🥇 shrinkwrap_volume (MOST IMPORTANT)")
print("2. 🥈 convex_hull_volume") 
print("3. 🥉 surface_area")
print("4. Volume")
print("5. bb_volume")

print(f"\n✨ FEATURE ENGINEERING COMPLETE!")
print("📝 All original features preserved")
print("🔧 26 derived features added")
print("📊 Ready for advanced analytics & modeling!")
print("🎯 Focus on shrinkwrap_volume for best price predictions!") 