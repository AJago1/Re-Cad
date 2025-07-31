import pandas as pd
import re
import numpy as np
from datetime import datetime

# Load data
print("Loading CSV files...")
df1 = pd.read_csv('Urejeni Podatki 2.10.csv', sep=';')
df2 = pd.read_csv('Convex hull + shrinkwraped 4mm all files. (urejen - brez cene).csv', sep=';')

print(f"Price file: {df1.shape[0]} rows")
print(f"Shrinkwrap file: {df2.shape[0]} rows")

# Extract project numbers
df1['project_num'] = df1['Project ID'].apply(lambda x: int(re.findall(r'\d+', str(x))[-1]) if pd.notna(x) and re.findall(r'\d+', str(x)) else None)
df2['project_num'] = df2['Project ID'].apply(lambda x: int(re.findall(r'\d+', str(x))[-1]) if pd.notna(x) and re.findall(r'\d+', str(x)) else None)

# Convert volumes to numeric
df1['volume_numeric'] = df1['Material Volume'].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else np.nan)
df2['volume_numeric'] = df2['volume'].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else np.nan)

# Find common projects
common_projects = set(df1['project_num'].dropna()) & set(df2['project_num'].dropna())
print(f"Found {len(common_projects)} common projects")

# Match parts
matched_results = []
unmatched_results = []

for project in sorted(common_projects):
    proj_parts = df1[df1['project_num'] == project].copy()
    shrink_parts = df2[df2['project_num'] == project].copy()
    
    print(f"Processing project {project}: {len(proj_parts)} price parts, {len(shrink_parts)} shrinkwrap parts")
    
    for idx, price_part in proj_parts.iterrows():
        price_volume = price_part['volume_numeric']
        
        if pd.isna(price_volume) or len(shrink_parts) == 0:
            # No volume or no shrinkwrap parts to match
            unmatched_results.append({
                'price_part_id': price_part['Part ID'],
                'project': project,
                'price_volume': price_volume,
                'reason': 'No volume data or no shrinkwrap parts'
            })
            continue
        
        # Find closest volume match
        volume_diffs = abs(shrink_parts['volume_numeric'] - price_volume)
        min_diff_idx = volume_diffs.idxmin()
        best_match = shrink_parts.loc[min_diff_idx]
        volume_diff = volume_diffs.loc[min_diff_idx]
        
        # Create matched result
        result = {}
        
        # Add price data with prefix
        for col in df1.columns:
            if col not in ['project_num', 'volume_numeric']:
                result[f'price_{col}'] = price_part[col]
        
        # Add shrinkwrap data with prefix
        for col in df2.columns:
            if col not in ['project_num', 'volume_numeric']:
                result[f'shrink_{col}'] = best_match[col]
        
        # Add matching metrics
        result['project_num'] = project
        result['price_volume'] = price_volume
        result['shrink_volume'] = best_match['volume_numeric']
        result['volume_difference'] = volume_diff
        result['volume_ratio'] = price_volume / best_match['volume_numeric'] if best_match['volume_numeric'] != 0 else np.inf
        
        matched_results.append(result)
        
        # Remove matched shrinkwrap part to avoid double-matching
        shrink_parts = shrink_parts.drop(min_diff_idx)

# Create matched DataFrame
matched_df = pd.DataFrame(matched_results)
print(f"\nMatched {len(matched_df)} parts")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'matched_data_{timestamp}.csv'

if len(matched_df) > 0:
    matched_df.to_csv(output_file, index=False, sep=';')
    print(f"Results saved to: {output_file}")
    
    # Show summary statistics
    print("\n" + "="*60)
    print("MATCHING SUMMARY")
    print("="*60)
    print(f"Total matched parts: {len(matched_df)}")
    print(f"Average volume difference: {matched_df['volume_difference'].mean():.2f}")
    print(f"Median volume difference: {matched_df['volume_difference'].median():.2f}")
    print(f"Max volume difference: {matched_df['volume_difference'].max():.2f}")
    
    # Show volume ratio statistics
    finite_ratios = matched_df['volume_ratio'][matched_df['volume_ratio'] != np.inf]
    if len(finite_ratios) > 0:
        print(f"Average volume ratio: {finite_ratios.mean():.3f}")
        print(f"Volume ratios close to 1.0 (±10%): {len(finite_ratios[(finite_ratios >= 0.9) & (finite_ratios <= 1.1)])}")
    
    # Show sample matches
    print("\nSample matches (first 5):")
    sample_cols = ['price_Part ID', 'shrink_name', 'price_volume', 'shrink_volume', 'volume_difference']
    available_cols = [col for col in sample_cols if col in matched_df.columns]
    print(matched_df[available_cols].head().to_string(index=False))
    
else:
    print("No matches found!")

print(f"\nProcess completed. Unmatched parts: {len(unmatched_results)}") 