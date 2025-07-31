import pandas as pd
from datetime import datetime

# Load the matched data
print("Loading matched data...")
df = pd.read_csv('matched_data_20250606_063220.csv', sep=';')
print(f"Loaded {len(df)} matched parts")

# Check available columns
print("\nChecking available volume/surface/bb columns...")
relevant_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['volume', 'surface', 'bb', 'bounding'])]
print(f"Found columns: {relevant_cols}")

# Calculate project-level aggregations
print("\nCalculating project-level totals...")
project_aggs = df.groupby('project_num').agg({
    'price_volume': 'sum',  # Total project volume from price data
    'shrink_volume': 'sum',  # Total shrinkwrap volume
    'shrink_surface_area': 'sum',  # Total surface area
    'shrink_bb_volume': 'sum',  # Total bounding box volume
    'price_Part ID': 'count'  # Number of parts in project
}).reset_index()

# Rename columns for clarity
project_aggs.columns = [
    'project_num',
    'project_total_price_volume',
    'project_total_shrinkwrap_volume', 
    'project_total_surface_area',
    'project_total_bb_volume',
    'project_part_count'
]

print(f"Calculated totals for {len(project_aggs)} projects")

# Merge back with original data
print("Adding project totals to each row...")
df_enhanced = df.merge(project_aggs, on='project_num', how='left')

# Calculate additional project-level ratios
df_enhanced['project_volume_efficiency'] = df_enhanced['project_total_price_volume'] / df_enhanced['project_total_bb_volume']
df_enhanced['project_surface_to_volume_ratio'] = df_enhanced['project_total_surface_area'] / df_enhanced['project_total_price_volume']
df_enhanced['project_shrinkwrap_ratio'] = df_enhanced['project_total_shrinkwrap_volume'] / df_enhanced['project_total_price_volume']

# Save enhanced data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'matched_data_with_project_totals_{timestamp}.csv'
df_enhanced.to_csv(output_file, index=False, sep=';')

print(f"\nEnhanced data saved to: {output_file}")
print(f"Added {len(project_aggs.columns)} new project-level columns")

# Show summary
print("\n" + "="*60)
print("PROJECT TOTALS SUMMARY")
print("="*60)
print("Sample project totals (first 5 projects):")
sample_cols = [
    'project_num', 
    'project_part_count',
    'project_total_price_volume',
    'project_total_shrinkwrap_volume',
    'project_total_surface_area',
    'project_total_bb_volume'
]
print(project_aggs[sample_cols].head().to_string(index=False))

print(f"\nTotal parts across all projects: {df_enhanced['project_part_count'].iloc[0] if len(df_enhanced) > 0 else 0}")
print(f"Average parts per project: {project_aggs['project_part_count'].mean():.1f}")
print(f"Total volume across all projects: {project_aggs['project_total_price_volume'].sum():,.2f}")
print(f"Total surface area across all projects: {project_aggs['project_total_surface_area'].sum():,.2f}")

# Show which columns were added
print(f"\nNew columns added to the CSV:")
new_cols = [col for col in df_enhanced.columns if col not in df.columns]
for col in new_cols:
    print(f"  - {col}") 