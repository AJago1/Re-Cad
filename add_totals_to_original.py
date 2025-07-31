import pandas as pd
import numpy as np

# Read the original CSV file
df = pd.read_csv('Combined_Project_Data PA2200 EDITED 4 (P3 and p1, no full volumes).csv', 
                 delimiter=';', 
                 decimal=',')

print(f"Processing {len(df)} rows with {df['Project ID'].nunique()} unique projects...")

# Define the columns to calculate totals for
volume_columns = ['convex_hull_volume', 'bb_volume', 'shrinkwrap_volume', 'Volume']
area_columns = ['surface_area']

# Calculate project totals
project_totals = {}

for project_id in df['Project ID'].unique():
    project_data = df[df['Project ID'] == project_id]
    
    # Calculate totals
    totals = {
        'Total_Parts_in_Project': len(project_data),
        'Total_Quantity_in_Project': project_data['Quantity'].sum()
    }
    
    # Volume totals
    for col in volume_columns:
        if col in df.columns:
            total_volume = (project_data[col] * project_data['Quantity']).sum()
            totals[f'Project_Total_{col}'] = total_volume
    
    # Area totals  
    for col in area_columns:
        if col in df.columns:
            total_area = (project_data[col] * project_data['Quantity']).sum()
            totals[f'Project_Total_{col}'] = total_area
    
    project_totals[project_id] = totals

# Add totals to original dataframe
for col_name in ['Total_Parts_in_Project', 'Total_Quantity_in_Project', 
                 'Project_Total_convex_hull_volume', 'Project_Total_bb_volume', 
                 'Project_Total_shrinkwrap_volume', 'Project_Total_Volume', 
                 'Project_Total_surface_area']:
    df[col_name] = df['Project ID'].map(lambda pid: project_totals[pid][col_name])

# Save enhanced CSV
output_filename = 'Combined_Project_Data_WITH_TOTALS.csv'
df.to_csv(output_filename, sep=';', decimal=',', index=False)

print(f"\nEnhanced data saved to: {output_filename}")
print(f"Added columns:")
for col in ['Total_Parts_in_Project', 'Total_Quantity_in_Project', 
           'Project_Total_convex_hull_volume', 'Project_Total_bb_volume', 
           'Project_Total_shrinkwrap_volume', 'Project_Total_Volume', 
           'Project_Total_surface_area']:
    print(f"  - {col}")

# Show sample of enhanced data
print(f"\nSample of enhanced data (first 5 rows):")
print("=" * 120)
sample_cols = ['Project ID', 'Part Name', 'Quantity', 'Total_Parts_in_Project', 
               'Total_Quantity_in_Project', 'Project_Total_Volume', 'Project_Total_surface_area']
print(df[sample_cols].head())

print(f"\nFile saved with {len(df)} rows and {len(df.columns)} columns (original: 18, added: 7)") 