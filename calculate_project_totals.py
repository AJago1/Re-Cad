import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Combined_Project_Data PA2200 EDITED 4 (P3 and p1, no full volumes).csv', 
                 delimiter=';', 
                 decimal=',')

print(f"Processing {len(df)} rows with {df['Project ID'].nunique()} unique projects...")

# Define the volume and area columns to calculate totals for
volume_columns = ['convex_hull_volume', 'bb_volume', 'shrinkwrap_volume', 'Volume']
area_columns = ['surface_area']

# Calculate totals for each project
project_totals = []

for project_id in df['Project ID'].unique():
    project_data = df[df['Project ID'] == project_id]
    
    total_parts = len(project_data)
    total_quantity = project_data['Quantity'].sum()
    
    # Initialize result dictionary
    result = {
        'Project_ID': project_id,
        'Total_Parts': total_parts,
        'Total_Quantity': total_quantity
    }
    
    # Calculate totals for each volume column
    for col in volume_columns:
        if col in df.columns:
            # Multiply volume by quantity for each part, then sum
            total_volume = (project_data[col] * project_data['Quantity']).sum()
            result[f'Total_{col}'] = total_volume
    
    # Calculate totals for surface area
    for col in area_columns:
        if col in df.columns:
            # Multiply area by quantity for each part, then sum
            total_area = (project_data[col] * project_data['Quantity']).sum()
            result[f'Total_{col}'] = total_area
    
    project_totals.append(result)

# Create DataFrame from results
totals_df = pd.DataFrame(project_totals)

# Sort by Project ID
totals_df = totals_df.sort_values('Project_ID')

# Display results
print("\nProject Totals Summary:")
print("=" * 80)
print(f"{'Project ID':<12} {'Parts':<6} {'Qty':<8} {'Total_convex_hull':<18} {'Total_bb_volume':<18} {'Total_shrinkwrap':<18} {'Total_Volume':<15} {'Total_surface_area':<18}")
print("=" * 80)

for _, row in totals_df.head(20).iterrows():  # Show first 20 projects
    print(f"{row['Project_ID']:<12} {row['Total_Parts']:<6} {row['Total_Quantity']:<8} "
          f"{row['Total_convex_hull_volume']:<18,.0f} {row['Total_bb_volume']:<18,.0f} "
          f"{row['Total_shrinkwrap_volume']:<18,.0f} {row['Total_Volume']:<15,.0f} "
          f"{row['Total_surface_area']:<18,.0f}")

print(f"\n... and {len(totals_df)-20} more projects")

# Save to CSV file with semicolon delimiter and comma as decimal separator
output_filename = 'Project_Totals_Summary.csv'
totals_df.to_csv(output_filename, sep=';', decimal=',', index=False)

print(f"\nResults saved to: {output_filename}")
print(f"Total projects processed: {len(totals_df)}")

# Some statistics
print(f"\nStatistics:")
print(f"Average parts per project: {totals_df['Total_Parts'].mean():.1f}")
print(f"Average quantity per project: {totals_df['Total_Quantity'].mean():.1f}")
print(f"Largest project by total volume: {totals_df.loc[totals_df['Total_Volume'].idxmax(), 'Project_ID']} ({totals_df['Total_Volume'].max():,.0f})")
print(f"Largest project by surface area: {totals_df.loc[totals_df['Total_surface_area'].idxmax(), 'Project_ID']} ({totals_df['Total_surface_area'].max():,.0f})") 