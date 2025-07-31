import pandas as pd
import numpy as np

def load_and_clean_data():
    """Load the PA2200-only CSV file and clean the data"""
    df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 only.csv', sep=';')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print(f"Loaded CSV columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Convert numeric columns, handling European decimal format
    numeric_columns = ['Geo_volume', 'Geo_surface_area', 'Geo_bb_volume', 
                      'Geo_convex_hull_volume', 'Geo_shrinkwrap_volume']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            print(f"Converted {col} to numeric")
    
    return df

def calculate_project_totals(df):
    """Calculate project-level totals for PA2200 parts only"""
    
    # Group by Project ID and calculate sums
    project_totals = df.groupby('Geo_Project ID').agg({
        'Geo_volume': 'sum',
        'Geo_surface_area': 'sum', 
        'Geo_bb_volume': 'sum',
        'Geo_convex_hull_volume': 'sum',
        'Geo_shrinkwrap_volume': 'sum'
    }).reset_index()
    
    # Rename columns to indicate they are project totals
    project_totals.columns = [
        'Geo_Project ID',
        'Full_Project_Volume_PA2200',
        'Full_Project_Surface_Area_PA2200', 
        'Full_Project_BB_Volume_PA2200',
        'Full_Project_Convex_Hull_Volume_PA2200',
        'Full_Project_Shrinkwrap_Volume_PA2200'
    ]
    
    print(f"Calculated totals for {len(project_totals)} projects")
    return project_totals

def add_project_totals_to_dataframe(df, project_totals):
    """Add project totals to each row in the dataframe"""
    
    # Merge project totals with the main dataframe
    df_with_totals = df.merge(project_totals, on='Geo_Project ID', how='left')
    
    # Reorder columns to put project totals after the main data
    project_total_cols = [
        'Full_Project_Volume_PA2200',
        'Full_Project_Surface_Area_PA2200', 
        'Full_Project_BB_Volume_PA2200',
        'Full_Project_Convex_Hull_Volume_PA2200',
        'Full_Project_Shrinkwrap_Volume_PA2200'
    ]
    
    # Get all other columns
    other_cols = [col for col in df_with_totals.columns if col not in project_total_cols]
    
    # Reorder: put project totals after the geometric data
    # Find a good position to insert project totals
    insert_position = len(other_cols)  # Default to end
    
    # Try to find a good position after geometric data
    for i, col in enumerate(other_cols):
        if col.startswith('Price_'):
            insert_position = i
            break
    
    # Insert project total columns at the determined position
    final_columns = other_cols[:insert_position] + project_total_cols + other_cols[insert_position:]
    
    df_with_totals = df_with_totals[final_columns]
    
    return df_with_totals

def main():
    """Main function"""
    try:
        print("Loading PA2200-only data...")
        df = load_and_clean_data()
        
        print("Calculating project totals...")
        project_totals = calculate_project_totals(df)
        
        print("Adding project totals to dataframe...")
        df_with_totals = add_project_totals_to_dataframe(df, project_totals)
        
        # Save to new CSV
        output_filename = 'Combined_Project_Data_PA2200_with_Project_Totals.csv'
        df_with_totals.to_csv(output_filename, index=False, sep=';')
        
        print(f"\n=== PA2200 DATASET WITH PROJECT TOTALS CREATED ===")
        print(f"Output file: {output_filename}")
        print(f"Total rows: {len(df_with_totals)}")
        print(f"Total columns: {len(df_with_totals.columns)}")
        
        print(f"\nNew project total columns added:")
        project_total_cols = [col for col in df_with_totals.columns if col.startswith('Full_Project_')]
        for col in project_total_cols:
            print(f"  - {col}")
        
        print(f"\nSample project totals:")
        if len(df_with_totals) > 0:
            sample_project = df_with_totals['Geo_Project ID'].iloc[0]
            sample_data = df_with_totals[df_with_totals['Geo_Project ID'] == sample_project].iloc[0]
            print(f"Project {sample_project} (PA2200 parts only):")
            for col in project_total_cols:
                if col in sample_data:
                    print(f"  {col}: {sample_data[col]:.2f}")
        
        print(f"\nFirst few column names:")
        for i, col in enumerate(df_with_totals.columns[:15], 1):
            print(f"{i:2d}. {col}")
        
        # Show some statistics
        print(f"\nProject statistics:")
        unique_projects = df_with_totals['Geo_Project ID'].nunique()
        total_parts = len(df_with_totals)
        print(f"  Unique projects: {unique_projects}")
        print(f"  Total PA2200 parts: {total_parts}")
        print(f"  Average parts per project: {total_parts/unique_projects:.1f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 