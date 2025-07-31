import pandas as pd
import numpy as np

def load_and_clean_data():
    """Load both CSV files and clean the data"""
    # Load geometric data CSV (semicolon separated)
    geometric_df = pd.read_csv('Projects 500 to 1500 shrinkwraped and convex hull.csv', sep=';')
    
    # Load pricing data CSV (semicolon separated)  
    pricing_df = pd.read_csv('CSVB edited 500 to 1500.csv', sep=';')
    
    # Clean column names
    geometric_df.columns = geometric_df.columns.str.strip()
    pricing_df.columns = pricing_df.columns.str.strip()
    
    # Convert volume columns to numeric, handling European decimal format
    numeric_columns_geo = ['volume', 'surface_area', 'bb_volume', 'convex_hull_volume', 'shrinkwrap_volume']
    for col in numeric_columns_geo:
        if col in geometric_df.columns:
            geometric_df[col] = pd.to_numeric(geometric_df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    if 'Material Volume' in pricing_df.columns:
        pricing_df['Material Volume'] = pd.to_numeric(pricing_df['Material Volume'].astype(str).str.replace(',', '.'), errors='coerce')
    
    if 'Surface Area' in pricing_df.columns:
        pricing_df['Surface Area'] = pd.to_numeric(pricing_df['Surface Area'].astype(str).str.replace(',', '.'), errors='coerce')
    
    return geometric_df, pricing_df

def match_parts_by_volume(geometric_df, pricing_df, variance_threshold=0.005):
    """Match parts between datasets based on volume with specified variance threshold"""
    
    matched_pairs = []
    
    # Get unique projects that exist in both datasets
    geometric_projects = set(geometric_df['Project ID'].unique())
    pricing_projects = set(pricing_df['Project ID'].unique())
    matched_projects = geometric_projects.intersection(pricing_projects)
    
    for project_id in matched_projects:
        # Get parts for this project from both datasets
        geo_parts = geometric_df[geometric_df['Project ID'] == project_id].copy()
        price_parts = pricing_df[pricing_df['Project ID'] == project_id].copy()
        
        if geo_parts.empty or price_parts.empty:
            continue
            
        # Drop rows with missing volume data
        geo_parts = geo_parts.dropna(subset=['volume'])
        price_parts = price_parts.dropna(subset=['Material Volume'])
        
        if geo_parts.empty or price_parts.empty:
            continue
        
        # Track which indices have been matched to avoid duplicates
        matched_price_indices = set()
        
        for geo_idx, geo_row in geo_parts.iterrows():
            geo_vol = geo_row['volume']
            
            best_match = None
            best_diff = float('inf')
            
            for price_idx, price_row in price_parts.iterrows():
                if price_idx in matched_price_indices:
                    continue
                    
                price_vol = price_row['Material Volume']
                
                # Calculate percentage difference
                if geo_vol > 0 and price_vol > 0:
                    diff = abs(geo_vol - price_vol) / geo_vol
                    if diff <= variance_threshold and diff < best_diff:
                        best_match = price_idx
                        best_diff = diff
            
            if best_match is not None:
                matched_pairs.append((geo_idx, best_match))
                matched_price_indices.add(best_match)
    
    return matched_pairs

def calculate_project_totals(geometric_df):
    """Calculate project-level totals from geometric data"""
    project_totals = geometric_df.groupby('Project ID').agg({
        'volume': 'sum',
        'surface_area': 'sum', 
        'bb_volume': 'sum',
        'convex_hull_volume': 'sum',
        'shrinkwrap_volume': 'sum'
    }).reset_index()
    
    # Rename columns to indicate they are project totals
    project_totals.columns = [
        'Project ID',
        'Full_Project_Volume',
        'Full_Project_Surface_Area', 
        'Full_Project_BB_Volume',
        'Full_Project_Convex_Hull_Volume',
        'Full_Project_Shrinkwrap_Volume'
    ]
    
    return project_totals

def create_combined_dataset():
    """Create the combined dataset with all features"""
    
    print("Loading and cleaning data...")
    geometric_df, pricing_df = load_and_clean_data()
    
    print("Matching parts by volume...")
    matched_pairs = match_parts_by_volume(geometric_df, pricing_df)
    
    print(f"Successfully matched {len(matched_pairs)} parts")
    
    print("Calculating project totals...")
    project_totals = calculate_project_totals(geometric_df)
    
    print("Creating combined dataset...")
    combined_rows = []
    
    for geo_idx, price_idx in matched_pairs:
        geo_row = geometric_df.loc[geo_idx]
        price_row = pricing_df.loc[price_idx]
        
        # Create combined row
        combined_row = {}
        
        # Add all geometric data columns with 'Geo_' prefix to avoid conflicts
        for col in geometric_df.columns:
            combined_row[f'Geo_{col}'] = geo_row[col]
        
        # Add all pricing data columns with 'Price_' prefix to avoid conflicts  
        for col in pricing_df.columns:
            combined_row[f'Price_{col}'] = price_row[col]
        
        # Add project totals
        project_id = geo_row['Project ID']
        project_total_row = project_totals[project_totals['Project ID'] == project_id]
        
        if not project_total_row.empty:
            combined_row['Full_Project_Volume'] = project_total_row['Full_Project_Volume'].iloc[0]
            combined_row['Full_Project_Surface_Area'] = project_total_row['Full_Project_Surface_Area'].iloc[0]
            combined_row['Full_Project_BB_Volume'] = project_total_row['Full_Project_BB_Volume'].iloc[0]
            combined_row['Full_Project_Convex_Hull_Volume'] = project_total_row['Full_Project_Convex_Hull_Volume'].iloc[0]
            combined_row['Full_Project_Shrinkwrap_Volume'] = project_total_row['Full_Project_Shrinkwrap_Volume'].iloc[0]
        
        combined_rows.append(combined_row)
    
    # Create DataFrame
    combined_df = pd.DataFrame(combined_rows)
    
    # Reorder columns to put important ones first
    important_cols = [
        'Geo_Project ID', 'Geo_Part Name', 'Geo_Part Id', 'Geo_Num of parts', 'Geo_Material',
        'Price_Uploaded File', 'Price_Net Price', 'Price_Gross Price', 'Price_Technology', 'Price_Quantity',
        'Geo_volume', 'Price_Material Volume', 'Geo_surface_area', 'Price_Surface Area',
        'Full_Project_Volume', 'Full_Project_Surface_Area', 'Full_Project_BB_Volume', 
        'Full_Project_Convex_Hull_Volume', 'Full_Project_Shrinkwrap_Volume'
    ]
    
    # Get remaining columns
    remaining_cols = [col for col in combined_df.columns if col not in important_cols]
    
    # Filter important_cols to only include those that exist
    existing_important_cols = [col for col in important_cols if col in combined_df.columns]
    
    # Reorder columns
    final_columns = existing_important_cols + remaining_cols
    combined_df = combined_df[final_columns]
    
    return combined_df

def main():
    """Main function"""
    try:
        combined_df = create_combined_dataset()
        
        # Save to CSV
        output_filename = 'Combined_Project_Data_with_Totals.csv'
        combined_df.to_csv(output_filename, index=False, sep=';')
        
        print(f"\n=== COMBINED DATASET CREATED ===")
        print(f"Output file: {output_filename}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")
        
        print(f"\nColumn list:")
        for i, col in enumerate(combined_df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\nFirst few rows preview:")
        print(combined_df.head(3).to_string())
        
        print(f"\nProject totals sample:")
        sample_project = combined_df['Geo_Project ID'].iloc[0]
        sample_data = combined_df[combined_df['Geo_Project ID'] == sample_project].iloc[0]
        print(f"Project {sample_project}:")
        print(f"  Full Project Volume: {sample_data['Full_Project_Volume']:.2f}")
        print(f"  Full Project Surface Area: {sample_data['Full_Project_Surface_Area']:.2f}")
        print(f"  Full Project BB Volume: {sample_data['Full_Project_BB_Volume']:.2f}")
        print(f"  Full Project Convex Hull Volume: {sample_data['Full_Project_Convex_Hull_Volume']:.2f}")
        print(f"  Full Project Shrinkwrap Volume: {sample_data['Full_Project_Shrinkwrap_Volume']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 