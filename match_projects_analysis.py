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
    
    print(f"Geometric data columns: {list(geometric_df.columns)}")
    print(f"Pricing data columns: {list(pricing_df.columns)}")
    
    # Convert volume columns to numeric, handling European decimal format
    if 'volume' in geometric_df.columns:
        geometric_df['volume'] = pd.to_numeric(geometric_df['volume'].astype(str).str.replace(',', '.'), errors='coerce')
    
    if 'Material Volume' in pricing_df.columns:
        pricing_df['Material Volume'] = pd.to_numeric(pricing_df['Material Volume'].astype(str).str.replace(',', '.'), errors='coerce')
    
    return geometric_df, pricing_df

def match_project_ids(geometric_df, pricing_df):
    """Match project IDs between the two datasets"""
    geometric_projects = set(geometric_df['Project ID'].unique())
    pricing_projects = set(pricing_df['Project ID'].unique())
    
    matched_projects = geometric_projects.intersection(pricing_projects)
    
    print(f"\n=== PROJECT ID MATCHING REPORT ===")
    print(f"Geometric data projects: {len(geometric_projects)}")
    print(f"Pricing data projects: {len(pricing_projects)}")
    print(f"Matched projects: {len(matched_projects)}")
    print(f"Match rate: {len(matched_projects)/max(len(geometric_projects), len(pricing_projects))*100:.1f}%")
    
    # Show some examples of matched projects
    print(f"\nFirst 10 matched projects: {sorted(list(matched_projects))[:10]}")
    
    return matched_projects

def match_parts_by_volume(geometric_df, pricing_df, matched_projects, variance_threshold=0.005):
    """Match parts within projects based on volume with specified variance threshold"""
    
    total_matches = 0
    total_geometric_parts = 0
    total_pricing_parts = 0
    
    print(f"\n=== PART MATCHING BY VOLUME (max {variance_threshold*100}% variance) ===")
    
    for project_id in sorted(matched_projects):
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
            
        geo_volumes = geo_parts['volume'].values
        price_volumes = price_parts['Material Volume'].values
        
        total_geometric_parts += len(geo_volumes)
        total_pricing_parts += len(price_volumes)
        
        # Find matches within variance threshold
        project_matches = 0
        matched_geo_indices = set()
        matched_price_indices = set()
        
        for i, geo_vol in enumerate(geo_volumes):
            for j, price_vol in enumerate(price_volumes):
                if j in matched_price_indices:
                    continue
                    
                # Calculate percentage difference
                if geo_vol > 0 and price_vol > 0:
                    diff = abs(geo_vol - price_vol) / geo_vol
                    if diff <= variance_threshold:
                        project_matches += 1
                        matched_geo_indices.add(i)
                        matched_price_indices.add(j)
                        break
        
        total_matches += project_matches
        
        # Print detailed info for projects with good match rates
        if len(geo_parts) > 0 and len(price_parts) > 0:
            match_rate = project_matches / min(len(geo_parts), len(price_parts)) * 100
            if match_rate > 50:  # Only show projects with >50% match rate
                print(f"\nProject {project_id}:")
                print(f"  Geometric parts: {len(geo_parts)}, Pricing parts: {len(price_parts)}")
                print(f"  Volume matches: {project_matches} ({match_rate:.1f}% match rate)")
                
                # Show sample volumes for debugging
                print(f"  Sample geo volumes: {sorted(geo_volumes)[:5]}")
                print(f"  Sample price volumes: {sorted(price_volumes)[:5]}")
    
    print(f"\n=== OVERALL MATCHING SUMMARY ===")
    print(f"Total geometric parts: {total_geometric_parts}")
    print(f"Total pricing parts: {total_pricing_parts}")
    print(f"Total volume matches: {total_matches}")
    if total_geometric_parts > 0:
        print(f"Geometric parts match rate: {total_matches/total_geometric_parts*100:.1f}%")
    if total_pricing_parts > 0:
        print(f"Pricing parts match rate: {total_matches/total_pricing_parts*100:.1f}%")

def main():
    """Main function to run the analysis"""
    try:
        # Load data
        geometric_df, pricing_df = load_and_clean_data()
        
        # Match project IDs
        matched_projects = match_project_ids(geometric_df, pricing_df)
        
        # Match parts by volume
        match_parts_by_volume(geometric_df, pricing_df, matched_projects)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 