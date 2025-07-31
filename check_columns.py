import pandas as pd

def check_columns():
    # Load the data
    df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (removed false prices).csv', sep=';')
    df.columns = df.columns.str.strip()
    
    print("COLUMN ANALYSIS")
    print("="*50)
    print(f"Total columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    print(f"\nAll columns in CSV:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Check for convex hull specifically
    convex_cols = [col for col in df.columns if 'convex' in col.lower()]
    print(f"\nColumns containing 'convex': {convex_cols}")
    
    # Check which geometric features are available
    expected_geometric = [
        'convex_hull_volume', 'bb_volume', 'surface_area', 'Volume',
        'Max D', 'Min D', 'D Ratio',
        'waste', 'convexity_ratio', 'shrinkwrap_volume', 'shrinkwrap_ratio', 'waste_ratio'
    ]
    
    print(f"\nExpected geometric features:")
    for feature in expected_geometric:
        exists = feature in df.columns
        print(f"  {feature:<25} {'✅' if exists else '❌'}")
    
    # Show a sample of the data
    print(f"\nFirst 3 rows of data:")
    print(df.head(3).to_string())

if __name__ == "__main__":
    check_columns()

df = pd.read_csv('correct totals 2.csv', sep=';')
print('Columns in new file:')
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'") 