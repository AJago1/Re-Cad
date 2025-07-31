import pandas as pd

# Read the enhanced CSV file
df = pd.read_csv('Combined_Project_Data_WITH_TOTALS.csv', delimiter=';', decimal=',')

print("NEW COLUMNS ADDED:")
new_cols = [col for col in df.columns if 'Total' in col or 'Project_Total' in col]
for col in new_cols:
    print(f"  - {col}")

print(f"\nORIGINAL DATA: {len(df)} rows, 18 columns")
print(f"ENHANCED DATA: {len(df)} rows, {len(df.columns)} columns")

print(f"\nSAMPLE RESULTS:")
print("="*100)
sample_data = df[['Project ID', 'Part Name', 'Quantity', 'Volume', 'surface_area', 
                  'Total_Parts_in_Project', 'Total_Quantity_in_Project', 
                  'Project_Total_Volume', 'Project_Total_surface_area']].head(8)

for idx, row in sample_data.iterrows():
    print(f"Project: {row['Project ID']:<8} | Part: {row['Part Name'][:30]:<30} | "
          f"Qty: {row['Quantity']:<4} | Vol: {row['Volume']:<10,.0f} | "
          f"Area: {row['surface_area']:<8,.0f}")
    if idx == 0 or row['Project ID'] != sample_data.iloc[idx-1]['Project ID']:
        print(f"  PROJECT TOTALS -> Parts: {row['Total_Parts_in_Project']:<2} | "
              f"Total Qty: {row['Total_Quantity_in_Project']:<4} | "
              f"Total Vol: {row['Project_Total_Volume']:<12,.0f} | "
              f"Total Area: {row['Project_Total_surface_area']:<10,.0f}")
    print()

print(f"\nSUMMARY STATISTICS:")
print(f"Total unique projects: {df['Project ID'].nunique()}")
print(f"Largest project by volume: {df.loc[df['Project_Total_Volume'].idxmax(), 'Project ID']} "
      f"({df['Project_Total_Volume'].max():,.0f})")
print(f"Largest project by surface area: {df.loc[df['Project_Total_surface_area'].idxmax(), 'Project ID']} "
      f"({df['Project_Total_surface_area'].max():,.0f})")

print(f"\nFILES CREATED:")
print(f"  1. Combined_Project_Data_WITH_TOTALS.csv - Original data with project totals added")
print(f"  2. Project_Totals_Summary.csv - Summary table with only project totals") 