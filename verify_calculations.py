import pandas as pd

# Read the fixed CSV
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 5.0 (fixed volumes).csv', sep=';', decimal=',')

# Check P-00810 project
print("Verifying P-00810 Project:")
p810 = df[df['Project ID'] == 'P-00810']
total_vol_qty = 0
for _, row in p810.iterrows():
    vol = float(str(row['Volume']).replace(',', '.'))
    qty = int(row['Quantity'])
    vol_qty = float(str(row['Volume*Quantity']).replace(',', '.'))
    total_vol_qty += vol_qty
    print(f"  {row['Part Name'][:30]:30} | Volume: {vol:8.2f} | Qty: {qty:2d} | Vol*Qty: {vol_qty:10.2f}")

full_project_vol = float(str(p810.iloc[0]['Full_Project_Volume_PA2200']).replace(',', '.'))
print(f"\nTotal Volume*Quantity: {total_vol_qty:.2f}")
print(f"Full_Project_Volume:   {full_project_vol:.2f}")
print(f"Match: {abs(total_vol_qty - full_project_vol) < 0.01}")

print("\n" + "="*70)

# Check P-00969 project
print("Verifying P-00969 Project:")
p969 = df[df['Project ID'] == 'P-00969']
total_vol_qty = 0
for _, row in p969.iterrows():
    vol = float(str(row['Volume']).replace(',', '.'))
    qty = int(row['Quantity'])
    vol_qty = float(str(row['Volume*Quantity']).replace(',', '.'))
    total_vol_qty += vol_qty
    print(f"  {row['Part Name'][:30]:30} | Volume: {vol:8.2f} | Qty: {qty:2d} | Vol*Qty: {vol_qty:10.2f}")

full_project_vol = float(str(p969.iloc[0]['Full_Project_Volume_PA2200']).replace(',', '.'))
print(f"\nTotal Volume*Quantity: {total_vol_qty:.2f}")
print(f"Full_Project_Volume:   {full_project_vol:.2f}")
print(f"Match: {abs(total_vol_qty - full_project_vol) < 0.01}")

print(f"\nTotal projects in CSV: {len(df['Project ID'].unique())}")
print(f"Total parts in CSV: {len(df)}") 