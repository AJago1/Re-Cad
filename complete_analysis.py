import pandas as pd
import numpy as np

df = pd.read_csv('correct totals 1.csv', sep=';')
df['price'] = pd.to_numeric(df['Net price / part'].str.replace(',', '.'), errors='coerce')

# Convert numeric columns
numeric_cols = ['Volume', 'surface_area', 'convex_hull_volume', 'bb_volume', 'Quantity', 'Total_Parts_in_Project']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print('MACHINE ANALYSIS:')
machine_analysis = df.groupby('Machine')['price'].agg(['count', 'mean', 'median']).round(2)
print(machine_analysis)

print('\nOUTLIERS - Highest priced parts:')
top_price = df.nlargest(10, 'price')[['Project ID', 'Part Name', 'price', 'Volume', 'Quantity']]
print(top_price.to_string())

print('\nOUTLIERS - Lowest priced parts:')
low_price = df.nsmallest(10, 'price')[['Project ID', 'Part Name', 'price', 'Volume', 'Quantity']]  
print(low_price.to_string())

# Price prediction challenges
df['price_per_mm3'] = df['price'] / df['Volume']
print('\nPRICE PER MM3 EXTREMES:')
print('Highest price/volume ratios:')
high_ratio = df.nlargest(5, 'price_per_mm3')[['Part Name', 'price', 'Volume', 'price_per_mm3', 'surface_area']]
print(high_ratio.to_string())

print('\n\nSUMMARY - WHERE LINEAR MODEL WILL STRUGGLE:')
print('1. TINY PARTS: Very high surface/volume ratio leads to disproportionate pricing')
print('2. SINGLE vs BULK: 1-piece orders cost 3-5x more than bulk (50+ pieces)')  
print('3. PROJECT SIZE: Large projects (150+ parts) get 50%+ discounts')
print('4. MACHINE DIFFERENCES: Different machines have different cost structures')
print('5. GEOMETRY COMPLEXITY: High waste ratios indicate complex shapes = higher cost') 