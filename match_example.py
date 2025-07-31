import pandas as pd
import re
import numpy as np

# Load data
df1 = pd.read_csv('Urejeni Podatki 2.10.csv', sep=';')
df2 = pd.read_csv('Convex hull + shrinkwraped 4mm all files. (urejen - brez cene).csv', sep=';')

# Extract project numbers
df1['project_num'] = df1['Project ID'].apply(lambda x: int(re.findall(r'\d+', str(x))[-1]) if pd.notna(x) and re.findall(r'\d+', str(x)) else None)
df2['project_num'] = df2['Project ID'].apply(lambda x: int(re.findall(r'\d+', str(x))[-1]) if pd.notna(x) and re.findall(r'\d+', str(x)) else None)

# Convert volumes to numeric
df1['volume_numeric'] = df1['Material Volume'].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else np.nan)
df2['volume_numeric'] = df2['volume'].apply(lambda x: float(str(x).replace(',', '.')) if pd.notna(x) else np.nan)

# Show example matching for project 72
print("PROJECT 72 MATCHING EXAMPLE:")
print("="*50)

proj_files = df1[df1['project_num'] == 72]
shrink_files = df2[df2['project_num'] == 72]

print(f"Price file has {len(proj_files)} parts")
print(f"Shrinkwrap file has {len(shrink_files)} parts")
print()

print("Price file parts (first 5):")
print(proj_files[['Part ID', 'volume_numeric']].head().to_string(index=False))
print()

print("Shrinkwrap file parts (first 5):")
print(shrink_files[['name', 'volume_numeric']].head().to_string(index=False))
print()

# Example of matching one part by closest volume
if len(proj_files) > 0 and len(shrink_files) > 0:
    sample_part = proj_files.iloc[0]
    sample_volume = sample_part['volume_numeric']
    
    # Find closest volume match in shrinkwrap data
    volume_diffs = abs(shrink_files['volume_numeric'] - sample_volume)
    closest_match = shrink_files.loc[volume_diffs.idxmin()]
    
    print(f"MATCHING EXAMPLE:")
    print(f"Price part: {sample_part['Part ID']} (volume: {sample_volume:.2f})")
    print(f"Best match: {closest_match['name']} (volume: {closest_match['volume_numeric']:.2f})")
    print(f"Volume difference: {abs(sample_volume - closest_match['volume_numeric']):.2f}") 