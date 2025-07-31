import pandas as pd

# Load the matched data
df = pd.read_csv('matched_data_20250606_063220.csv', sep=';')

# Get volume ratios (exclude infinite values)
ratios = df['volume_ratio']
ratios_finite = ratios[ratios != float('inf')]

# Calculate matches within different tolerances
within_1_percent = ratios_finite[(ratios_finite >= 0.99) & (ratios_finite <= 1.01)]
within_2_percent = ratios_finite[(ratios_finite >= 0.98) & (ratios_finite <= 1.02)]
within_5_percent = ratios_finite[(ratios_finite >= 0.95) & (ratios_finite <= 1.05)]
within_10_percent = ratios_finite[(ratios_finite >= 0.9) & (ratios_finite <= 1.1)]

print("PRECISION MATCHING ANALYSIS:")
print("="*50)
print(f"Total matches: {len(ratios_finite)}")
print()
print(f"Within ±1%:  {len(within_1_percent):3d} ({len(within_1_percent)/len(ratios_finite)*100:.1f}%)")
print(f"Within ±2%:  {len(within_2_percent):3d} ({len(within_2_percent)/len(ratios_finite)*100:.1f}%)")
print(f"Within ±5%:  {len(within_5_percent):3d} ({len(within_5_percent)/len(ratios_finite)*100:.1f}%)")
print(f"Within ±10%: {len(within_10_percent):3d} ({len(within_10_percent)/len(ratios_finite)*100:.1f}%)")
print()

print("Volume ratio statistics:")
print(f"Mean: {ratios_finite.mean():.3f}")
print(f"Std:  {ratios_finite.std():.3f}")
print(f"Min:  {ratios_finite.min():.3f}")
print(f"Max:  {ratios_finite.max():.3f}")
print()

# Show exact matches (volume difference = 0)
exact_matches = df[df['volume_difference'] == 0.0]
print(f"EXACT VOLUME MATCHES: {len(exact_matches)} ({len(exact_matches)/len(df)*100:.1f}%)")
print()

# Show worst matches
print("WORST MATCHES (highest volume difference):")
worst_matches = df.nlargest(5, 'volume_difference')[['price_Part ID', 'shrink_name', 'price_volume', 'shrink_volume', 'volume_difference', 'volume_ratio']]
print(worst_matches.to_string(index=False)) 