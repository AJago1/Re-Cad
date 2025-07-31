import pandas as pd

try:
    df = pd.read_csv('correct totals 3.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print("First 5 columns:", list(df.columns)[:5])
    print("Last 5 columns:", list(df.columns)[-5:])
    
    # Check for price column
    price_cols = [col for col in df.columns if 'price' in col.lower() or 'total' in col.lower()]
    print("Price-related columns:", price_cols)
    
except Exception as e:
    print(f"Error: {e}") 