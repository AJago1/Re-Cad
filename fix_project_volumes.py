import pandas as pd
import numpy as np

def fix_project_volumes():
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (removed false prices).csv', 
                     sep=';', decimal=',')
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert Volume and Quantity columns to numeric, handling comma decimal separators
    print("Converting Volume and Quantity columns to numeric...")
    
    # Replace commas with dots for decimal conversion
    if df['Volume'].dtype == 'object':
        df['Volume'] = df['Volume'].str.replace(',', '.').astype(float)
    
    if df['Quantity'].dtype == 'object':
        df['Quantity'] = df['Quantity'].str.replace(',', '.').astype(int)
    
    # Create the Volume*Quantity column
    print("Creating Volume*Quantity column...")
    df['Volume*Quantity'] = df['Volume'] * df['Quantity']
    
    # Calculate the correct Full_Project_Volume for each project
    print("Calculating correct Full_Project_Volume by Project ID...")
    project_volumes = df.groupby('Project ID')['Volume*Quantity'].sum().to_dict()
    
    # Update the Full_Project_Volume_PA2200 column
    df['Full_Project_Volume_PA2200'] = df['Project ID'].map(project_volumes)
    
    # Display some statistics
    print("\nSample of the updated data:")
    print(df[['Project ID', 'Part Name', 'Volume', 'Quantity', 'Volume*Quantity', 'Full_Project_Volume_PA2200']].head(10))
    
    print(f"\nProject volume comparison (first 5 projects):")
    for project_id in df['Project ID'].unique()[:5]:
        project_data = df[df['Project ID'] == project_id]
        individual_volumes = project_data['Volume'].sum()
        corrected_volume = project_data['Volume*Quantity'].sum()
        print(f"{project_id}: Original sum of volumes = {individual_volumes:.2f}, "
              f"Corrected volume*quantity sum = {corrected_volume:.2f}")
    
    # Save the fixed CSV
    output_filename = 'Combined_Project_Data_with_Totals PA2200 EDITED 5.0 (fixed volumes).csv'
    print(f"\nSaving fixed CSV to: {output_filename}")
    
    # Convert back to comma decimal format for consistency
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'Quantity':  # Keep quantity as integer
            df[col] = df[col].apply(lambda x: f"{x:.10f}".replace('.', ',').rstrip('0').rstrip(',') if pd.notna(x) else x)
    
    df.to_csv(output_filename, sep=';', index=False)
    
    print("Done! The CSV has been fixed with:")
    print("1. New 'Volume*Quantity' column added")
    print("2. 'Full_Project_Volume_PA2200' column recalculated as sum of Volume*Quantity by Project ID")
    
    return output_filename

if __name__ == "__main__":
    fix_project_volumes() 