import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🎯 CREATING PROJECT SCALE PRICING MODEL PKL")
print("=" * 60)

# Load dataset with all features
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (COMPLETE with ALL features).csv', sep=';', decimal=',')
print(f'Dataset shape: {df.shape}')

# FEATURES FOR PROJECT SCALE PRICING
project_scale_features = [
    # Original volumes
    'shrinkwrap_volume',
    'convex_hull_volume', 
    'bb_volume',
    'Volume',
    
    # Dimensions
    'Max D',
    'Min D',
    
    # Scale indicators
    'Quantity',
    'surface_area',
    
    # Full project volumes (project-level aggregated data)
    'Full_Project_Shrinkwrap_Volume_PA2200',
    'Full_Project_BB_Volume_PA2200',
    'Full_Project_Surface_Area_PA2200', 
    'Full_Project_Volume_PA2200',
    
    # Best derived feature
    'Size_Factor'
]

target = 'Net price / part'

# Convert to numeric
for col in project_scale_features + [target]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# CREATE SIZE_FACTOR (derived feature)
print('Creating Size_Factor derived feature...')
df['Size_Factor'] = np.cbrt(df['shrinkwrap_volume']) * np.sqrt(df['surface_area'])
print('✓ Size_Factor = ∛(shrinkwrap_volume) × √(surface_area)')

# Clean data
all_features = project_scale_features + [target]
df_clean = df[all_features].dropna()
print(f'Clean dataset shape: {df_clean.shape}')

# Prepare features and target
X = df_clean[project_scale_features]
y = df_clean[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Test model
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'\n🎯 MODEL PERFORMANCE:')
print(f'R² Score: {r2:.4f} ({r2*100:.1f}%)')
print(f'Mean Absolute Error: €{mae:.2f}')

# Create model package
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': project_scale_features,
    'target_name': target,
    'model_performance': {
        'r2_score': r2,
        'mae': mae,
        'intercept': model.intercept_
    },
    'feature_coefficients': dict(zip(project_scale_features, model.coef_)),
    'training_info': {
        'total_samples': len(df_clean),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': len(project_scale_features)
    }
}

# Save model as pickle
model_filename = 'project_scale_pricing_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)

print(f'\n💾 MODEL SAVED: {model_filename}')

# Create usage example
usage_example = '''
# USAGE EXAMPLE:
import pickle
import numpy as np

# Load the model
with open('project_scale_pricing_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler'] 
feature_names = model_package['feature_names']

# Example prediction
example_features = {
    'shrinkwrap_volume': 50000,
    'convex_hull_volume': 55000,
    'bb_volume': 80000,
    'Volume': 45000,
    'Max D': 100,
    'Min D': 50,
    'Quantity': 5,
    'surface_area': 15000,
    'Full_Project_Shrinkwrap_Volume_PA2200': 250000,
    'Full_Project_BB_Volume_PA2200': 400000,
    'Full_Project_Surface_Area_PA2200': 75000,
    'Full_Project_Volume_PA2200': 225000,
    'Size_Factor': np.cbrt(50000) * np.sqrt(15000)  # Calculate Size_Factor
}

# Prepare feature vector in correct order
feature_vector = [example_features[name] for name in feature_names]

# Scale and predict
feature_vector_scaled = scaler.transform([feature_vector])
predicted_price = model.predict(feature_vector_scaled)[0]

print(f"Predicted price: €{predicted_price:.2f}")
'''

# Save usage example
with open('model_usage_example.py', 'w') as f:
    f.write(usage_example)

print(f'\n📋 FEATURE NAMES (in order):')
for i, name in enumerate(project_scale_features, 1):
    coef = model.coef_[i-1]
    direction = '↑' if coef > 0 else '↓'
    print(f'{i:2d}. {name} {direction}')

print(f'\n📄 Usage example saved: model_usage_example.py')

# Test the saved model
print(f'\n🧪 TESTING SAVED MODEL:')
with open(model_filename, 'rb') as f:
    loaded_package = pickle.load(f)

loaded_model = loaded_package['model']
loaded_scaler = loaded_package['scaler']

# Test prediction
test_sample = X_test.iloc[0:1]
test_sample_scaled = loaded_scaler.transform(test_sample)
test_prediction = loaded_model.predict(test_sample_scaled)[0]
actual_price = y_test.iloc[0]

print(f'Test prediction: €{test_prediction:.2f}')
print(f'Actual price: €{actual_price:.2f}') 
print(f'Error: €{abs(test_prediction - actual_price):.2f}')

print(f'\n✅ MODEL PACKAGE READY FOR USE!')
print(f'File: {model_filename}')
print(f'Features: {len(project_scale_features)}')
print(f'Accuracy: {r2*100:.1f}%')
print(f'Average Error: ±€{mae:.2f}') 