
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
