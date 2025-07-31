"""
Comprehensive Pricing Model for STL Analyzer GUI
Uses the top 5 predictive features: Size_Factor, shrinkwrap_volume, convex_hull_volume, bb_volume, surface_area
"""

import numpy as np
import pickle
import json
import os
from typing import Dict, Any, Tuple, Optional

class ComprehensivePricingModel:
    """
    Comprehensive pricing model using optimal 5 features for 3D printing price prediction
    Features: Size_Factor, shrinkwrap_volume, convex_hull_volume, bb_volume, surface_area
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the pricing model
        
        Args:
            model_path: Path to saved model file (.json or .pkl)
        """
        # Default model parameters (from our analysis)
        self.default_params = {
            'features': ['Size_Factor', 'shrinkwrap_volume', 'convex_hull_volume', 'bb_volume', 'surface_area'],
            'coefficients': [137.273097, 8.679136, 27.367166, 43.446650, -23.383602],
            'intercept': 64.948784,
            'feature_means': [12518.206330, 287293.901400, 1243161.945407, 2080028.729143, 51632.368134],
            'feature_scales': [25032.783691, 776109.360299, 9999710.459629, 15641128.935934, 146231.991264],
            'r2_score': 0.9432,
            'mae': 19.45,
            'model_name': 'Comprehensive 5-Feature Model'
        }
        
        # Initialize with default parameters
        self.features = self.default_params['features']
        self.coefficients = np.array(self.default_params['coefficients'])
        self.intercept = self.default_params['intercept']
        self.feature_means = np.array(self.default_params['feature_means'])
        self.feature_scales = np.array(self.default_params['feature_scales'])
        self.r2_score = self.default_params['r2_score']
        self.mae = self.default_params['mae']
        self.model_name = self.default_params['model_name']
        
        # Load custom model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def calculate_size_factor(self, shrinkwrap_volume: float, surface_area: float) -> float:
        """
        Calculate the Size_Factor derived feature
        Formula: ∛(shrinkwrap_volume) × √(surface_area)
        """
        if shrinkwrap_volume <= 0 or surface_area <= 0:
            return 0.0
        
        size_factor = (shrinkwrap_volume ** (1/3)) * (surface_area ** 0.5)
        return size_factor
    
    def predict_price(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict price using the comprehensive 5-feature model
        
        Args:
            features: Dictionary containing part features
                Required: shrinkwrap_volume, convex_hull_volume, bb_volume, surface_area
                
        Returns:
            Tuple of (predicted_price, detailed_breakdown)
        """
        try:
            # Extract required features
            shrinkwrap_volume = features.get('shrinkwrap_volume', 0)
            convex_hull_volume = features.get('convex_hull_volume', 0)
            bb_volume = features.get('bb_volume', 0)
            surface_area = features.get('surface_area', 0)
            
            # Validate inputs
            if any(val <= 0 for val in [shrinkwrap_volume, convex_hull_volume, bb_volume, surface_area]):
                return 0.0, {'error': 'Invalid feature values - all volumes and surface area must be > 0'}
            
            # Calculate derived feature
            size_factor = self.calculate_size_factor(shrinkwrap_volume, surface_area)
            
            # Prepare feature vector
            feature_values = np.array([
                size_factor,
                shrinkwrap_volume,
                convex_hull_volume,
                bb_volume,
                surface_area
            ])
            
            # Standardize features
            feature_values_scaled = (feature_values - self.feature_means) / self.feature_scales
            
            # Calculate prediction
            prediction = self.intercept + np.dot(self.coefficients, feature_values_scaled)
            
            # Ensure non-negative price
            final_price = max(0.0, prediction)
            
            # Create detailed breakdown
            breakdown = {
                'model_name': self.model_name,
                'features_used': {
                    'Size_Factor': size_factor,
                    'shrinkwrap_volume': shrinkwrap_volume,
                    'convex_hull_volume': convex_hull_volume,
                    'bb_volume': bb_volume,
                    'surface_area': surface_area
                },
                'scaled_features': {
                    f'{feature}_scaled': scaled_val 
                    for feature, scaled_val in zip(self.features, feature_values_scaled)
                },
                'feature_contributions': {
                    feature: coef * scaled_val 
                    for feature, coef, scaled_val in zip(self.features, self.coefficients, feature_values_scaled)
                },
                'intercept': self.intercept,
                'raw_prediction': prediction,
                'final_price': final_price,
                'model_accuracy': f"R² = {self.r2_score:.1%}, MAE = €{self.mae:.2f}",
                'confidence': 'High' if abs(prediction - final_price) < 0.01 else 'Medium'
            }
            
            return final_price, breakdown
            
        except Exception as e:
            return 0.0, {'error': f'Prediction failed: {str(e)}'}
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a pricing model from file
        
        Args:
            model_path: Path to model file (.json or .pkl)
            
        Returns:
            bool: Success status
        """
        try:
            if model_path.endswith('.json'):
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                print(f"Unsupported model file format: {model_path}")
                return False
            
            # Update model parameters
            self.features = model_data.get('features', self.features)
            self.coefficients = np.array(model_data.get('coefficients', self.coefficients))
            self.intercept = model_data.get('intercept', self.intercept)
            self.feature_means = np.array(model_data.get('feature_means', self.feature_means))
            self.feature_scales = np.array(model_data.get('feature_scales', self.feature_scales))
            self.r2_score = model_data.get('r2_score', self.r2_score)
            self.mae = model_data.get('mae', self.mae)
            self.model_name = model_data.get('model_name', f'Custom Model from {os.path.basename(model_path)}')
            
            print(f"✅ Successfully loaded model: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        Save the current model to file
        
        Args:
            model_path: Path to save model (.json or .pkl)
            
        Returns:
            bool: Success status
        """
        try:
            model_data = {
                'features': self.features,
                'coefficients': self.coefficients.tolist(),
                'intercept': self.intercept,
                'feature_means': self.feature_means.tolist(),
                'feature_scales': self.feature_scales.tolist(),
                'r2_score': self.r2_score,
                'mae': self.mae,
                'model_name': self.model_name
            }
            
            if model_path.endswith('.json'):
                with open(model_path, 'w') as f:
                    json.dump(model_data, f, indent=2)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            else:
                print(f"Unsupported file format: {model_path}")
                return False
            
            print(f"✅ Model saved to: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model to {model_path}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'name': self.model_name,
            'features': self.features,
            'accuracy': f"R² = {self.r2_score:.1%}",
            'average_error': f"€{self.mae:.2f}",
            'feature_count': len(self.features)
        }
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate that required features are present and valid
        
        Args:
            features: Dictionary of features to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_features = ['shrinkwrap_volume', 'convex_hull_volume', 'bb_volume', 'surface_area']
        
        # Check for missing features
        missing = [f for f in required_features if f not in features]
        if missing:
            return False, f"Missing required features: {', '.join(missing)}"
        
        # Check for invalid values
        invalid = [f for f in required_features if features[f] <= 0]
        if invalid:
            return False, f"Invalid values (must be > 0): {', '.join(invalid)}"
        
        return True, "Features are valid"


class PricingModelManager:
    """Manager class for handling multiple pricing models"""
    
    def __init__(self, models_directory: str = "pricing_models"):
        """
        Initialize the pricing model manager
        
        Args:
            models_directory: Directory to search for model files
        """
        self.models_directory = models_directory
        self.available_models = {}
        self.current_model = ComprehensivePricingModel()  # Default model
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)
        
        # Scan for available models
        self.refresh_available_models()
    
    def refresh_available_models(self):
        """Scan the models directory for available pricing models"""
        self.available_models = {'Default (Built-in)': None}  # Default model
        
        if not os.path.exists(self.models_directory):
            return
        
        for filename in os.listdir(self.models_directory):
            if filename.endswith(('.json', '.pkl')):
                model_path = os.path.join(self.models_directory, filename)
                model_name = os.path.splitext(filename)[0]
                self.available_models[model_name] = model_path
    
    def get_available_models(self) -> Dict[str, Optional[str]]:
        """Get dictionary of available models {name: path}"""
        return self.available_models.copy()
    
    def set_current_model(self, model_name: str) -> bool:
        """
        Set the current active model
        
        Args:
            model_name: Name of model to activate
            
        Returns:
            bool: Success status
        """
        if model_name not in self.available_models:
            return False
        
        model_path = self.available_models[model_name]
        
        if model_path is None:  # Default model
            self.current_model = ComprehensivePricingModel()
        else:
            self.current_model = ComprehensivePricingModel(model_path)
        
        return True
    
    def predict_price(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """Predict price using the current model"""
        return self.current_model.predict_price(features)
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.current_model.get_model_info() 