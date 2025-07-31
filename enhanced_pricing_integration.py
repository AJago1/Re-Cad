"""
Enhanced Pricing Integration Module
Provides non-linear feature calculation for the GUI pricing system
"""

import numpy as np
import pickle
import json
import os

class EnhancedPricingCalculator:
    """Enhanced pricing calculator with non-linear derived features"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_coefficients = {}
        self.model_metadata = {}
        
    def load_enhanced_model(self):
        """Load the enhanced Lasso model with non-linear features"""
        try:
            # Load model
            with open('enhanced_lasso_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open('enhanced_feature_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load metadata
            with open('enhanced_model_metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
            
            self.feature_names = self.model_metadata['selected_features']
            self.feature_coefficients = self.model_metadata['feature_coefficients']
            
            print(f"✅ Enhanced model loaded successfully!")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   R² Score: {self.model_metadata['performance']['test_r2']:.3f}")
            print(f"   MAE: €{self.model_metadata['performance']['test_mae']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load enhanced model: {e}")
            return False
    
    def calculate_nonlinear_features(self, part_data):
        """Calculate all non-linear derived features for a part"""
        try:
            # Extract base features
            volume = part_data.get('volume', 0)
            surface_area = part_data.get('surface_area', 0)
            shrinkwrap_vol = part_data.get('shrinkwrap_volume', 0)
            convex_hull_vol = part_data.get('convex_hull_volume', 0)
            bb_volume = part_data.get('bb_volume', 0)
            max_dim = part_data.get('x', 0)  # Largest dimension
            min_dim = part_data.get('z', 0)  # Smallest dimension
            waste = part_data.get('waste', 0)
            quantity = part_data.get('quantity', 1)
            
            # Ensure no zero/negative values for calculations
            volume = max(volume, 1)
            surface_area = max(surface_area, 1)
            shrinkwrap_vol = max(shrinkwrap_vol, 1)
            convex_hull_vol = max(convex_hull_vol, 1)
            bb_volume = max(bb_volume, 1)
            max_dim = max(max_dim, 0.1)
            min_dim = max(min_dim, 0.1)
            
            # Calculate all derived features
            features = {}
            
            # Base features
            features['shrinkwrap_volume'] = shrinkwrap_vol
            features['convex_hull_volume'] = convex_hull_vol
            features['Volume'] = volume
            features['Min D'] = min_dim
            features['Max D'] = max_dim
            features['D Ratio'] = max_dim / min_dim
            features['waste'] = waste
            features['surface_area'] = surface_area
            features['bb_volume'] = bb_volume
            features['Quantity'] = quantity
            
            # 1. SIZE-DEPENDENT FEATURES
            features['size_factor'] = volume ** (1/3)
            features['micro_part_penalty'] = 1 / (1 + volume/1000)
            features['large_part_efficiency'] = volume / (volume + 10000)
            features['size_category_micro'] = int(volume < 1000)
            features['size_category_small'] = int(1000 <= volume < 10000)
            features['size_category_medium'] = int(10000 <= volume < 50000)
            features['size_category_large'] = int(volume >= 50000)
            
            # 2. LOGARITHMIC TRANSFORMATIONS
            features['log_volume'] = np.log1p(volume)
            features['log_surface_area'] = np.log1p(surface_area)
            features['log_shrinkwrap_volume'] = np.log1p(shrinkwrap_vol)
            features['log_convex_hull_volume'] = np.log1p(convex_hull_vol)
            features['log_bb_volume'] = np.log1p(bb_volume)
            
            # 3. POWER TRANSFORMATIONS
            features['volume_sqrt'] = volume ** 0.5
            features['volume_cbrt'] = volume ** (1/3)
            features['surface_area_sqrt'] = surface_area ** 0.5
            features['shrinkwrap_sqrt'] = shrinkwrap_vol ** 0.5
            
            # 4. COMPLEXITY-SIZE INTERACTIONS
            features['complexity_size_ratio'] = (surface_area / volume) * features['micro_part_penalty']
            features['dimension_instability'] = (max_dim / min_dim) * features['micro_part_penalty']
            features['waste_complexity'] = (waste / volume) * features['micro_part_penalty']
            features['surface_volume_interaction'] = (surface_area * volume) ** 0.5
            features['complexity_volume_interaction'] = (surface_area / volume) * features['log_volume']
            
            # 5. ECONOMIC BEHAVIOR FEATURES
            features['fixed_cost_impact'] = 1000 / max(volume, 100)
            features['handling_complexity'] = 1 / (shrinkwrap_vol ** 0.3)
            features['material_waste_curve'] = (waste / bb_volume) * (1 + features['micro_part_penalty'])
            features['support_material_factor'] = ((bb_volume - volume) / volume) * features['micro_part_penalty']
            
            # 6. MANUFACTURING EFFICIENCY FEATURES
            features['scale_efficiency'] = quantity * (volume ** 0.3)
            features['batch_efficiency'] = quantity / (1 + 1000/volume)
            features['manufacturing_challenge'] = (surface_area / volume) / (volume ** 0.2)
            features['precision_requirement'] = (max_dim / min_dim) / (volume ** 0.1)
            
            # 7. MATERIAL UTILIZATION FEATURES
            features['material_density'] = volume / bb_volume
            features['packing_efficiency'] = shrinkwrap_vol / bb_volume
            features['shape_complexity'] = surface_area / (volume ** (2/3))
            features['convexity_efficiency'] = volume / convex_hull_vol
            
            # 8. DIMENSIONAL ANALYSIS FEATURES
            features['aspect_ratio_penalty'] = max(max_dim/min_dim - 1, 0) ** 0.5
            features['dimensional_balance'] = min_dim * max_dim / volume
            features['slenderness_ratio'] = max_dim / (volume ** (1/3))
            
            # 9. QUANTITY-DEPENDENT FEATURES
            features['quantity_volume_interaction'] = quantity * features['log_volume']
            features['quantity_complexity_interaction'] = quantity * (surface_area / volume)
            features['quantity_efficiency'] = np.log1p(quantity) * features['large_part_efficiency']
            
            # 10. ADVANCED INTERACTION TERMS
            features['volume_surface_complexity'] = (volume * surface_area * (max_dim/min_dim)) ** (1/3)
            features['geometric_mean_dims'] = (max_dim * min_dim) ** 0.5
            features['harmonic_mean_efficiency'] = 2 / (1/volume + 1/surface_area)
            
            return features
            
        except Exception as e:
            print(f"❌ Error calculating non-linear features: {e}")
            return {}
    
    def predict_price(self, part_data):
        """Predict price using enhanced model with non-linear features"""
        try:
            if not self.model or not self.scaler:
                return None, "Enhanced model not loaded"
            
            # Calculate all non-linear features
            all_features = self.calculate_nonlinear_features(part_data)
            
            if not all_features:
                return None, "Failed to calculate features"
            
            # Create feature array with ALL 52 features in the order expected by the scaler
            # This must match the order used during training in nonlinear_feature_engineering.py
            feature_order = [
                # Base features (10)
                'shrinkwrap_volume', 'convex_hull_volume', 'Volume', 'Min D', 'Max D', 
                'D Ratio', 'waste', 'surface_area', 'bb_volume', 'Quantity',
                
                # Size-dependent features (7)
                'size_factor', 'micro_part_penalty', 'large_part_efficiency', 
                'size_category_micro', 'size_category_small', 'size_category_medium', 'size_category_large',
                
                # Logarithmic transformations (5)
                'log_volume', 'log_surface_area', 'log_shrinkwrap_volume', 'log_convex_hull_volume', 'log_bb_volume',
                
                # Power transformations (4)
                'volume_sqrt', 'volume_cbrt', 'surface_area_sqrt', 'shrinkwrap_sqrt',
                
                # Complexity-size interactions (5)
                'complexity_size_ratio', 'dimension_instability', 'waste_complexity', 
                'surface_volume_interaction', 'complexity_volume_interaction',
                
                # Economic behavior features (4)
                'fixed_cost_impact', 'handling_complexity', 'material_waste_curve', 'support_material_factor',
                
                # Manufacturing efficiency features (4)
                'scale_efficiency', 'batch_efficiency', 'manufacturing_challenge', 'precision_requirement',
                
                # Material utilization features (4)
                'material_density', 'packing_efficiency', 'shape_complexity', 'convexity_efficiency',
                
                # Dimensional analysis features (3)
                'aspect_ratio_penalty', 'dimensional_balance', 'slenderness_ratio',
                
                # Quantity-dependent features (3)
                'quantity_volume_interaction', 'quantity_complexity_interaction', 'quantity_efficiency',
                
                # Advanced interaction terms (3)
                'volume_surface_complexity', 'geometric_mean_dims', 'harmonic_mean_efficiency'
            ]
            
            # Create feature array with all 52 features
            feature_array = []
            missing_features = []
            
            for feature_name in feature_order:
                if feature_name in all_features:
                    feature_array.append(all_features[feature_name])
                else:
                    feature_array.append(0)
                    missing_features.append(feature_name)
            
            if missing_features:
                print(f"⚠️ Missing features: {missing_features}")
            
            print(f"🔧 Created feature array with {len(feature_array)} features (expected: 52)")
            
            # Scale features and predict
            feature_array = np.array(feature_array).reshape(1, -1)
            feature_array_scaled = self.scaler.transform(feature_array)
            raw_prediction = self.model.predict(feature_array_scaled)[0]
            
            # No minimum price threshold - allow natural model predictions
            final_price = raw_prediction
            
            # Create detailed breakdown using only the selected features for display
            breakdown = {
                'model_type': 'Enhanced Lasso with Non-Linear Features',
                'raw_prediction': raw_prediction,
                'final_price': final_price,
                'feature_count': len(self.feature_names),
                'top_features': {},
                'confidence': 'High' if abs(raw_prediction) > 1.0 else 'Medium'
            }
            
            # Calculate feature contributions for the selected features only
            for i, (feature_name, coef) in enumerate(self.feature_coefficients.items()):
                if feature_name in all_features:
                    # Find the index of this feature in the full feature array
                    try:
                        feature_idx = feature_order.index(feature_name)
                        if feature_idx < len(feature_array_scaled[0]):
                            scaled_value = feature_array_scaled[0][feature_idx]
                            contribution = coef * scaled_value
                            breakdown['top_features'][feature_name] = {
                                'value': all_features[feature_name],
                                'scaled_value': scaled_value,
                                'coefficient': coef,
                                'contribution': contribution
                            }
                    except ValueError:
                        # Feature not in feature_order, skip
                        pass
            
            return final_price, breakdown
            
        except Exception as e:
            print(f"❌ Error predicting price: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Prediction error: {e}"
    
    def get_model_info(self):
        """Get information about the enhanced model"""
        if not self.model_metadata:
            return "Enhanced model not loaded"
        
        perf = self.model_metadata['performance']
        info = f"Enhanced Lasso Model | R² = {perf['test_r2']:.3f} | MAE = €{perf['test_mae']:.2f} | Features: {len(self.feature_names)}"
        return info
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if not self.feature_coefficients:
            return []
        
        # Sort by absolute coefficient value
        sorted_features = sorted(
            self.feature_coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features[:top_n]

# Global instance for GUI integration
enhanced_calculator = EnhancedPricingCalculator()

def initialize_enhanced_pricing():
    """Initialize the enhanced pricing calculator"""
    return enhanced_calculator.load_enhanced_model()

def calculate_enhanced_price(part_data):
    """Calculate price using enhanced model"""
    return enhanced_calculator.predict_price(part_data)

def get_enhanced_model_info():
    """Get enhanced model information"""
    return enhanced_calculator.get_model_info()

def get_enhanced_feature_importance(top_n=10):
    """Get enhanced model feature importance"""
    if enhanced_calculator and enhanced_calculator.model:
        return enhanced_calculator.get_feature_importance(top_n)
    return []

def get_enhanced_feature_breakdown(part_data):
    """Get detailed feature breakdown for a part showing impact on pricing"""
    try:
        if not enhanced_calculator or not enhanced_calculator.model:
            return {"error": "Enhanced model not loaded"}
        
        # Calculate all features
        all_features = enhanced_calculator.calculate_nonlinear_features(part_data)
        
        if not all_features:
            return {"error": "Failed to calculate features"}
        
        # Get prediction
        price, error = enhanced_calculator.predict_price(part_data)
        
        if price is None:
            return {"error": f"Prediction failed: {error}"}
        
        # Get feature importance
        feature_importance = enhanced_calculator.get_feature_importance(20)  # Top 20 features
        
        # Create breakdown showing how each important feature affects price
        breakdown = {
            "predicted_price": price,
            "total_features": len(all_features),
            "feature_values": {},
            "feature_impacts": {},
            "model_info": enhanced_calculator.get_model_info()
        }
        
        # Add top features with their values and impacts
        for feature_name, importance in feature_importance:
            if feature_name in all_features:
                feature_value = all_features[feature_name]
                coefficient = enhanced_calculator.feature_coefficients.get(feature_name, 0)
                
                breakdown["feature_values"][feature_name] = feature_value
                breakdown["feature_impacts"][feature_name] = {
                    "importance": importance,
                    "coefficient": coefficient,
                    "value": feature_value,
                    "contribution": coefficient * feature_value if coefficient else 0
                }
        
        return breakdown
        
    except Exception as e:
        return {"error": f"Feature breakdown calculation failed: {str(e)}"}

def debug_enhanced_pricing(part_data):
    """Debug function to understand why enhanced pricing is failing"""
    try:
        print(f"🔍 DEBUG: Enhanced pricing debug for part:")
        print(f"   Calculator exists: {enhanced_calculator is not None}")
        print(f"   Model loaded: {enhanced_calculator.model is not None if enhanced_calculator else False}")
        print(f"   Scaler loaded: {enhanced_calculator.scaler is not None if enhanced_calculator else False}")
        
        if enhanced_calculator and enhanced_calculator.model:
            # Try to calculate features
            features = enhanced_calculator.calculate_nonlinear_features(part_data)
            print(f"   Features calculated: {len(features) if features else 0}")
            
            if features:
                print(f"   Sample features: {list(features.keys())[:5]}")
                
                # Try prediction
                price, error = enhanced_calculator.predict_price(part_data)
                print(f"   Prediction result: price={price}, error={error}")
                
                return price, error
        
        return None, "Debug: No model or calculator available"
        
    except Exception as e:
        print(f"❌ DEBUG ERROR: {e}")
        return None, f"Debug error: {str(e)}" 