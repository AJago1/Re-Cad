#!/usr/bin/env python3
"""
Add Model Library to STL Analyzer GUI
====================================

This script adds model library functionality to the STL Analyzer.
It scans for available models and allows switching between them.
"""

# Add this code to your gui.py file after the existing methods

MODEL_LIBRARY_CODE = '''

    def initialize_model_library(self):
        """Initialize the model library system"""
        if hasattr(self, 'pricing_model_combo'):
            self.scan_available_models()

    def scan_available_models(self):
        """Scan for available pricing models and populate the library"""
        print("📚 Scanning for available pricing models...")
        
        self.available_models = {}
        
        # Define model patterns to look for
        model_patterns = [
            {
                'files': ['high_performance_random_forest_model.pkl', 'high_performance_random_forest_metadata.json'],
                'name': '🚀 High Performance RF',
                'type': 'RandomForest',
                'id': 'high_performance'
            },
            {
                'files': ['random_forest_project_aware_with_machine_color.pkl', 'random_forest_project_aware_with_machine_color_scaler.pkl', 'random_forest_project_aware_with_machine_color_metadata.json'],
                'name': '🌲 RF v2.0 (Machine/Color)',
                'type': 'RandomForest',
                'id': 'rf_machine_color'
            },
            {
                'files': ['improved_random_forest_model.pkl', 'improved_random_forest_metadata.json'],
                'name': '📈 Improved RF Model',
                'type': 'RandomForest', 
                'id': 'improved_rf'
            },
            {
                'files': ['random_forest_project_aware_model.pkl', 'random_forest_project_aware_scaler.pkl', 'random_forest_project_aware_metadata.json'],
                'name': '🎯 Project-Aware RF',
                'type': 'RandomForest',
                'id': 'rf_project_aware'
            },
            {
                'files': ['enhanced_lasso_model.pkl', 'enhanced_feature_scaler.pkl', 'enhanced_model_metadata.json'],
                'name': '🎯 Enhanced Lasso',
                'type': 'Lasso',
                'id': 'enhanced_lasso'
            }
        ]
        
        # Check which models are available
        for pattern in model_patterns:
            if all(os.path.exists(f) for f in pattern['files']):
                try:
                    # Load metadata to get performance info
                    metadata_file = next(f for f in pattern['files'] if f.endswith('_metadata.json'))
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract performance metrics
                    performance = metadata.get('performance', {})
                    r2 = performance.get('r2', metadata.get('r2', 0))
                    mae = performance.get('mae', metadata.get('mae', 0))
                    
                    self.available_models[pattern['id']] = {
                        'name': pattern['name'],
                        'type': pattern['type'],
                        'files': pattern['files'],
                        'metadata': metadata,
                        'r2': r2,
                        'mae': mae,
                        'features': len(metadata.get('features', metadata.get('feature_names', []))),
                        'display_name': f"{pattern['name']} (R²={r2:.3f}, MAE=€{mae:.2f})"
                    }
                    print(f"   ✅ Found: {pattern['name']}")
                    
                except Exception as e:
                    print(f"   ❌ Error loading {pattern['name']}: {e}")
        
        # Update the combo box
        if hasattr(self, 'pricing_model_combo'):
            self.pricing_model_combo.clear()
            if self.available_models:
                # Sort by R² score (best first)
                sorted_models = sorted(self.available_models.items(), key=lambda x: x[1]['r2'], reverse=True)
                for model_id, model_info in sorted_models:
                    self.pricing_model_combo.addItem(model_info['display_name'], model_id)
                
                print(f"📚 Found {len(self.available_models)} available models")
                
                # Auto-select the best performing model
                self.pricing_model_combo.setCurrentIndex(0)
                self.update_model_performance_display_quick()
            else:
                self.pricing_model_combo.addItem("No models found")
                print("⚠️ No pricing models found")

    def on_model_library_changed(self, display_name):
        """Handle model selection change from library"""
        if not hasattr(self, 'pricing_model_combo'):
            return
            
        model_id = self.pricing_model_combo.currentData()
        if not model_id or not hasattr(self, 'available_models') or model_id not in self.available_models:
            return
        
        model_info = self.available_models[model_id]
        print(f"🔄 Switching to model: {model_info['name']}")
        
        try:
            # Simple model loading - reuse existing initialization
            if model_id == 'high_performance':
                # This will load the high performance model
                self.initialize_ai_pricing_model()
            else:
                # For other models, we'd need more complex loading
                print(f"   Model switching for {model_id} not yet implemented")
                self.show_status_message(f"Model switching for {model_info['name']} coming soon!")
                return
            
            # Update performance display
            self.update_model_performance_display_quick()
            
            # Recalculate prices if parts are loaded
            if hasattr(self, 'project_parts') and self.project_parts:
                print("🔄 Recalculating prices with new model...")
                self.calculate_all_prices()
            
            self.show_status_message(f"✅ Switched to {model_info['name']}")
            
        except Exception as e:
            print(f"❌ Error switching to model {model_info['name']}: {e}")
            self.show_status_message(f"❌ Failed to load {model_info['name']}")

    def update_model_performance_display_quick(self):
        """Update the model performance display"""
        if not hasattr(self, 'model_performance_label') or not hasattr(self, 'pricing_model_combo'):
            return
            
        model_id = self.pricing_model_combo.currentData()
        if model_id and hasattr(self, 'available_models') and model_id in self.available_models:
            model_info = self.available_models[model_id]
            perf_text = (f"📊 {model_info['type']} | "
                        f"R² = {model_info['r2']:.3f} | "
                        f"MAE = €{model_info['mae']:.2f} | "
                        f"Features: {model_info['features']}")
            
            # Add machine/color support indicator
            if 'machine' in model_info['name'].lower():
                perf_text += " | 🏭🎨 M/C Support"
            
            self.model_performance_label.setText(perf_text)

    def show_model_info_dialog(self):
        """Show detailed model information dialog"""
        if not hasattr(self, 'pricing_model_combo'):
            return
            
        model_id = self.pricing_model_combo.currentData()
        if not model_id or not hasattr(self, 'available_models') or model_id not in self.available_models:
            QtWidgets.QMessageBox.information(self, "No Model", "No model selected or available.")
            return
        
        model_info = self.available_models[model_id]
        metadata = model_info['metadata']
        
        # Show simplified info for now
        info_text = f"""
Model: {model_info['name']}
Type: {model_info['type']}
R² Score: {model_info['r2']:.4f}
MAE: €{model_info['mae']:.2f}
Features: {model_info['features']}

Files:
{chr(10).join('• ' + f for f in model_info['files'])}
"""
        
        QtWidgets.QMessageBox.information(self, f"Model Info: {model_info['name']}", info_text)

'''

def add_to_gui():
    """Instructions for adding to GUI"""
    print("🔧 Model Library Code Generated!")
    print("📝 To add to your GUI:")
    print("1. Open stl_analyzer/gui.py")
    print("2. Add the methods above after the 'update_model_info_display' method")
    print("3. Add this line after 'self.initialize_ai_pricing_model()' in __init__:")
    print("   QtCore.QTimer.singleShot(100, self.initialize_model_library)")
    print("4. The GUI will automatically scan and populate available models!")

if __name__ == "__main__":
    print(MODEL_LIBRARY_CODE)
    add_to_gui() 