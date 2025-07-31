# STL Analyzer Rotation Optimization Fixes

## Issues Identified and Fixed

### **Problem Summary**
When pressing "Optimize", the model would:
1. Drop to the end of the STL database
2. Require multiple presses to work
3. Update info in database and model details BUT not update the actual position
4. When picked from database again, the model wasn't actually optimized (info was but position wasn't)

## **Root Causes Found:**

### **1. Database Entry Duplication**
- `on_optimization_complete()` was **adding new entries** instead of updating existing ones
- This caused models to "drop to the end" of the database
- **Fix**: Changed to use `update_entry()` instead of `add_entry()`

### **2. Stored Transform Not Being Used**
- Database stored `optimal_transform` but `on_db_table_clicked()` ignored it
- When loading from database, it would recalculate orientation instead of using stored one
- **Fix**: Modified `on_db_table_clicked()` to extract and use stored `optimal_transform`

### **3. Viewer Not Respecting Provided Transform**
- `load_stl()` function always calculated new optimal orientation even when transform was provided
- **Fix**: Modified `load_stl()` to check if `optimal_transform` parameter is provided and use it directly

### **4. Missing Method in Main Directory**
- Main directory viewer was missing `orient_to_minimal_bounding_box()` method
- **Fix**: Added the missing method to ensure consistency

## **Files Modified:**

### **1. `gui.py` (Both directories)**
```python
# Fixed on_optimization_complete() to update instead of add
def on_optimization_complete(self, file_path, optimized_features):
    # Find and UPDATE existing entry instead of adding new one
    existing_entry = self.stl_db.get_entry(normalized_path)
    if existing_entry:
        # Update the existing entry
        for key, value in optimized_features.items():
            existing_entry[key] = value
        success = self.stl_db.update_entry(existing_entry)
    else:
        # Only add if truly new
        success = self.stl_db.add_entry(optimized_features)

# Fixed on_db_table_clicked() to use stored transforms
def on_db_table_clicked(self, index):
    # Extract optimal transform if available
    optimal_transform = None
    if file_data and 'optimal_transform' in file_data:
        # Convert stored transform to numpy array
        optimal_transform = np.array(transform_data)
    
    # Load with stored transform
    success = self.stl_viewer.load_stl(filename, optimal_transform)
```

### **2. `viewer.py` (Both directories)**
```python
# Fixed load_stl() to respect provided transforms
def load_stl(self, filepath, optimal_transform=None):
    if optimal_transform is not None:
        print(f"Applying provided optimal transform")
        # Apply the provided transformation
        mesh.apply_transform(optimal_transform)
        self.mesh = mesh
        self.optimal_transform = optimal_transform
        self.update_display()
    else:
        # Only calculate if no transform provided
        self.mesh = mesh
        self.orient_to_minimal_bounding_box()
```

### **3. `stl_utils.py` (Both directories)**
```python
# Fixed extract_features_from_mesh() to return consistent field names
def extract_features_from_mesh(mesh, file_path, model_name=None):
    # Ensure consistent field names
    features = {
        'filename': file_path,  # Always use full path
        'name': model_name,     # Consistent naming
        'volume': volume,
        'surface_area': surface_area,
        # ... all other fields with consistent names
    }
```

## **How It Works Now:**

### **Optimization Process:**
1. User clicks "Optimize" → `optimize_orientation()` runs
2. Calculates optimal transform and applies it to mesh
3. Extracts features from optimized mesh (including transform)
4. Calls `on_optimization_complete()` with optimized features
5. **Updates existing database entry** (doesn't create new one)
6. Refreshes database view **maintaining current position**
7. Updates model details view with new data

### **Loading from Database:**
1. User clicks on database entry → `on_db_table_clicked()` runs
2. Gets file data from database including `optimal_transform`
3. **Uses stored transform** when calling `load_stl()`
4. Viewer applies stored transform directly (no recalculation)
5. Model appears in optimized position immediately

## **Result:**
✅ Model stays in same database position after optimization  
✅ Optimization works on first press  
✅ Both database info AND actual model position are updated  
✅ Loading from database shows the actual optimized model position  
✅ Consistent behavior between embedded and main versions 