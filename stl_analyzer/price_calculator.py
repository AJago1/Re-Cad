"""
Price Calculator module for STL Analyzer.
Contains functions for calculating prices based on STL geometry.
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime

# Coefficients for the linear model pricing
LINEAR_MODEL_COEFFICIENTS = {
    "volume": 0.0002,                  # Volume in mm³ 
    "surface_area": 0.0015,            # Surface area in mm²
    "convex_hull_volume": -0.00008,    # Convex hull volume in mm³
    "bb_volume": 0.00005,              # Bounding box volume in mm³
    "bb_max_dim": 0.08,                # Maximum dimension in mm
    "convexity_ratio": -12.0,          # Volume/Convex Hull Volume ratio (0-1)
    "bb_dim_ratio": 0.5,               # Max dimension / Min dimension ratio
    "surface_to_volume": 0.2,          # Surface area / Volume ratio
    "volume_to_bb": -8.0,              # Volume / Bounding Box Volume ratio (0-1)
    "intercept": 12.5                  # Base price
}

def calculate_part_price(features, config, quantity=1):
    """Calculate price for a single part based on its features and configuration"""
    # Extract part features
    volume_mm3 = features["volume"]
    volume_cm3 = volume_mm3 / 1000  # Convert mm³ to cm³
    surface_area = features["surface_area"]
    convex_hull_volume = features["convex_hull_volume"]
    
    # Extract configuration parameters
    material_density = config.get("material_density", 1.05)  # g/cm³
    material_price = config.get("material_price", 85.0)  # EUR/kg
    material_reuse_ratio = config.get("material_reuse_ratio", 0.5)  # 50% reuse
    
    # Calculate material weight and cost
    material_weight_g = volume_cm3 * material_density
    material_cost = (material_weight_g / 1000) * material_price  # Convert g to kg
    
    # Calculate printing time based on build speed (mm/h)
    build_speed = config.get("build_speed", 15.0)  # mm/h
    
    # Calculate part height along Z axis
    if "bb_dims" in features:
        part_height = features["bb_dims"][2]  # Z dimension
    else:
        part_height = features["bb_max_dim"]  # Use max dimension as fallback
    
    # Calculate base build time in hours
    base_build_time = part_height / build_speed
    
    # Calculate complexity factor
    convexity_ratio = features.get("convexity_ratio", 
                                  volume_mm3 / convex_hull_volume 
                                  if convex_hull_volume > 0 else 1.0)
    
    complexity_weight = config.get("complexity_weight", 0.35)
    complexity_factor = 1 + (1 - convexity_ratio) * complexity_weight
    
    # Apply complexity factor to build time
    adjusted_build_time = base_build_time * complexity_factor
    
    # Additional machine time
    heating_time = config.get("heating_time", 2.5)  # hours
    cooling_time = config.get("cooling_time", 3.0)  # hours
    
    # Total machine time
    machine_time = adjusted_build_time + heating_time + cooling_time
    
    # Calculate machine cost
    machine_investment = config.get("machine_investment", 200000.0)  # EUR
    machine_amort_years = config.get("machine_amort_years", 5)  # years
    weekly_machine_hours = config.get("weekly_machine_hours", 70.0)  # hours/week
    working_weeks_per_year = config.get("working_weeks_per_year", 48)  # weeks/year
    
    # Calculate total available machine hours per year
    total_machine_hours_per_year = weekly_machine_hours * working_weeks_per_year
    
    # Calculate hourly machine cost
    hourly_machine_cost = machine_investment / (machine_amort_years * total_machine_hours_per_year)
    
    # Machine cost for this part
    machine_cost = hourly_machine_cost * machine_time
    
    # Energy costs
    energy_usage_per_hour = config.get("energy_usage_per_hour", 5.0)  # kWh
    energy_cost_per_kwh = config.get("energy_cost", 0.15)  # EUR/kWh
    energy_cost = machine_time * energy_usage_per_hour * energy_cost_per_kwh
    
    # Labor costs
    labor_cost_per_hour = config.get("labor_cost", 25.0)  # EUR/h
    setup_time = config.get("setup_time", 0.5)  # hours
    monitoring_time = config.get("monitoring_time", 0.1) * adjusted_build_time  # × build time
    post_processing_time = config.get("post_processing_time", 0.5)  # hours
    packaging_time = config.get("packaging_time", 0.2)  # hours
    
    total_labor_time = setup_time + monitoring_time + post_processing_time + packaging_time
    labor_cost = total_labor_time * labor_cost_per_hour
    
    # Maintenance buffer
    maintenance_buffer = config.get("maintenance_buffer", 1.0)  # EUR
    
    # Calculate base total
    base_total = material_cost + machine_cost + energy_cost + labor_cost + maintenance_buffer
    
    # Apply margin
    margin_percentage = config.get("margin", 15.0) / 100.0
    final_price = base_total * (1 + margin_percentage)
    
    # Create result dictionary
    result = {
        "base_price": final_price,
        "final_price": final_price * quantity,
        "breakdown": {
            "material_cost": material_cost,
            "part_weight_g": material_weight_g,
            "material_total_use_g": material_weight_g / material_reuse_ratio,
            "material_new_powder_g": material_weight_g * (1 - material_reuse_ratio),
            "powder_volume_cm3": volume_cm3,
            "base_build_time": base_build_time,
            "complexity_factor": complexity_factor,
            "build_time": adjusted_build_time,
            "machine_time": machine_time,
            "total_time": machine_time * quantity,
            "machine_cost": machine_cost,
            "energy_cost": energy_cost,
            "labor_cost": labor_cost,
            "maintenance_cost": maintenance_buffer,
            "total_cost": base_total,
            "complexity_factor": complexity_factor
        }
    }
    
    return result

def calculate_linear_model_price(features, scaling_factor=1.0):
    """Calculate price using the linear model coefficients
    
    Args:
        features (dict): Part features extracted from STL
        scaling_factor (float): Scaling factor to adjust final price
        
    Returns:
        dict: Results including price and breakdown
    """
    # Extract required features
    surface_area = features["surface_area"]
    volume = features["volume"]
    convex_hull_volume = features["convex_hull_volume"]
    bb_volume = features["bb_volume"]
    
    # Get dimensions
    if "bb_dims" in features:
        dimensions = features["bb_dims"]
    else:
        dimensions = [features.get("bb_max_dim", 0)] * 3
    
    max_d = max(dimensions)
    min_d = min(d for d in dimensions if d > 0) if any(d > 0 for d in dimensions) else 1
    avg_d = sum(dimensions) / 3 if all(d > 0 for d in dimensions) else max_d / 2
    
    # Calculate derived features used in the model
    derived_features = {
        "volume": volume,
        "surface_area": surface_area,
        "convex_hull_volume": convex_hull_volume,
        "bb_volume": bb_volume,
        "bb_max_dim": max_d,
        "convexity_ratio": volume / convex_hull_volume if convex_hull_volume > 0 else 1.0,
        "bb_dim_ratio": max_d / min_d if min_d > 0 else 1.0,
        "surface_to_volume": surface_area / volume if volume > 0 else 1.0,
        "volume_to_bb": volume / bb_volume if bb_volume > 0 else 1.0
    }
    
    # Start with the intercept (base price)
    price = LINEAR_MODEL_COEFFICIENTS.get("intercept", 0)
    price_breakdown = {}
    
    # Add the intercept to the breakdown
    if "intercept" in LINEAR_MODEL_COEFFICIENTS:
        price_breakdown["intercept"] = {
            "value": 1.0,
            "coefficient": LINEAR_MODEL_COEFFICIENTS["intercept"],
            "price_impact": LINEAR_MODEL_COEFFICIENTS["intercept"]
        }
    
    # Calculate price contributions from each feature
    for feature_name, coefficient in LINEAR_MODEL_COEFFICIENTS.items():
        if feature_name != "intercept" and feature_name in derived_features:
            feature_value = derived_features[feature_name]
            price_impact = coefficient * feature_value
            price += price_impact
            
            # Store in breakdown
            price_breakdown[feature_name] = {
                "value": feature_value,
                "coefficient": coefficient,
                "price_impact": price_impact
            }
    
    # Apply scaling factor
    final_price = max(0, price * scaling_factor)  # Ensure price is never negative
    
    # Create result dictionary
    result = {
        "price": final_price,
        "scaling_factor": scaling_factor,
        "breakdown": price_breakdown
    }
    
    return result

def calculate_hybrid_price(manufacturing_price, linear_price, linear_weight=0.5):
    """Calculate price using a hybrid of manufacturing-based and linear models
    
    Args:
        manufacturing_price (float): Price from manufacturing-based model
        linear_price (float): Price from linear model
        linear_weight (float): Weight of linear model (0-1)
        
    Returns:
        dict: Results including hybrid price and breakdown
    """
    # Calculate weighted price
    hybrid_price = (linear_weight * linear_price) + ((1 - linear_weight) * manufacturing_price)
    
    # Create result dictionary
    result = {
        "price": hybrid_price,
        "manufacturing_contribution": (1 - linear_weight) * manufacturing_price,
        "linear_contribution": linear_weight * linear_price,
        "manufacturing_weight": 1 - linear_weight,
        "linear_weight": linear_weight
    }
    
    return result

def save_price_calculation_to_db(file_path, features, breakdown, config, quantity, db_path=None):
    """Save price calculation to the database"""
    try:
        # Determine the path for the price database
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_calculations.db")
        
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_calculations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            part_name TEXT,
            volume REAL,
            surface_area REAL,
            convex_hull_volume REAL,
            bb_volume REAL,
            max_dimension REAL,
            quantity INTEGER,
            material TEXT,
            material_price REAL,
            material_density REAL,
            machine_cost REAL,
            energy_cost REAL,
            print_speed REAL,
            complexity_weight REAL,
            labor_cost REAL,
            maintenance_buffer REAL,
            margin REAL,
            material_cost REAL,
            estimated_time REAL,
            machine_cost_calculated REAL,
            energy_cost_calculated REAL,
            base_total REAL,
            final_price REAL,
            linear_model_price REAL,
            hybrid_model_price REAL,
            linear_weight REAL,
            config_json TEXT,
            breakdown_json TEXT
        )
        ''')
        
        # JSON serialize config and breakdown
        config_json = json.dumps(config)
        breakdown_json = json.dumps(breakdown)
        
        # Extract linear model price if available
        linear_model_price = 0
        hybrid_model_price = 0
        linear_weight = 0
        
        if "linear_model" in breakdown:
            linear_model_price = breakdown["linear_model"]["price"]
            hybrid_model_price = breakdown.get("hybrid_price", 0)
            linear_weight = breakdown["linear_model"]["weight"]
        
        # Insert data
        cursor.execute('''
        INSERT INTO price_calculations (
            timestamp, filename, part_name, volume, surface_area, 
            convex_hull_volume, bb_volume, max_dimension, quantity,
            material, material_price, material_density, machine_cost,
            energy_cost, print_speed, complexity_weight, labor_cost,
            maintenance_buffer, margin, material_cost, estimated_time,
            machine_cost_calculated, energy_cost_calculated, base_total,
            final_price, linear_model_price, hybrid_model_price, linear_weight,
            config_json, breakdown_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            file_path,
            os.path.basename(file_path),
            features["volume"],
            features["surface_area"],
            features["convex_hull_volume"],
            features["bb_volume"],
            features["x"],  # max dimension
            quantity,
            config["material"],
            config["material_price"],
            config["material_density"],
            config["machine_cost"],
            config["energy_cost"],
            config["print_speed"],
            config["complexity_weight"],
            config["labor_cost"],
            config["maintenance_buffer"],
            config["margin"],
            breakdown.get("manufacturing_model", {}).get("breakdown", {}).get("material_cost", 0),
            breakdown.get("manufacturing_model", {}).get("breakdown", {}).get("estimated_time", 0),
            breakdown.get("manufacturing_model", {}).get("breakdown", {}).get("machine_cost", 0),
            breakdown.get("manufacturing_model", {}).get("breakdown", {}).get("energy_cost", 0),
            breakdown.get("manufacturing_model", {}).get("breakdown", {}).get("base_total", 0),
            breakdown.get("manufacturing_model", {}).get("price", 0),
            linear_model_price,
            hybrid_model_price,
            linear_weight,
            config_json,
            breakdown_json
        ))
        
        # Commit and close connection
        conn.commit()
        conn.close()
        
        return True, "Price calculation saved to database"
        
    except Exception as e:
        return False, f"Error saving price calculation: {e}"

def get_default_pricing_config():
    """Get default pricing configuration"""
    return {
        "material": "PA12",
        "material_price": 0.85,
        "material_density": 1.05,
        "machine_cost": 5.50,
        "energy_cost": 0.15,
        "print_speed": 30.0,
        "complexity_weight": 0.35,
        "labor_cost": 5.0,
        "maintenance_buffer": 1.0,
        "margin": 15.0,
        "use_linear_model": False,
        "linear_weight": 0.5,
        "linear_scaling_factor": 1.0
    }

def load_pricing_config(file_path):
    """Load pricing configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return True, config
    except Exception as e:
        return False, f"Error loading configuration: {e}"

def save_pricing_config(config, file_path):
    """Save pricing configuration to JSON file"""
    try:
        # Make sure it has .json extension
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
            
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        return True, f"Saved configuration to {os.path.basename(file_path)}"
        
    except Exception as e:
        return False, f"Error saving configuration: {e}" 