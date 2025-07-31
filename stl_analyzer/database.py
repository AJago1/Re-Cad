import pandas as pd
import os
import json
from pathlib import Path

class STLDatabase:
    """Database for storing STL model data"""
    
    def __init__(self, db_path=None):
        """Initialize database connection"""
        if db_path is None:
            # Default to home directory
            db_path = os.path.join(str(Path.home()), "stl_analyzer", "stl_database.csv")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.df = None
        self.load_database()
        
    def load_database(self):
        """Load database from file"""
        try:
            if os.path.exists(self.db_path):
                self.df = pd.read_csv(self.db_path)
                # Calculate missing fields if needed
                self._normalize_data()
            else:
                # Create empty dataframe with required columns
                self.df = pd.DataFrame(columns=[
                    'name', 'filename', 'x', 'y', 'z', 'volume', 
                    'surface_area', 'bb_volume', 'waste',
                    'convex_hull_volume', 'convexity_ratio',
                    'shrinkwrap_volume', 'shrinkwrap_ratio',
                    'optimal_transform'
                ])
        except Exception as e:
            print(f"Error loading database: {e}")
            # Create a new DataFrame if the file is corrupted
            self.df = pd.DataFrame(columns=[
                'name', 'filename', 'x', 'y', 'z', 'volume', 
                'surface_area', 'bb_volume', 'waste',
                'convex_hull_volume', 'convexity_ratio',
                'shrinkwrap_volume', 'shrinkwrap_ratio',
                'optimal_transform'
            ])
    
    def _normalize_data(self):
        """Normalize data and calculate any missing derived fields"""
        # Normalize path separators for cross-platform compatibility
        if 'filename' in self.df.columns:
            self.df['filename'] = self.df['filename'].apply(lambda x: os.path.normpath(str(x)) if pd.notna(x) else x)
            
        # Calculate waste if it doesn't exist but we have volume and bb_volume
        if 'waste' not in self.df.columns and 'volume' in self.df.columns and 'bb_volume' in self.df.columns:
            self.df['waste'] = self.df.apply(
                lambda row: (row['bb_volume'] - row['volume']) / row['bb_volume'] * 100 
                if pd.notna(row['bb_volume']) and row['bb_volume'] > 0 else 0,
                axis=1
            )
            
        # Ensure convex hull fields exist
        if 'convex_hull_volume' not in self.df.columns:
            self.df['convex_hull_volume'] = 0.0
            
        if 'convexity_ratio' not in self.df.columns:
            self.df['convexity_ratio'] = self.df.apply(
                lambda row: row['volume'] / row['convex_hull_volume'] 
                if pd.notna(row['convex_hull_volume']) and row['convex_hull_volume'] > 0 else 0.0,
                axis=1
            )
            
        # Ensure shrinkwrap fields exist
        if 'shrinkwrap_volume' not in self.df.columns:
            self.df['shrinkwrap_volume'] = 0.0
            
        if 'shrinkwrap_ratio' not in self.df.columns:
            self.df['shrinkwrap_ratio'] = self.df.apply(
                lambda row: row['volume'] / row['shrinkwrap_volume'] 
                if pd.notna(row['shrinkwrap_volume']) and row['shrinkwrap_volume'] > 0 else 0.0,
                axis=1
            )
            
        # Add new shrinkwrap STL export fields
        if 'shrinkwrap_stl_path' not in self.df.columns:
            self.df['shrinkwrap_stl_path'] = None
            
        if 'shrinkwrap_offset_percent' not in self.df.columns:
            self.df['shrinkwrap_offset_percent'] = 5.0  # Default 5% offset
            
        if 'shrinkwrap_export_date' not in self.df.columns:
            self.df['shrinkwrap_export_date'] = None
            
        # Normalize shrinkwrap STL paths for cross-platform compatibility
        if 'shrinkwrap_stl_path' in self.df.columns:
            self.df['shrinkwrap_stl_path'] = self.df['shrinkwrap_stl_path'].apply(
                lambda x: os.path.normpath(str(x)) if pd.notna(x) and x != '' else None
            )
            
        # Handle optimal transform (stored as JSON string)
        if 'optimal_transform' in self.df.columns:
            # Make sure it's stored as a string
            self.df['optimal_transform'] = self.df['optimal_transform'].apply(
                lambda x: json.dumps(x) if not isinstance(x, str) and pd.notna(x) else x
            )
    
    def save_database(self):
        """Save database to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Save to CSV
            self.df.to_csv(self.db_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_entry(self, data):
        """Add a new entry to the database"""
        if not data:
            return False
            
        # Convert transformation matrix to JSON if needed
        if 'optimal_transform' in data and data['optimal_transform'] is not None:
            if not isinstance(data['optimal_transform'], str):
                data['optimal_transform'] = json.dumps(data['optimal_transform'])
        
        # Check if entry already exists
        if 'filename' in data and 'filename' in self.df.columns:
            # Normalize path for comparison
            filename = os.path.normpath(data['filename'])
            
            # Remove existing entry
            self.df = self.df[self.df['filename'] != filename]
        
        # Add entry to dataframe
        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        
        # Normalize data
        self._normalize_data()
        return True
    
    def get_all_entries(self):
        """Get all entries from the database"""
        return self.df.copy()
    
    def get_entry(self, filename):
        """Get a specific entry by filename"""
        if 'filename' not in self.df.columns:
            return None
        
        # Normalize path for comparison
        filename = os.path.normpath(filename)
        
        # Find the entry
        entry = self.df[self.df['filename'] == filename]
        if entry.empty:
            return None
        
        # Convert to dictionary
        return entry.iloc[0].to_dict()
    
    def remove_entry(self, filename):
        """Remove an entry from the database"""
        if 'filename' not in self.df.columns:
            return False
        
        # Normalize path for comparison
        filename = os.path.normpath(filename)
        
        # Check if entry exists
        if not self.df[self.df['filename'] == filename].empty:
            # Remove entry
            self.df = self.df[self.df['filename'] != filename]
            return True
        
        return False
    
    def clear(self):
        """Clear all data from the database"""
        # Create empty dataframe with required columns
        self.df = pd.DataFrame(columns=[
            'name', 'filename', 'x', 'y', 'z', 'volume', 
            'surface_area', 'bb_volume', 'waste',
            'convex_hull_volume', 'convexity_ratio',
            'shrinkwrap_volume', 'shrinkwrap_ratio',
            'optimal_transform'
        ])
        return True

    def _normalize_path(self, path):
        """Normalize file path to handle Windows backslashes and other path issues"""
        if path:
            return os.path.normpath(path)
        return path

    def add_multiple_entries(self, entries_list):
        """Add multiple STL entries to the database at once"""
        if not entries_list:
            return 0
            
        success_count = 0
        for entry in entries_list:
            if self.add_entry(entry):
                success_count += 1
                
        return success_count
    
    def get_entry_count(self):
        """Return the number of entries in the database"""
        return len(self.df)
    
    def export_to_json(self, json_path="stl_library.json"):
        """Export the database to JSON format"""
        try:
            self.df.to_json(json_path, orient="records")
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
            
    def import_from_json(self, json_path="stl_library.json"):
        """Import database from JSON format"""
        try:
            if os.path.exists(json_path):
                imported_df = pd.read_json(json_path, orient="records")
                self.df = pd.concat([self.df, imported_df], ignore_index=True)
                return True
            return False
        except Exception as e:
            print(f"Error importing from JSON: {e}")
            return False

    def get_file_data(self, filename):
        """Get data for a specific file from the database"""
        # Normalize the path for comparison
        filename = self._normalize_path(filename)
        
        # Find matching entries
        for idx, row in self.df.iterrows():
            if self._normalize_path(row["filename"]) == filename:
                # Convert row to dictionary
                data = row.to_dict()
                
                # Round numeric values
                for key, value in data.items():
                    if isinstance(value, float):
                        data[key] = round(value, 2)
                
                return data
        
        return None
        
    def update_file_data(self, filename, update_dict):
        """Update specific fields for a file in the database"""
        # Normalize the path for comparison
        filename = self._normalize_path(filename)
        
        # Find and update the matching entry
        for idx, row in self.df.iterrows():
            if self._normalize_path(row["filename"]) == filename:
                # Update the values
                for key, value in update_dict.items():
                    self.df.at[idx, key] = value
                return True
        
        return False 