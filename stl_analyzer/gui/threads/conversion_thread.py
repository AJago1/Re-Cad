"""
Thread for converting STEP files to STL format
"""

import os
import sys
import subprocess
from PyQt5.QtCore import QThread, pyqtSignal


class StepConversionThread(QThread):
    """Worker thread for converting STEP to STL"""
    progress = pyqtSignal(int)
    conversion_complete = pyqtSignal(bool, str)
    
    def __init__(self, input_file, output_file, resolution):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.resolution = resolution
        
    def run(self):
        """Convert STEP file to STL"""
        self.progress.emit(10)
        
        try:
            # Try using pythonOCC for direct conversion first
            try:
                self.progress.emit(20)
                
                # Import required pythonOCC modules
                try:
                    from OCC.Core.STEPControl import STEPControl_Reader
                    from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
                    from OCC.Core.StlAPI import StlAPI_Writer
                    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
                    from OCC.Core.TopoDS import TopoDS_Compound
                    from OCC.Core.BRep import BRep_Builder
                    from OCC.Core.gp import gp_Pnt
                    from OCC.Core.TopAbs import TopAbs_FACE
                    from OCC.Core.TopExp import TopExp_Explorer
                    
                    # Normalize file paths for consistency
                    input_path = os.path.normpath(self.input_file)
                    output_path = os.path.normpath(self.output_file)
                    
                    self.progress.emit(30)
                    
                    # Read the STEP file
                    step_reader = STEPControl_Reader()
                    status = step_reader.ReadFile(input_path)
                    
                    if status == IFSelect_RetDone:  # check status
                        step_reader.PrintCheckLoad(False, IFSelect_ItemsByEntity)
                        step_reader.PrintCheckTransfer(False, IFSelect_ItemsByEntity)
                        
                        self.progress.emit(50)
                        
                        # Transfer shapes
                        step_reader.TransferRoots()
                        shape = step_reader.OneShape()
                        
                        # Create a compound
                        aCompound = TopoDS_Compound()
                        aBuilder = BRep_Builder()
                        aBuilder.MakeCompound(aCompound)
                        
                        # Add all shapes to the compound
                        aBuilder.Add(aCompound, shape)
                        
                        # Mesh the shape with the specified resolution
                        # Higher resolution means smaller deflection
                        deflection = 0.1 / self.resolution
                        mesh = BRepMesh_IncrementalMesh(aCompound, deflection, False, deflection, True)
                        mesh.Perform()
                        
                        self.progress.emit(80)
                        
                        # Write STL file
                        stl_writer = StlAPI_Writer()
                        stl_writer.SetASCIIMode(False)  # Write binary STL
                        result = stl_writer.Write(aCompound, output_path)
                        
                        if result:
                            self.progress.emit(100)
                            self.conversion_complete.emit(True, self.output_file)
                            return
                        else:
                            raise Exception("Failed to write STL file")
                    else:
                        raise Exception("Failed to read STEP file")
                        
                except ImportError as e:
                    print(f"PythonOCC import error: {e}")
                    # Will try other methods
                    pass
                except Exception as e:
                    print(f"PythonOCC conversion error: {e}")
                    # Will try other methods
                    pass
            
            except Exception as e:
                print(f"Error during direct conversion attempt: {e}")
            
            # If direct conversion fails, fall back to external tools
            # Get common program paths based on platform
            if sys.platform == 'win32':
                freecad_paths = [
                    r"C:\Program Files\FreeCAD\bin\FreeCAD.exe",
                    r"C:\Program Files (x86)\FreeCAD\bin\FreeCAD.exe"
                ]
                openscad_paths = [
                    r"C:\Program Files\OpenSCAD\openscad.exe",
                    r"C:\Program Files (x86)\OpenSCAD\openscad.exe"
                ]
            elif sys.platform == 'darwin':  # macOS
                freecad_paths = [
                    "/Applications/FreeCAD.app/Contents/MacOS/FreeCAD",
                    "/Applications/FreeCAD.app/Contents/bin/FreeCAD"
                ]
                openscad_paths = [
                    "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
                ]
            else:  # Linux and other
                freecad_paths = ["/usr/bin/freecad", "/usr/local/bin/freecad"]
                openscad_paths = ["/usr/bin/openscad", "/usr/local/bin/openscad"]
            
            # Method 1: Try using FreeCAD if available
            freecad_found = False
            for freecad_path in freecad_paths:
                if os.path.exists(freecad_path):
                    freecad_found = True
                    try:
                        # Create a temporary Python script for FreeCAD
                        script_content = f"""
import FreeCAD
import Mesh
import Part
import sys

# Open the STEP file
doc = FreeCAD.newDocument("Converter")
Part.open("{self.input_file.replace('\\', '/')}", "Converter")
shape = doc.findObjects("Part::Feature")[0].Shape

# Set mesh parameters based on resolution (1-10)
# Higher resolution = more triangles
params = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Mesh/Mesh")
params.SetFloat("MeshDeviation", 0.1 / {self.resolution})

# Convert to mesh
mesh = Mesh.Mesh()
mesh.addFacets(shape.tessellate({self.resolution}))

# Export to STL
mesh.write("{self.output_file.replace('\\', '/')}")
sys.exit(0)
"""
                        # Write script to temporary file
                        script_path = os.path.join(os.path.dirname(self.output_file), "temp_convert_script.py")
                        with open(script_path, "w") as f:
                            f.write(script_content)
                        
                        self.progress.emit(30)
                        
                        # Run FreeCAD with the script
                        cmd = [freecad_path, "-c", script_path]
                        result = subprocess.run(cmd, check=True, timeout=60, 
                                              capture_output=True, text=True)
                        
                        # Clean up
                        try:
                            os.remove(script_path)
                        except:
                            pass
                            
                        self.progress.emit(100)
                        self.conversion_complete.emit(True, self.output_file)
                        return
                        
                    except subprocess.CalledProcessError as e:
                        print(f"FreeCAD conversion failed with exit code {e.returncode}")
                        print(f"Output: {e.stdout}")
                        print(f"Error: {e.stderr}")
                    except Exception as e:
                        print(f"FreeCAD conversion error: {e}")
            
            if not freecad_found:
                print("FreeCAD not found in standard locations")
            
            # Method 2: Try using OpenSCAD if available
            openscad_found = False
            for openscad_path in openscad_paths:
                if os.path.exists(openscad_path):
                    openscad_found = True
                    try:
                        # Create temporary SCAD file
                        scad_content = f'import("{self.input_file.replace("\\", "/")}"); // Import STEP file'
                        scad_file = os.path.join(os.path.dirname(self.output_file), "temp_import.scad")
                        
                        with open(scad_file, "w") as f:
                            f.write(scad_content)
                        
                        self.progress.emit(50)
                        
                        # Set resolution parameters
                        # Higher resolution = more triangles (but OpenSCAD has different params)
                        # Use $fn for OpenSCAD
                        
                        # Run OpenSCAD to convert to STL
                        cmd = [
                            openscad_path,
                            "-o", self.output_file,
                            "--export-format=binstl",
                            scad_file
                        ]
                        result = subprocess.run(cmd, check=True, timeout=60, 
                                              capture_output=True, text=True)
                        
                        # Clean up
                        try:
                            os.remove(scad_file)
                        except:
                            pass
                        
                        self.progress.emit(100)
                        self.conversion_complete.emit(True, self.output_file)
                        return
                        
                    except subprocess.CalledProcessError as e:
                        print(f"OpenSCAD conversion failed with exit code {e.returncode}")
                        print(f"Output: {e.stdout}")
                        print(f"Error: {e.stderr}")
                    except Exception as e:
                        print(f"OpenSCAD conversion error: {e}")
            
            if not openscad_found:
                print("OpenSCAD not found in standard locations")
            
            # If all methods fail, report error
            self.progress.emit(100)
            self.conversion_complete.emit(False, "Conversion failed. Please make sure pythonocc-core is installed, or install FreeCAD or OpenSCAD to use as fallback.\n\nRun: pip install pythonocc-core\n\nOr download:\nFreeCAD: https://www.freecad.org/downloads.php\nOpenSCAD: https://openscad.org/downloads.html")
            
        except Exception as e:
            self.progress.emit(100)
            self.conversion_complete.emit(False, f"Error during conversion: {str(e)}") 