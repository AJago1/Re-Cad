# STL Analyzer - 3D Model Analysis and Pricing Tool

A comprehensive desktop application for analyzing STL files, extracting geometric features, and calculating pricing for 3D printing projects using machine learning.

## Features

### 🔍 **STL File Analysis**
- **Automated Scanning**: Batch process entire directories of STL files
- **Geometric Feature Extraction**: Volume, surface area, bounding box dimensions, convexity ratio
- **Advanced Shrinkwrap Analysis**: Calculate optimal material usage and support requirements
- **Smart Bounding Box Calculation**: Multiple algorithms (PCA, weighted PCA, oriented bounding box)

### 📊 **Database Management**
- SQLite database for storing analysis results
- Real-time data updates and visualization
- Export functionality for further analysis
- Search and filtering capabilities

### 🎯 **Similarity Search**
- Find similar parts based on geometric properties
- Multiple comparison parameters (volume, surface area, complexity)
- Visual similarity results with 3D preview

### 💰 **AI-Powered Pricing**
- Machine learning models for automatic price calculation
- Multiple pricing strategies and presets
- Project-aware pricing with volume discounts
- Feature importance analysis

### 🖥️ **Modern GUI Interface**
- Intuitive tabbed interface built with PyQt5
- Real-time 3D model visualization using VTK
- Comprehensive details panel with all measurements
- Progress tracking for batch operations

### ⚙️ **Advanced Processing Options**
- Configurable shrinkwrap settings (offset, output modes)
- File generation options for workflow integration
- Custom output directories and organization
- Fast mode for large datasets

## Installation

### Prerequisites
- Python 3.8+
- Windows 10/11 (tested)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/stl-analyzer.git
cd stl-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
cd stl_analyzer
python main.py
```

## Usage

### Basic Workflow

1. **Select Directory**: Browse and select a folder containing STL files
2. **Configure Settings**: Adjust shrinkwrap parameters and output options
3. **Start Scanning**: Click "Scan Directory" to begin batch processing
4. **View Results**: Examine the database table with extracted features
5. **3D Preview**: Click any entry to load the 3D model and view details
6. **Price Calculation**: Use the pricing tabs for cost estimation

### Key Components

#### STL Analyzer Tab
- **Directory Selection**: Choose folders to scan
- **Shrinkwrap Settings**: Configure offset (0.1-20.0mm) and output modes
- **Database View**: Sortable table with all geometric properties
- **Similarity Search**: Find parts with similar characteristics

#### Price Calculator Tab
- **Individual Pricing**: Calculate costs for specific parts
- **Multiple Pricing Models**: Choose from various algorithms
- **Custom Parameters**: Adjust material costs, labor rates, margins

#### Project Pricing Tab
- **Batch Pricing**: Calculate costs for entire projects
- **Volume Discounts**: Automatic scaling based on project size
- **Export Options**: Generate pricing reports

#### STEP Converter Tab
- **File Conversion**: Convert STEP files to STL format
- **Quality Settings**: Configurable resolution and accuracy

## Technical Details

### Supported File Formats
- **Input**: STL (ASCII and Binary), STEP (via conversion)
- **Database**: SQLite for data persistence
- **Export**: CSV, Excel for analysis results

### Machine Learning Models
- **Random Forest**: Primary pricing algorithm
- **LASSO Regression**: Alternative lightweight model
- **Feature Engineering**: Automated geometric feature calculation
- **Model Persistence**: Trained models saved for reuse

### Performance Features
- **Multi-threading**: Parallel processing for large datasets
- **Memory Optimization**: Efficient handling of large STL files
- **Fast Mode**: Skip intensive calculations for rapid scanning
- **Progress Tracking**: Real-time feedback on long operations

## Architecture

### Modular Design
```
stl_analyzer/
├── gui/                    # User interface components
│   ├── main_window.py     # Main application window
│   ├── tabs/              # Individual tab implementations
│   ├── widgets/           # Reusable UI components
│   ├── threads/           # Background processing
│   └── dialogs/           # Modal dialogs
├── stl_utils.py           # Core STL processing functions
├── database.py            # Data persistence layer
├── viewer.py              # 3D visualization component
├── comprehensive_pricing_model.py # ML pricing engine
└── main.py               # Application entry point
```

### Key Technologies
- **PyQt5**: Modern desktop GUI framework
- **VTK**: 3D visualization and mesh processing
- **NumPy**: Numerical computations and geometry
- **Pandas**: Data analysis and manipulation
- **Scikit-learn**: Machine learning algorithms
- **Trimesh**: 3D mesh processing utilities

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before submitting

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all functions
- Include type hints where appropriate

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is provided for educational and research purposes. Pricing calculations are estimates and should be verified before commercial use. The accuracy of geometric analysis depends on STL file quality and resolution.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Note**: This tool is designed for 3D printing cost estimation and geometric analysis. Database files and trained models are excluded from version control to protect sensitive business data.
