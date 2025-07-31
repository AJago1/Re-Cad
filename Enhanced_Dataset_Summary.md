# Enhanced 3D Printing Dataset Analysis Summary

## Original Dataset
- **File**: `Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (removed false prices).csv`
- **Original Columns**: 22
- **Records**: 2,282 3D printing parts

## Enhanced Dataset  
- **File**: `Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (with derived features).csv`
- **Total Columns**: 50 (28 new derived features)
- **Enhanced Features Added**: 26

## New Derived Features Created

### 1. **Geometric Ratios & Efficiency Metrics**
- **SA_to_Volume_Ratio**: Surface Area / Volume (affects printing time)
- **Volume_Density**: Volume / Bounding Box Volume (packing efficiency)
- **Packing_Efficiency**: Volume / BB Volume (how well part fills space)
- **Material_Efficiency**: Volume / (Volume + Waste) (material usage efficiency)
- **Volume_to_SA_Ratio**: Volume / Surface Area (inverse complexity)

### 2. **Pricing Analysis Features**
- **Price_per_Volume**: Net price / Volume (cost efficiency)
- **Price_per_Surface_Area**: Net price / Surface Area (surface cost)
- **Price_per_BB_Volume**: Net price / BB Volume (space cost)
- **Total_Part_Value**: Price × Quantity (total value per part type)

### 3. **Size & Complexity Categorization**
- **Size_Category**: Small/Medium/Large/XLarge (based on volume)
- **Aspect_Ratio_Category**: Compact/Elongated/Very_Elongated
- **Complexity_Score**: Composite metric of geometric complexity

### 4. **Production & Waste Analysis**
- **Waste_per_Volume**: Waste / Volume (waste intensity)
- **Total_Material_Usage**: Volume + Waste (total material needed)
- **Waste_Efficiency**: Volume / Shrinkwrap Volume
- **Production_Scale**: Large_Batch/Medium_Batch/Small_Batch/Prototype

### 5. **Project-Level Insights**
- **Total_Project_Value**: Sum of all part values in project
- **Avg_Part_Price**: Average part price in project
- **Price_Std_Dev**: Price variation within project
- **Min_Part_Price/Max_Part_Price**: Price range in project
- **Part_Count**: Number of different parts in project
- **Part_Value_Share**: This part's share of total project value
- **Price_Position_in_Project**: Standardized price position within project

## Key Pricing Insights Discovered

### **Strongest Price Correlators** (in order of correlation strength):
1. **Surface Area** (r = 0.939) - STRONGEST predictor
2. **Bounding Box Volume** (r = 0.869) 
3. **Part Volume** (r = 0.776)
4. **Maximum Dimension** (r = 0.767)
5. **Waste per Volume** (r = 0.516)

### **Size Categories & Pricing**:
- **Small**: €2.69 average (251 parts)
- **Medium**: €10.65 average (907 parts) 
- **Large**: €38.01 average (637 parts)
- **XLarge**: €240.56 average (487 parts)

### **Production Scale Impact**:
- **Large Batch**: €4.30 average
- **Medium Batch**: €9.28 average  
- **Small Batch**: €43.59 average
- **Prototype**: €131.52 average

### **Machine Type Differences**:
Different machines show distinct pricing patterns, suggesting operational cost differences.

## Business Insights

### **🎯 Pricing Optimization Opportunities**:
1. **Surface area is the #1 pricing factor** - 94% correlation with price
2. **Batch size dramatically affects pricing** - 30x difference between large batch vs prototype
3. **Size categories show exponential pricing** - XLarge parts cost 89x more than Small
4. **Waste efficiency varies significantly** - opportunity for material optimization

### **📊 Parts Requiring Price Review**:
- Parts with exceptionally high price/volume ratios identified
- Prototypes vs batch production pricing inconsistencies flagged
- Material efficiency outliers highlighted

### **⚡ Process Improvements**:
- Better batch planning could reduce costs significantly
- Material efficiency improvements could lower waste costs
- Surface area optimization could be key pricing lever

## Files Created
1. `add_derived_features.py` - Script to generate enhanced dataset
2. `analyze_pricing_factors.py` - Correlation and insights analysis
3. `Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (with derived features).csv` - Enhanced dataset
4. This summary document

## Usage
The enhanced dataset now provides powerful analytical capabilities for:
- **Predictive pricing models**
- **Cost optimization strategies** 
- **Production planning insights**
- **Material efficiency improvements**
- **Quality vs cost trade-off analysis** 