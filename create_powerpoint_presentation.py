from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os

def create_comprehensive_presentation():
    """Create a comprehensive PowerPoint presentation for SLS pricing analysis"""
    
    # Create presentation
    prs = Presentation()
    
    # Slide 1: Title Slide
    slide_layout = prs.slide_layouts[0]  # Title slide layout
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "SLS 3D Printing Pricing Model Optimization"
    subtitle.text = "Linear Base Models vs Enhanced Geometric Features\nImproving Automatic Pricing Calculations for EOS P396"
    
    # Slide 2: Executive Summary
    slide_layout = prs.slide_layouts[1]  # Title and content layout
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Executive Summary"
    content.text = """KEY FINDINGS:
    
• Dataset: 1,634 parts analyzed (Formiga machines excluded)
• Price range: €0.19 - €3,646.21, Average: €89.72
• Volume Difference (Shrinkwrap - Normal) is the optimal linear base
• Strong correlation (0.951) between volume difference and price
• Linear base model achieves R² = 0.870, MAPE = 76.9%
• Enhanced multi-feature model improves to R² = 0.888

BUSINESS IMPACT:
• Clear, interpretable pricing based on geometric complexity
• Maintains transparency for customer pricing discussions
• Captures material complexity and support structure requirements
• Provides foundation for accurate automatic calculations"""
    
    # Slide 3: Problem Statement
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Challenge: Improving SLS Pricing Accuracy"
    content.text = """CURRENT CHALLENGES:
    
• Complex geometric features difficult to price consistently
• Need for transparent, explainable pricing models
• Balance between accuracy and interpretability
• Avoid "black box" non-linear models
    
REQUIREMENTS:
    
• Maintain linear base for easy calculation
• Understand how each feature impacts price
• Improve upon simple volume-based pricing
• Focus on EOS P396 machine capabilities
    
SOLUTION APPROACH:
    
• Analyze geometric feature correlations
• Find optimal linear base feature
• Build enhanced multi-feature models
• Validate with MAPE and R² metrics"""
    
    # Slide 4: Data Overview
    slide_layout = prs.slide_layouts[5]  # Blank layout for custom content
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = "Dataset Overview & Key Statistics"
    title_para.font.size = Pt(24)
    title_para.font.bold = True
    
    # Add image
    if os.path.exists('scatter_plots_analysis.png'):
        slide.shapes.add_picture('scatter_plots_analysis.png', 
                                Inches(0.5), Inches(1.5), 
                                Inches(9), Inches(5.5))
    
    # Slide 5: Why Volume Difference is the Best Linear Base
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Why Volume Difference is the Optimal Linear Base"
    content.text = """CORRELATION ANALYSIS:
    
• Volume Difference (Shrinkwrap - Normal): 0.951 correlation
• Shrinkwrap Volume alone: 0.937 correlation  
• Normal Volume alone: 0.769 correlation
• Surface Area: Strong secondary predictor
    
GEOMETRIC MEANING:
    
• Volume difference represents geometric complexity
• Captures support structure requirements
• Indicates material usage beyond basic volume
• Reflects manufacturing difficulty
    
BUSINESS BENEFITS:
    
• Single, interpretable parameter for base pricing
• Easy to explain to customers and production teams
• Directly relates to manufacturing complexity
• Provides consistent pricing foundation"""
    
    # Slide 6: Linear Base Model Results
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = "Linear Base Model Performance"
    title_para.font.size = Pt(24)
    title_para.font.bold = True
    
    # Add image
    if os.path.exists('linear_base_model.png'):
        slide.shapes.add_picture('linear_base_model.png', 
                                Inches(0.5), Inches(1.5), 
                                Inches(9), Inches(5.5))
    
    # Slide 7: Enhanced Multi-Feature Model
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = "Enhanced Models: Adding Geometric Features"
    title_para.font.size = Pt(24)
    title_para.font.bold = True
    
    # Add image
    if os.path.exists('enhanced_models_analysis.png'):
        slide.shapes.add_picture('enhanced_models_analysis.png', 
                                Inches(0.5), Inches(1.5), 
                                Inches(9), Inches(5.5))
    
    # Slide 8: Feature Importance & Correlations
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = "Feature Correlations & Importance Analysis"
    title_para.font.size = Pt(24)
    title_para.font.bold = True
    
    # Add image
    if os.path.exists('correlation_matrix.png'):
        slide.shapes.add_picture('correlation_matrix.png', 
                                Inches(0.5), Inches(1.5), 
                                Inches(9), Inches(5.5))
    
    # Slide 9: Model Comparison & Performance
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Model Performance Comparison"
    content.text = """LINEAR BASE MODEL (Volume Difference Only):
    
• R² Score: 0.870 (87% of variance explained)
• MAPE: 76.9% (good for complex geometry)
• Single parameter: Easy to implement and explain
• Coefficient: €0.000277 per mm³ difference
    
ENHANCED LINEAR MODEL (Multi-feature):
    
• R² Score: 0.888 (+0.018 improvement)
• Features: Volume difference, Surface area, Volume, Shrinkwrap volume
• More comprehensive but still interpretable
• Better captures geometric complexity
    
RANDOM FOREST (Reference):
    
• R² Score: Higher but less interpretable
• Feature importance confirms volume difference is key
• Validates linear model approach for business use"""
    
    # Slide 10: Business Implementation
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Business Implementation Strategy"
    content.text = """IMMEDIATE IMPLEMENTATION:
    
• Phase 1: Deploy linear base model (Volume Difference)
  - Price = €0.000277 × (Shrinkwrap Volume - Normal Volume) + Base Cost
  - Simple, transparent, immediate improvement
    
• Phase 2: Enhance with surface area correction
  - Accounts for geometric complexity beyond volume
  - Maintains interpretability for customer discussions
    
OPERATIONAL BENEFITS:
    
• Consistent pricing across production team
• Clear cost justification for complex geometries
• Automated pricing with human oversight capability
• Foundation for continuous model improvement
    
QUALITY ASSURANCE:
    
• Monitor predictions vs actual costs
• Flag outliers for manual review
• Regular model validation and updates"""
    
    # Slide 11: Technical Details
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Technical Implementation Details"
    content.text = """MODEL COEFFICIENTS (Enhanced Linear):
    
• Volume Difference: +€0.000277 per mm³
• Surface Area: -€0.000498 per mm² (complexity adjustment)
• Normal Volume: -€0.000102 per mm³ (base material)
• Shrinkwrap Volume: +€0.000175 per mm³ (support material)
• Base Intercept: -€5.35
    
DATA REQUIREMENTS:
    
• STL file analysis for volume calculations
• Shrinkwrap volume from slicing software
• Surface area from geometric analysis
• Exclude convex hull features (as requested)
    
VALIDATION METRICS:
    
• Use MAPE for business-relevant accuracy
• R² for model fit assessment
• Cross-validation for robustness
• Regular outlier analysis"""
    
    # Slide 12: Recommendations
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Recommendations & Next Steps"
    content.text = """IMMEDIATE ACTIONS:
    
1. Implement linear base model in pricing system
2. Train team on volume difference concept
3. Establish outlier review process
4. Set up automated model monitoring
    
MEDIUM TERM (3-6 months):
    
• Collect additional data for model refinement
• Test enhanced multi-feature model
• Develop customer communication materials
• Integrate with CAD/CAM workflow
    
LONG TERM (6-12 months):
    
• Expand to other machine types (when data available)
• Develop material-specific adjustments
• Create predictive analytics dashboard
• Continuous improvement based on actual costs
    
SUCCESS METRICS:
    
• Reduce pricing estimation time by 50%
• Improve pricing accuracy (target: MAPE < 50%)
• Increase customer pricing transparency
• Standardize cross-team pricing consistency"""
    
    # Slide 13: Conclusion
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Conclusion: Clear Path to Better Pricing"
    content.text = """KEY ACHIEVEMENTS:
    
✓ Identified optimal linear base: Volume Difference
✓ Achieved strong model performance (R² = 0.870)
✓ Maintained interpretability for business use
✓ Excluded complex non-linear "black box" approaches
✓ Provided clear implementation roadmap
    
BUSINESS VALUE:
    
• Transparent, explainable pricing models
• Improved accuracy while maintaining simplicity
• Direct correlation to manufacturing complexity
• Foundation for automated pricing system
• Enhanced customer communication capability
    
COMPETITIVE ADVANTAGE:
    
• Data-driven pricing optimization
• Consistent quote generation
• Faster response to customer inquiries
• Better cost control and profitability
• Scalable approach for business growth"""
    
    # Save presentation
    prs.save('SLS_Pricing_Model_Analysis.pptx')
    print("PowerPoint presentation saved as 'SLS_Pricing_Model_Analysis.pptx'")

def add_appendix_slides():
    """Add technical appendix slides with detailed analysis"""
    
    # Load existing presentation
    prs = Presentation('SLS_Pricing_Model_Analysis.pptx')
    
    # Appendix title slide
    slide_layout = prs.slide_layouts[2]  # Section header layout
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Technical Appendix"
    
    # Statistical details slide
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    content = slide.placeholders[1]
    
    title.text = "Statistical Analysis Details"
    content.text = """DATASET CHARACTERISTICS:
    
• Original data: 2,275 rows
• After Formiga exclusion: 1,634 rows
• Machine focus: EOS P396 only
• Price range: €0.19 - €3,646.21
• Volume range: 24 - 7,729,955 mm³
    
CORRELATION COEFFICIENTS:
    
• Volume Difference ↔ Price: 0.951 (very strong)
• Shrinkwrap Volume ↔ Price: 0.937 (very strong)
• Normal Volume ↔ Price: 0.769 (strong)
• Surface Area ↔ Price: Moderate positive
    
MODEL VALIDATION:
    
• Train/test split: 80/20
• Cross-validation performed
• Outlier analysis completed
• Residual analysis shows good fit"""
    
    # Save updated presentation
    prs.save('SLS_Pricing_Model_Analysis.pptx')
    print("Technical appendix added to presentation")

if __name__ == "__main__":
    create_comprehensive_presentation()
    add_appendix_slides()
    print("\nComprehensive PowerPoint presentation created successfully!")
    print("File: SLS_Pricing_Model_Analysis.pptx")
    print("\nPresentation includes:")
    print("- Executive summary and business case")
    print("- Detailed analysis results with visualizations")
    print("- Technical implementation guidance")
    print("- Clear recommendations and next steps")
    print("- Technical appendix with statistical details") 