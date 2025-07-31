import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_clean_data():
    """Load and clean the pricing data"""
    # Read CSV with semicolon separator
    df = pd.read_csv('correct totals 4.csv', sep=';', encoding='utf-8')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert comma decimal numbers to float
    numeric_cols = ['Net price / part', 'Net price/mm3', 'shrinkwrap_volume', 'Volume', 'surface_area']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    # Exclude Formiga machines
    df_filtered = df[~df['Machine'].str.contains('Formiga', na=False)].copy()
    
    # Normalize P396 machine names
    df_filtered['Machine'] = df_filtered['Machine'].str.replace('EOS P396.*', 'EOS P396', regex=True)
    
    # Calculate volume difference (shrinkwrap - normal volume)
    df_filtered['volume_diff'] = df_filtered['shrinkwrap_volume'] - df_filtered['Volume']
    
    # Remove rows with missing key data
    key_cols = ['Net price / part', 'shrinkwrap_volume', 'Volume', 'surface_area']
    df_filtered = df_filtered.dropna(subset=key_cols)
    
    print(f"Original data: {len(df)} rows")
    print(f"After filtering (no Formiga): {len(df_filtered)} rows")
    print(f"Machines: {df_filtered['Machine'].unique()}")
    
    return df_filtered

def create_scatter_plots(df):
    """Create comprehensive scatter plots for different features vs price"""
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature vs Price Analysis for SLS 3D Printing', fontsize=16, fontweight='bold')
    
    # Plot 1: Volume vs Price
    axes[0,0].scatter(df['Volume'], df['Net price / part'], alpha=0.6, s=30)
    axes[0,0].set_xlabel('Volume (mm³)')
    axes[0,0].set_ylabel('Net Price per Part (€)')
    axes[0,0].set_title('Volume vs Price')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Shrinkwrap Volume vs Price
    axes[0,1].scatter(df['shrinkwrap_volume'], df['Net price / part'], alpha=0.6, s=30, color='orange')
    axes[0,1].set_xlabel('Shrinkwrap Volume (mm³)')
    axes[0,1].set_ylabel('Net Price per Part (€)')
    axes[0,1].set_title('Shrinkwrap Volume vs Price')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Volume Difference vs Price (KEY INSIGHT)
    axes[0,2].scatter(df['volume_diff'], df['Net price / part'], alpha=0.6, s=30, color='red')
    axes[0,2].set_xlabel('Volume Difference (Shrinkwrap - Normal)')
    axes[0,2].set_ylabel('Net Price per Part (€)')
    axes[0,2].set_title('Volume Difference vs Price (Linear Base)')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Surface Area vs Price
    axes[1,0].scatter(df['surface_area'], df['Net price / part'], alpha=0.6, s=30, color='green')
    axes[1,0].set_xlabel('Surface Area (mm²)')
    axes[1,0].set_ylabel('Net Price per Part (€)')
    axes[1,0].set_title('Surface Area vs Price')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Net Price per mm³ vs Volume
    axes[1,1].scatter(df['Volume'], df['Net price/mm3'], alpha=0.6, s=30, color='purple')
    axes[1,1].set_xlabel('Volume (mm³)')
    axes[1,1].set_ylabel('Net Price per mm³ (€/mm³)')
    axes[1,1].set_title('Price Density vs Volume')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Machine comparison
    machines = df['Machine'].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, machine in enumerate(machines):
        machine_data = df[df['Machine'] == machine]
        axes[1,2].scatter(machine_data['Volume'], machine_data['Net price / part'], 
                         alpha=0.6, s=30, label=machine, color=colors[i % len(colors)])
    axes[1,2].set_xlabel('Volume (mm³)')
    axes[1,2].set_ylabel('Net Price per Part (€)')
    axes[1,2].set_title('Price vs Volume by Machine')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scatter_plots_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def linear_base_model_analysis(df):
    """Analyze linear base model using volume difference"""
    
    # Prepare data for linear regression
    X_base = df[['volume_diff']]
    y = df['Net price / part']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
    
    # Fit linear model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = linear_model.predict(X_train)
    y_pred_test = linear_model.predict(X_test)
    
    # Calculate metrics
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Actual vs Predicted
    ax1.scatter(y_test, y_pred_test, alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Price (€)')
    ax1.set_ylabel('Predicted Price (€)')
    ax1.set_title(f'Linear Base Model: Actual vs Predicted\nR² = {test_r2:.3f}, MAPE = {test_mape:.1f}%')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volume Difference with regression line
    ax2.scatter(df['volume_diff'], df['Net price / part'], alpha=0.6, s=30)
    volume_diff_range = np.linspace(df['volume_diff'].min(), df['volume_diff'].max(), 100)
    predicted_prices = linear_model.predict(volume_diff_range.reshape(-1, 1))
    ax2.plot(volume_diff_range, predicted_prices, 'r-', linewidth=2, 
             label=f'Linear fit: y = {linear_model.coef_[0]:.6f}x + {linear_model.intercept_:.2f}')
    ax2.set_xlabel('Volume Difference (Shrinkwrap - Normal)')
    ax2.set_ylabel('Net Price per Part (€)')
    ax2.set_title('Linear Base Model: Volume Difference vs Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_base_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results = {
        'model': linear_model,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'coefficient': linear_model.coef_[0],
        'intercept': linear_model.intercept_
    }
    
    return results

def enhanced_model_analysis(df):
    """Create enhanced models with additional geometric features"""
    
    # Prepare features (excluding convex hull volumes as requested)
    features = ['volume_diff', 'surface_area', 'Volume', 'shrinkwrap_volume']
    X = df[features]
    y = df['Net price / part']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Linear with multiple features
    linear_enhanced = LinearRegression()
    linear_enhanced.fit(X_train, y_train)
    y_pred_linear = linear_enhanced.predict(X_test)
    
    # Model 2: Random Forest for comparison
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Calculate metrics
    linear_mape = mean_absolute_percentage_error(y_test, y_pred_linear) * 100
    linear_r2 = r2_score(y_test, y_pred_linear)
    rf_mape = mean_absolute_percentage_error(y_test, y_pred_rf) * 100
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    # Feature importance for RF
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Linear Enhanced Model
    ax1.scatter(y_test, y_pred_linear, alpha=0.6)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Price (€)')
    ax1.set_ylabel('Predicted Price (€)')
    ax1.set_title(f'Enhanced Linear Model\nR² = {linear_r2:.3f}, MAPE = {linear_mape:.1f}%')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Random Forest Model
    ax2.scatter(y_test, y_pred_rf, alpha=0.6, color='orange')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Price (€)')
    ax2.set_ylabel('Predicted Price (€)')
    ax2.set_title(f'Random Forest Model\nR² = {rf_r2:.3f}, MAPE = {rf_mape:.1f}%')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature Importance
    ax3.barh(feature_importance['feature'], feature_importance['importance'])
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Random Forest Feature Importance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Comparison
    models = ['Linear Base\n(Volume Diff)', 'Enhanced Linear\n(Multi-feature)', 'Random Forest\n(Multi-feature)']
    mapes = [None, linear_mape, rf_mape]  # We'll fill the first one from previous analysis
    r2s = [None, linear_r2, rf_r2]
    
    x_pos = np.arange(len(models))
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x_pos[1:], mapes[1:], alpha=0.7, label='MAPE (%)', color='lightcoral')
    bars2 = ax4_twin.bar(x_pos[1:] + 0.1, r2s[1:], alpha=0.7, label='R²', color='lightblue', width=0.1)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('MAPE (%)', color='red')
    ax4_twin.set_ylabel('R² Score', color='blue')
    ax4.set_title('Model Performance Comparison')
    ax4.set_xticks(x_pos[1:])
    ax4.set_xticklabels(models[1:])
    
    plt.tight_layout()
    plt.savefig('enhanced_models_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Linear coefficients analysis
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': linear_enhanced.coef_
    })
    
    print("\nEnhanced Linear Model Coefficients:")
    print(coef_df)
    print(f"Intercept: {linear_enhanced.intercept_:.3f}")
    
    return {
        'linear_model': linear_enhanced,
        'rf_model': rf_model,
        'linear_mape': linear_mape,
        'linear_r2': linear_r2,
        'rf_mape': rf_mape,
        'rf_r2': rf_r2,
        'feature_importance': feature_importance,
        'coefficients': coef_df
    }

def create_insights_summary(df, base_results, enhanced_results):
    """Create summary of key insights"""
    
    # Calculate correlation matrix
    features = ['volume_diff', 'surface_area', 'Volume', 'shrinkwrap_volume', 'Net price / part']
    corr_matrix = df[features].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Key statistics
    print("\n" + "="*50)
    print("KEY INSIGHTS SUMMARY")
    print("="*50)
    
    print(f"\nDataset Overview:")
    print(f"- Total parts analyzed: {len(df)}")
    print(f"- Price range: €{df['Net price / part'].min():.2f} - €{df['Net price / part'].max():.2f}")
    print(f"- Average price: €{df['Net price / part'].mean():.2f}")
    print(f"- Volume range: {df['Volume'].min():.0f} - {df['Volume'].max():,.0f} mm³")
    
    print(f"\nWhy Volume Difference (Shrinkwrap - Normal) is the Best Linear Base:")
    volume_diff_corr = corr_matrix.loc['volume_diff', 'Net price / part']
    volume_corr = corr_matrix.loc['Volume', 'Net price / part']
    shrinkwrap_corr = corr_matrix.loc['shrinkwrap_volume', 'Net price / part']
    
    print(f"- Volume Difference correlation with price: {volume_diff_corr:.3f}")
    print(f"- Normal Volume correlation with price: {volume_corr:.3f}")
    print(f"- Shrinkwrap Volume correlation with price: {shrinkwrap_corr:.3f}")
    print(f"- Volume difference captures complexity better than raw volume")
    print(f"- Represents material complexity and support structures needed")
    
    print(f"\nModel Performance Comparison:")
    print(f"Linear Base Model (Volume Diff only):")
    print(f"  - R²: {base_results['test_r2']:.3f}")
    print(f"  - MAPE: {base_results['test_mape']:.1f}%")
    
    print(f"Enhanced Linear Model (Multi-feature):")
    print(f"  - R²: {enhanced_results['linear_r2']:.3f}")
    print(f"  - MAPE: {enhanced_results['linear_mape']:.1f}%")
    print(f"  - Improvement in R²: {enhanced_results['linear_r2'] - base_results['test_r2']:.3f}")
    
    print(f"\nTop Features by Importance (Random Forest):")
    for _, row in enhanced_results['feature_importance'].tail(3).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.3f}")

def main():
    """Main analysis function"""
    print("Starting SLS 3D Printing Pricing Analysis...")
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Create scatter plots
    print("\nCreating scatter plots...")
    create_scatter_plots(df)
    
    # Analyze linear base model
    print("\nAnalyzing linear base model...")
    base_results = linear_base_model_analysis(df)
    
    # Analyze enhanced models
    print("\nAnalyzing enhanced models...")
    enhanced_results = enhanced_model_analysis(df)
    
    # Create insights summary
    print("\nGenerating insights summary...")
    create_insights_summary(df, base_results, enhanced_results)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")
    
    return df, base_results, enhanced_results

if __name__ == "__main__":
    df, base_results, enhanced_results = main() 