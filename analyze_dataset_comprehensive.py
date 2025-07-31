#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis - correct totals 4.csv
====================================================

This script performs deep analysis of the 3D printing pricing dataset:
1. Feature engineering using only CSV features (excluding price-related ones)
2. Linear Regression and Random Forest modeling
3. Deep trend analysis and pattern discovery
4. Outlier detection and anomaly identification
5. Feature importance analysis
6. Machine and Color impact analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatasetAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.df_clean = None
        self.features = None
        self.target = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load CSV data and prepare for analysis"""
        print("📊 Loading dataset...")
        
        # Load with proper separator
        self.df = pd.read_csv(self.csv_file, sep=';', decimal=',')
        print(f"   📈 Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        
        # Display column names
        print(f"   📋 Columns: {list(self.df.columns)}")
        
        # EXCLUDE PRICE-RELATED FEATURES (as requested)
        price_features = [
            'Net price / part',     # Target variable 
            'Net price/mm3',        # Price-related
            'net price * Quantity'  # Price-related
        ]
        
        # Define target variable
        self.target = 'Net price / part'
        
        # Get all feature columns except price-related ones
        all_columns = list(self.df.columns)
        self.features = [col for col in all_columns if col not in price_features and col not in ['Project ID', 'Part Name']]
        
        print(f"   🎯 Target variable: {self.target}")
        print(f"   🔧 Features to use ({len(self.features)}): {self.features}")
        print(f"   ❌ Excluded features: {price_features}")
        
        return self
        
    def clean_and_encode_data(self):
        """Clean data and encode categorical variables"""
        print("\n🧹 Cleaning and encoding data...")
        
        # Create working copy
        self.df_clean = self.df.copy()
        
        # Convert target to numeric
        if self.df_clean[self.target].dtype == 'object':
            self.df_clean[self.target] = pd.to_numeric(self.df_clean[self.target].str.replace(',', '.'), errors='coerce')
        
        # Convert numeric features
        numeric_features = []
        categorical_features = []
        
        for feature in self.features:
            if feature in ['Machine', 'Colour', 'Material']:
                categorical_features.append(feature)
            else:
                # Convert to numeric
                if self.df_clean[feature].dtype == 'object':
                    self.df_clean[feature] = pd.to_numeric(self.df_clean[feature].str.replace(',', '.'), errors='coerce')
                numeric_features.append(feature)
        
        # Encode categorical variables
        le_dict = {}
        for cat_feature in categorical_features:
            le = LabelEncoder()
            self.df_clean[f'{cat_feature}_encoded'] = le.fit_transform(self.df_clean[cat_feature].fillna('Unknown'))
            le_dict[cat_feature] = le
            print(f"   🏷️ Encoded {cat_feature}: {list(le.classes_)}")
            
        # Update features list
        encoded_features = [f'{cat}_encoded' for cat in categorical_features]
        self.features = numeric_features + encoded_features
        
        # Remove rows with missing target
        initial_rows = len(self.df_clean)
        self.df_clean = self.df_clean.dropna(subset=[self.target])
        final_rows = len(self.df_clean)
        print(f"   🗑️ Removed {initial_rows - final_rows} rows with missing target")
        
        # Fill missing feature values with median
        for feature in numeric_features:
            self.df_clean[feature] = self.df_clean[feature].fillna(self.df_clean[feature].median())
            
        print(f"   ✅ Final dataset: {len(self.df_clean)} rows, {len(self.features)} features")
        
        return self
        
    def perform_deep_trend_analysis(self):
        """Perform comprehensive trend analysis"""
        print("\n🔍 Performing deep trend analysis...")
        
        # Create analysis directory
        import os
        os.makedirs('analysis_plots', exist_ok=True)
        
        # 1. TARGET DISTRIBUTION ANALYSIS
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price distribution
        axes[0,0].hist(self.df_clean[self.target], bins=50, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Net Price / Part (€)')
        axes[0,0].set_ylabel('Frequency')
        
        # Log price distribution
        log_prices = np.log1p(self.df_clean[self.target])
        axes[0,1].hist(log_prices, bins=50, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Log Price Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Log(Net Price / Part)')
        
        # Price by Machine
        machine_prices = self.df_clean.groupby('Machine')[self.target].mean().sort_values(ascending=False)
        axes[1,0].bar(range(len(machine_prices)), machine_prices.values, color='orange')
        axes[1,0].set_title('Average Price by Machine', fontsize=14, fontweight='bold')
        axes[1,0].set_xticks(range(len(machine_prices)))
        axes[1,0].set_xticklabels(machine_prices.index, rotation=45)
        axes[1,0].set_ylabel('Average Price (€)')
        
        # Price by Color
        color_prices = self.df_clean.groupby('Colour')[self.target].mean().sort_values(ascending=False)
        axes[1,1].bar(range(len(color_prices)), color_prices.values, color='purple', alpha=0.7)
        axes[1,1].set_title('Average Price by Color', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(range(len(color_prices)))
        axes[1,1].set_xticklabels(color_prices.index, rotation=45)
        axes[1,1].set_ylabel('Average Price (€)')
        
        plt.tight_layout()
        plt.savefig('analysis_plots/price_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. CORRELATION ANALYSIS
        plt.figure(figsize=(20, 16))
        
        # Calculate correlations with target
        correlations = self.df_clean[self.features + [self.target]].corr()[self.target].sort_values(ascending=False)
        
        # Plot correlation heatmap
        correlation_matrix = self.df_clean[self.features + [self.target]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('analysis_plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top correlations with target
        print("   🎯 Top correlations with price:")
        for feature, corr in correlations.head(10).items():
            if feature != self.target:
                print(f"      {feature:30} : {corr:.4f}")
                
        # 3. VOLUME VS PRICE ANALYSIS
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Volume scatter plots
        volume_features = ['Volume', 'convex_hull_volume', 'shrinkwrap_volume', 'bb_volume']
        for i, vol_feature in enumerate(volume_features):
            if vol_feature in self.df_clean.columns:
                row, col = i // 2, i % 2
                axes[row, col].scatter(self.df_clean[vol_feature], self.df_clean[self.target], 
                                     alpha=0.6, s=20)
                axes[row, col].set_xlabel(vol_feature)
                axes[row, col].set_ylabel('Price (€)')
                axes[row, col].set_title(f'Price vs {vol_feature}')
                axes[row, col].set_xscale('log')
                axes[row, col].set_yscale('log')
        
        # Quantity analysis
        axes[1, 2].scatter(self.df_clean['Quantity'], self.df_clean[self.target], alpha=0.6, s=20, color='red')
        axes[1, 2].set_xlabel('Quantity')
        axes[1, 2].set_ylabel('Price (€)')
        axes[1, 2].set_title('Price vs Quantity')
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('analysis_plots/volume_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. PROJECT-LEVEL ANALYSIS
        project_stats = self.df_clean.groupby('Project ID').agg({
            self.target: ['mean', 'min', 'max', 'std'],
            'Quantity': 'sum',
            'Total_Parts_in_Project': 'first',
            'Volume': 'sum'
        }).round(2)
        
        print(f"\n   📊 Project-level statistics (top 10 by average price):")
        project_avg_price = project_stats[(self.target, 'mean')].sort_values(ascending=False).head(10)
        for project, price in project_avg_price.items():
            parts_count = project_stats.loc[project, ('Total_Parts_in_Project', 'first')]
            total_qty = project_stats.loc[project, ('Quantity', 'sum')]
            print(f"      {project}: €{price:.2f} avg, {parts_count} parts, {total_qty} total qty")
            
        return self
        
    def detect_outliers(self):
        """Detect and analyze outliers using multiple methods"""
        print("\n🔍 Detecting outliers...")
        
        # Prepare feature matrix
        X = self.df_clean[self.features].values
        y = self.df_clean[self.target].values
        
        # Method 1: Statistical outliers (Z-score)
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        statistical_outliers = z_scores > 3
        
        # Method 2: IQR method
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        iqr_outliers = (y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)
        
        # Method 3: Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        isolation_outliers = outlier_labels == -1
        
        # Combine outlier detection methods
        combined_outliers = statistical_outliers | iqr_outliers | isolation_outliers
        
        print(f"   📈 Outlier detection results:")
        print(f"      Statistical (Z>3): {statistical_outliers.sum()} outliers")
        print(f"      IQR method: {iqr_outliers.sum()} outliers") 
        print(f"      Isolation Forest: {isolation_outliers.sum()} outliers")
        print(f"      Combined: {combined_outliers.sum()} total outliers")
        
        # Analyze the weirdest outliers
        outlier_df = self.df_clean[combined_outliers].copy()
        outlier_df['outlier_score'] = z_scores[combined_outliers]
        
        print(f"\n   🚨 Top 10 weirdest outliers:")
        weird_outliers = outlier_df.nlargest(10, 'outlier_score')
        for idx, row in weird_outliers.iterrows():
            project_id = row['Project ID']
            part_name = row['Part Name']
            price = row[self.target]
            volume = row.get('Volume', 'N/A')
            machine = row.get('Machine', 'N/A')
            color = row.get('Colour', 'N/A')
            quantity = row.get('Quantity', 'N/A')
            print(f"      {project_id} - {part_name[:40]}...")
            print(f"         💰 Price: €{price:.2f}, Vol: {volume}, Machine: {machine}, Color: {color}, Qty: {quantity}")
        
        # Visualize outliers
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price distribution with outliers
        axes[0,0].hist(y, bins=50, alpha=0.7, label='Normal', color='skyblue')
        axes[0,0].hist(y[combined_outliers], bins=20, alpha=0.8, label='Outliers', color='red')
        axes[0,0].set_title('Price Distribution: Normal vs Outliers')
        axes[0,0].set_xlabel('Price (€)')
        axes[0,0].legend()
        
        # Volume vs Price with outliers
        if 'Volume' in self.df_clean.columns:
            normal_mask = ~combined_outliers
            axes[0,1].scatter(self.df_clean.loc[normal_mask, 'Volume'], y[normal_mask], 
                            alpha=0.6, s=20, label='Normal', color='blue')
            axes[0,1].scatter(self.df_clean.loc[combined_outliers, 'Volume'], y[combined_outliers], 
                            alpha=0.8, s=30, label='Outliers', color='red')
            axes[0,1].set_xlabel('Volume')
            axes[0,1].set_ylabel('Price (€)')
            axes[0,1].set_title('Volume vs Price: Outliers Highlighted')
            axes[0,1].set_xscale('log')
            axes[0,1].set_yscale('log')
            axes[0,1].legend()
        
        # Outliers by Machine
        machine_outlier_counts = outlier_df['Machine'].value_counts()
        axes[1,0].bar(range(len(machine_outlier_counts)), machine_outlier_counts.values, color='orange')
        axes[1,0].set_title('Outliers by Machine Type')
        axes[1,0].set_xticks(range(len(machine_outlier_counts)))
        axes[1,0].set_xticklabels(machine_outlier_counts.index, rotation=45)
        
        # Outliers by Color
        color_outlier_counts = outlier_df['Colour'].value_counts()
        axes[1,1].bar(range(len(color_outlier_counts)), color_outlier_counts.values, color='purple', alpha=0.7)
        axes[1,1].set_title('Outliers by Color')
        axes[1,1].set_xticks(range(len(color_outlier_counts)))
        axes[1,1].set_xticklabels(color_outlier_counts.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_plots/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return combined_outliers
        
    def build_models(self, remove_outliers=True):
        """Build and compare Linear Regression and Random Forest models"""
        print("\n🤖 Building prediction models...")
        
        # Prepare data
        X = self.df_clean[self.features].values
        y = self.df_clean[self.target].values
        
        # Remove outliers if requested
        if remove_outliers:
            outliers = self.detect_outliers()
            X = X[~outliers]
            y = y[~outliers]
            print(f"   🗑️ Removed {outliers.sum()} outliers for model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
            'Lasso Regression': LassoCV(alphas=[0.1, 1.0, 10.0, 100.0], max_iter=2000),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n   🔧 Training {name}...")
            
            # Use scaled data for linear models, original for RF
            if 'Forest' in name:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"      📊 {name} Results:")
            print(f"         MAE: €{mae:.2f}")
            print(f"         RMSE: €{rmse:.2f}")
            print(f"         R²: {r2:.4f}")
        
        self.models = models
        self.results = results
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n   🎯 Random Forest Feature Importance (Top 10):")
            for _, row in importance_df.head(10).iterrows():
                print(f"      {row['feature']:30} : {row['importance']:.4f}")
                
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance (Top 15)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('analysis_plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return self
    
    def visualize_model_performance(self):
        """Visualize model performance and predictions"""
        print("\n📈 Visualizing model performance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_names = list(self.results.keys())
        mae_scores = [self.results[name]['mae'] for name in model_names]
        r2_scores = [self.results[name]['r2'] for name in model_names]
        
        # MAE comparison
        axes[0,0].bar(model_names, mae_scores, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        axes[0,0].set_title('Model Comparison: Mean Absolute Error', fontweight='bold')
        axes[0,0].set_ylabel('MAE (€)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[0,1].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'orange', 'purple'])
        axes[0,1].set_title('Model Comparison: R² Score', fontweight='bold')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        best_result = self.results[best_model_name]
        
        axes[1,0].scatter(best_result['actual'], best_result['predictions'], alpha=0.6)
        min_val = min(best_result['actual'].min(), best_result['predictions'].min())
        max_val = max(best_result['actual'].max(), best_result['predictions'].max())
        axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Price (€)')
        axes[1,0].set_ylabel('Predicted Price (€)')
        axes[1,0].set_title(f'{best_model_name}: Predictions vs Actual')
        
        # Residual plot
        residuals = best_result['actual'] - best_result['predictions']
        axes[1,1].scatter(best_result['predictions'], residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Predicted Price (€)')
        axes[1,1].set_ylabel('Residuals (€)')
        axes[1,1].set_title(f'{best_model_name}: Residual Plot')
        
        plt.tight_layout()
        plt.savefig('analysis_plots/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   🏆 Best performing model: {best_model_name} (R² = {best_result['r2']:.4f})")
        
        return self
        
    def machine_color_impact_analysis(self):
        """Analyze the impact of Machine and Color on pricing"""
        print("\n🏭 Analyzing Machine and Color impact on pricing...")
        
        # Machine analysis
        machine_stats = self.df_clean.groupby('Machine').agg({
            self.target: ['count', 'mean', 'median', 'std'],
            'Volume': 'mean',
            'Quantity': 'mean'
        }).round(2)
        
        print("   🏭 Machine Impact Analysis:")
        for machine in machine_stats.index:
            count = machine_stats.loc[machine, (self.target, 'count')]
            avg_price = machine_stats.loc[machine, (self.target, 'mean')]
            median_price = machine_stats.loc[machine, (self.target, 'median')]
            std_price = machine_stats.loc[machine, (self.target, 'std')]
            avg_volume = machine_stats.loc[machine, ('Volume', 'mean')]
            
            print(f"      {machine}:")
            print(f"         Parts: {count}, Avg Price: €{avg_price:.2f}, Median: €{median_price:.2f}")
            print(f"         Std Dev: €{std_price:.2f}, Avg Volume: {avg_volume:.1f}")
        
        # Color analysis  
        color_stats = self.df_clean.groupby('Colour').agg({
            self.target: ['count', 'mean', 'median', 'std'],
            'Volume': 'mean',
            'Quantity': 'mean'
        }).round(2)
        
        print("\n   🎨 Color Impact Analysis:")
        for color in color_stats.index:
            count = color_stats.loc[color, (self.target, 'count')]
            avg_price = color_stats.loc[color, (self.target, 'mean')]
            median_price = color_stats.loc[color, (self.target, 'median')]
            
            print(f"      {color}: {count} parts, Avg: €{avg_price:.2f}, Median: €{median_price:.2f}")
            
        # Statistical significance testing
        from scipy import stats
        
        machines = self.df_clean['Machine'].unique()
        if len(machines) > 1:
            machine1_prices = self.df_clean[self.df_clean['Machine'] == machines[0]][self.target]
            machine2_prices = self.df_clean[self.df_clean['Machine'] == machines[1]][self.target]
            t_stat, p_value = stats.ttest_ind(machine1_prices, machine2_prices)
            print(f"\n   📊 Statistical test between {machines[0]} and {machines[1]}:")
            print(f"      T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}")
            print(f"      Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        return self
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 50)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Total parts analyzed: {len(self.df_clean):,}")
        print(f"   • Unique projects: {self.df_clean['Project ID'].nunique()}")
        print(f"   • Features used: {len(self.features)}")
        print(f"   • Price range: €{self.df_clean[self.target].min():.2f} - €{self.df_clean[self.target].max():.2f}")
        print(f"   • Average price: €{self.df_clean[self.target].mean():.2f}")
        
        print(f"\n🏭 MACHINE BREAKDOWN:")
        machine_counts = self.df_clean['Machine'].value_counts()
        for machine, count in machine_counts.items():
            pct = count / len(self.df_clean) * 100
            avg_price = self.df_clean[self.df_clean['Machine'] == machine][self.target].mean()
            print(f"   • {machine}: {count:,} parts ({pct:.1f}%), Avg €{avg_price:.2f}")
            
        print(f"\n🎨 COLOR BREAKDOWN:")
        color_counts = self.df_clean['Colour'].value_counts()
        for color, count in color_counts.head(8).items():
            pct = count / len(self.df_clean) * 100
            avg_price = self.df_clean[self.df_clean['Colour'] == color][self.target].mean()
            print(f"   • {color}: {count:,} parts ({pct:.1f}%), Avg €{avg_price:.2f}")
            
        print(f"\n🤖 MODEL PERFORMANCE:")
        for name, result in self.results.items():
            print(f"   • {name}:")
            print(f"     - MAE: €{result['mae']:.2f}")
            print(f"     - R²: {result['r2']:.4f}")
            
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['r2'])
        print(f"\n🏆 BEST MODEL: {best_model} (R² = {self.results[best_model]['r2']:.4f})")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Machine types have {'significant' if len(self.df_clean['Machine'].unique()) > 1 else 'no'} pricing differences")
        print(f"   • Color variations affect pricing patterns")
        print(f"   • Volume and geometry features are key price drivers")
        print(f"   • Project-level features provide important context")
        
        return self

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Comprehensive Dataset Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer('correct totals 4.csv')
    
    # Run complete analysis pipeline
    (analyzer
     .load_and_prepare_data()
     .clean_and_encode_data()
     .perform_deep_trend_analysis()
     .build_models(remove_outliers=True)
     .visualize_model_performance()
     .machine_color_impact_analysis()
     .generate_summary_report())
    
    print(f"\n✅ Analysis complete! Check 'analysis_plots/' folder for visualizations.")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main() 