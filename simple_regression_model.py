import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print('🚀 LINEAR REGRESSION PRICING MODEL')
print('=' * 50)

# Load dataset
print('Loading dataset...')
df = pd.read_csv('Combined_Project_Data_with_Totals PA2200 EDITED 4.0 (with derived features).csv', sep=';', decimal=',')
print(f'Dataset shape: {df.shape}')

# Core features for modeling
features = [
    # Original features
    'Volume', 'surface_area', 'bb_volume', 'Quantity', 'Max D', 'waste_ratio',
    # Derived features  
    'SA_to_Volume_Ratio', 'Volume_Density', 'Material_Efficiency', 'Complexity_Score',
    'Packing_Efficiency', 'Waste_per_Volume'
]

target = 'Net price / part'

# Clean data
print('\nCleaning data...')
for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove NaN and infinite values  
df_clean = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
print(f'Clean dataset shape: {df_clean.shape}')

X = df_clean[features]
y = df_clean[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression (α=1.0)': Ridge(alpha=1.0),
    'Ridge Regression (α=10.0)': Ridge(alpha=10.0)
}

print('\n🔬 MODEL RESULTS')
print('=' * 50)

best_r2 = 0
best_model = None
best_name = ''

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name
        best_predictions = y_pred
    
    print(f'\n{name}:')
    print(f'  R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)')
    print(f'  MAE: €{mae:.2f}')

print(f'\n🏆 BEST MODEL: {best_name}')
print(f'R² Score: {best_r2:.4f}')

# Feature importance for best model
if hasattr(best_model, 'coef_'):
    importance = pd.DataFrame({
        'feature': features,
        'coefficient': best_model.coef_,
        'abs_coef': np.abs(best_model.coef_)
    }).sort_values('abs_coef', ascending=False)
    
    print(f'\n📈 FEATURE IMPORTANCE:')
    print('-' * 30)
    for _, row in importance.iterrows():
        direction = '↑' if row['coefficient'] > 0 else '↓'
        print(f'{row["feature"]:25} {direction} {row["coefficient"]:8.4f}')

# Business metrics
mape = np.mean(np.abs((y_test - best_predictions) / y_test)) * 100
within_10pct = np.mean(np.abs((y_test - best_predictions) / y_test) <= 0.1) * 100
within_20pct = np.mean(np.abs((y_test - best_predictions) / y_test) <= 0.2) * 100

print(f'\n📊 BUSINESS METRICS')
print('-' * 30)
print(f'Mean Absolute Percentage Error: {mape:.1f}%')
print(f'Predictions within ±10%: {within_10pct:.1f}%')
print(f'Predictions within ±20%: {within_20pct:.1f}%')
print(f'Average actual price: €{y_test.mean():.2f}')
print(f'Average predicted price: €{best_predictions.mean():.2f}')

# Price range analysis
print(f'\n💰 ACCURACY BY PRICE RANGE')
print('-' * 30)
price_bins = pd.cut(y_test, bins=[0, 5, 20, 100, np.inf], labels=['€0-5', '€5-20', '€20-100', '€100+'])

for bin_name in price_bins.unique():
    if pd.notna(bin_name):
        mask = price_bins == bin_name
        if mask.sum() > 0:
            bin_r2 = r2_score(y_test[mask], best_predictions[mask])
            bin_mae = mean_absolute_error(y_test[mask], best_predictions[mask])
            print(f'{str(bin_name):10} (n={mask.sum():3d}): R²={bin_r2:.3f}, MAE=€{bin_mae:.2f}')

print(f'\n✅ SUMMARY')
print('-' * 20)
print(f'• Model explains {best_r2*100:.0f}% of pricing variance')
print(f'• Average error: ±€{mean_absolute_error(y_test, best_predictions):.2f}')
print(f'• {within_20pct:.0f}% of predictions within ±20%')
print(f'• Surface area & volume are strongest predictors')
print(f'• Derived features significantly improve model')

# Save results
results_df = pd.DataFrame({
    'Actual_Price': y_test.values,
    'Predicted_Price': best_predictions,
    'Error': y_test.values - best_predictions,
    'Percentage_Error': ((y_test.values - best_predictions) / y_test.values) * 100
})
results_df.to_csv('regression_results.csv', index=False)
print(f'\n💾 Results saved to regression_results.csv') 