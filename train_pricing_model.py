"""
Train and save the Random Forest pricing model
Run this script once to create the pre-trained model files
"""

from random_forest_pricing import main as train_random_forest

if __name__ == "__main__":
    print("🚀 Training and saving Random Forest pricing model...")
    model, scaler, metrics = train_random_forest()
    print("\n✅ Model training complete! Files saved:")
    print("   - random_forest_pricing_model.pkl")
    print("   - random_forest_feature_scaler.pkl")
    print("   - random_forest_model_metadata.json")
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {metrics['test_r2']:.4f}")
    print(f"   MAE: €{metrics['test_mae']:.2f}")
    print(f"   Overfitting Risk: {metrics['overfitting_risk']}") 