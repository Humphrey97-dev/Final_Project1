# Disease Outbreak Prediction System
# SDG 3: Good Health and Well-being
# Author: [Your Name]
# Date: [Current Date]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class DiseaseOutbreakPredictor:
    """
    Machine Learning model to predict disease outbreaks for SDG 3
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_preprocess_data(self, data_path=None):
        """
        Load and preprocess disease outbreak data
        """
        # For demo purposes, creating synthetic data
        # Replace this with actual dataset loading
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        data = {
            'population_density': np.random.exponential(100, n_samples),
            'temperature_avg': np.random.normal(25, 10, n_samples),
            'rainfall_mm': np.random.exponential(50, n_samples),
            'humidity_percent': np.random.normal(60, 20, n_samples),
            'gdp_per_capita': np.random.lognormal(8, 1, n_samples),
            'hospital_beds_per_1000': np.random.exponential(2, n_samples),
            'vaccination_rate': np.random.beta(2, 2, n_samples) * 100,
            'travel_volume': np.random.exponential(1000, n_samples),
            'sanitation_index': np.random.beta(3, 2, n_samples) * 100,
            'education_index': np.random.beta(4, 2, n_samples) * 100
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (outbreak risk)
        # Higher risk with: high density, extreme temperatures, low healthcare capacity
        risk_factors = (
            (df['population_density'] > df['population_density'].quantile(0.8)) * 0.3 +
            (abs(df['temperature_avg'] - 25) > 10) * 0.2 +
            (df['hospital_beds_per_1000'] < df['hospital_beds_per_1000'].quantile(0.3)) * 0.2 +
            (df['vaccination_rate'] < 50) * 0.2 +
            (df['sanitation_index'] < 60) * 0.1
        )
        
        # Add some randomness
        df['outbreak_risk'] = (risk_factors + np.random.normal(0, 0.1, len(df)) > 0.4).astype(int)
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional features for better prediction
        """
        # Temperature-humidity interaction
        df['temp_humidity_interaction'] = df['temperature_avg'] * df['humidity_percent']
        
        # Healthcare capacity ratio
        df['healthcare_capacity'] = df['hospital_beds_per_1000'] / df['population_density'] * 1000
        
        # Socioeconomic health index
        df['socioeconomic_health'] = (df['gdp_per_capita'] / 1000 + 
                                    df['education_index'] + 
                                    df['sanitation_index']) / 3
        
        # Climate stress index
        df['climate_stress'] = abs(df['temperature_avg'] - 25) + df['rainfall_mm'] / 100
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training
        """
        feature_columns = [col for col in df.columns if col != 'outbreak_risk']
        X = df[feature_columns].copy()
        y = df['outbreak_risk'].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        self.feature_names = feature_columns
        return X, y
    
    def train_model(self, X, y):
        """
        Train the disease outbreak prediction model
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        print("Model training completed!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_model(self):
        """
        Evaluate model performance
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Classification report
        print("\n=== Model Performance ===")
        print(classification_report(self.y_test, self.y_pred))
        
        # AUC Score
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        print(f"\nAUC-ROC Score: {auc_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Top 5 Most Important Features ===")
        print(feature_importance.head())
        
        return feature_importance
    
    def visualize_results(self, feature_importance):
        """
        Create visualizations for the results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature Importance
        axes[0, 0].barh(feature_importance.head(8)['feature'], 
                       feature_importance.head(8)['importance'])
        axes[0, 0].set_title('Top 8 Feature Importances')
        axes[0, 0].set_xlabel('Importance')
        
        # 2. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. Prediction Distribution
        axes[1, 0].hist(self.y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of Outbreak Probabilities')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Risk Score Distribution by Class
        prob_df = pd.DataFrame({
            'probability': self.y_pred_proba,
            'actual': self.y_test
        })
        
        for class_val in [0, 1]:
            subset = prob_df[prob_df['actual'] == class_val]['probability']
            axes[1, 1].hist(subset, alpha=0.6, 
                          label=f'Class {class_val} ({"No Outbreak" if class_val == 0 else "Outbreak"})',
                          bins=15)
        
        axes[1, 1].set_title('Risk Score Distribution by Actual Class')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_outbreak_risk(self, new_data):
        """
        Predict outbreak risk for new data
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Ensure new_data has the same features
        new_data_scaled = self.scaler.transform(new_data)
        
        # Get prediction and probability
        prediction = self.model.predict(new_data_scaled)
        probability = self.model.predict_proba(new_data_scaled)[:, 1]
        
        return prediction, probability

# Example usage and demonstration
def main():
    """
    Main function to demonstrate the disease outbreak prediction system
    """
    print("🏥 Disease Outbreak Prediction System for SDG 3")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DiseaseOutbreakPredictor()
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    df = predictor.load_and_preprocess_data()
    
    # Engineer features
    print("🔧 Engineering features...")
    df = predictor.engineer_features(df)
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train model
    print("🤖 Training machine learning model...")
    X_train, X_test, y_train, y_test = predictor.train_model(X, y)
    
    # Evaluate model
    print("📈 Evaluating model performance...")
    feature_importance = predictor.evaluate_model()
    
    # Visualize results
    print("📊 Creating visualizations...")
    predictor.visualize_results(feature_importance)
    
    # Example prediction for a new scenario
    print("\n🔮 Example Prediction for High-Risk Scenario:")
    high_risk_scenario = np.array([[
        500,    # population_density (high)
        35,     # temperature_avg (extreme)
        20,     # rainfall_mm (low)
        80,     # humidity_percent (high)
        3000,   # gdp_per_capita (low)
        0.5,    # hospital_beds_per_1000 (very low)
        30,     # vaccination_rate (low)
        2000,   # travel_volume (high)
        40,     # sanitation_index (low)
        50,     # education_index (medium)
        35 * 80,  # temp_humidity_interaction
        0.5 / 500 * 1000,  # healthcare_capacity
        (3000/1000 + 50 + 40) / 3,  # socioeconomic_health
        abs(35 - 25) + 20/100  # climate_stress
    ]])
    
    prediction, probability = predictor.predict_outbreak_risk(high_risk_scenario)
    print(f"Outbreak Risk: {'HIGH' if prediction[0] == 1 else 'LOW'}")
    print(f"Risk Probability: {probability[0]:.2%}")
    
    print("\n🌍 Impact on SDG 3: Good Health and Well-being")
    print("- Early warning system for disease outbreaks")
    print("- Helps allocate healthcare resources efficiently")  
    print("- Supports proactive public health interventions")
    print("- Reduces response time and saves lives")

if __name__ == "__main__":
    main()