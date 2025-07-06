"""
Credit Card Fraud Detection - Model Training
My MS-AICTE internship project - training different ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from sklearn.model_selection import train_test_split

def load_prepared_training_data():
    """Load the data we prepared earlier"""
    print("Loading the processed training data...")
    
    features_train = np.load('models/X_train.npy')
    features_test = np.load('models/X_test.npy')
    labels_train = np.load('models/y_train.npy')
    labels_test = np.load('models/y_test.npy')
    
    print(f"Training data: {features_train.shape}")
    print(f"Test data: {features_test.shape}")
    
    return features_train, features_test, labels_train, labels_test

def train_all_models(features_train, labels_train):
    """Train different machine learning models"""
    print("\n=== TRAINING MACHINE LEARNING MODELS ===")
    
    trained_models = {}
    
    # 1. Random Forest - good for this type of problem
    print("Training Random Forest model...")
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(features_train, labels_train)
    trained_models['Random Forest'] = random_forest_model
    print("Random Forest training completed!")
    
    # 2. Neural Network - let's try deep learning
    print("Training Neural Network model...")
    neural_network_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    neural_network_model.fit(features_train, labels_train)
    trained_models['Neural Network'] = neural_network_model
    print("Neural Network training completed!")
    
    # 3. Support Vector Machine - use smaller subset for speed
    print("Training Support Vector Machine model...")
    print("Using smaller subset for faster training...")
    
    # Use only 10% of data for SVM to make it faster
    features_svm, _, labels_svm, _ = train_test_split(features_train, labels_train, test_size=0.9, random_state=42, stratify=labels_train)
    
    svm_model = SVC(probability=True, random_state=42, kernel='rbf', C=1.0)
    svm_model.fit(features_svm, labels_svm)
    trained_models['SVM'] = svm_model
    print("SVM training completed!")
    
    # 4. Logistic Regression - simple baseline
    print("Training Logistic Regression model...")
    logistic_regression_model = LogisticRegression(random_state=42, max_iter=1000)
    logistic_regression_model.fit(features_train, labels_train)
    trained_models['Logistic Regression'] = logistic_regression_model
    print("Logistic Regression training completed!")
    
    return trained_models

def evaluate_all_models(trained_models, features_test, labels_test):
    """See how well each model performs"""
    print("\n=== MODEL EVALUATION ===")
    
    model_performance = {}
    
    for model_name, model in trained_models.items():
        print(f"\n--- {model_name} Results ---")
        
        # Make predictions
        predictions = model.predict(features_test)
        probabilities = model.predict_proba(features_test)[:, 1]
        
        # Calculate how well it did
        accuracy = accuracy_score(labels_test, predictions)
        precision = precision_score(labels_test, predictions)
        recall = recall_score(labels_test, predictions)
        f1 = f1_score(labels_test, predictions)
        
        # Store the results
        model_performance[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        # Show the results
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Show detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(labels_test, predictions))
    
    return model_performance

def create_comparison_plots(model_performance, labels_test, trained_models):
    """Create plots to compare all models"""
    print("\n=== CREATING MODEL COMPARISON PLOTS ===")
    
    # 1. Compare accuracy across models
    model_names = list(model_performance.keys())
    accuracies = [model_performance[model]['accuracy'] for model in model_names]
    
    plt.figure(figsize=(15, 10))
    
    # Performance comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Confusion matrix for best model
    plt.subplot(2, 2, 2)
    best_model_name = max(model_performance.keys(), key=lambda x: model_performance[x]['accuracy'])
    cm = confusion_matrix(labels_test, model_performance[best_model_name]['predictions'])
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add numbers in the confusion matrix
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    # 3. Compare all metrics
    plt.subplot(2, 2, 3)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [model_performance[model][metric] for model in model_names]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Metrics Comparison')
    plt.xticks(x + width*1.5, model_names, rotation=45)
    plt.legend()
    
    # 4. Feature importance for Random Forest
    plt.subplot(2, 2, 4)
    if 'Random Forest' in trained_models:
        rf_model = trained_models['Random Forest']
        if hasattr(rf_model, 'feature_importances_'):
            feature_importance = rf_model.feature_importances_
            top_features = np.argsort(feature_importance)[-10:]  # Top 10 features
            plt.barh(range(len(top_features)), feature_importance[top_features])
            plt.title('Top 10 Feature Importance (Random Forest)')
            plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Model comparison saved as 'model_comparison.png'")

def save_all_models(trained_models, model_performance):
    """Save all the trained models"""
    print("\n=== SAVING TRAINED MODELS ===")
    
    # Save each model
    for model_name, model in trained_models.items():
        filename = f'models/{model_name.lower().replace(" ", "_")}.pkl'
        joblib.dump(model, filename)
        print(f"Saved {model_name} to {filename}")
    
    # Save the results summary
    results_df = pd.DataFrame(model_performance).T
    results_df = results_df[['accuracy', 'precision', 'recall', 'f1']]
    results_df.to_csv('models/model_results.csv')
    
    # Find and save the best model
    best_model_name = max(model_performance.keys(), key=lambda x: model_performance[x]['accuracy'])
    best_accuracy = model_performance[best_model_name]['accuracy']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Save the best model separately
    joblib.dump(trained_models[best_model_name], 'models/best_model.pkl')
    print(f"Best model saved as 'models/best_model.pkl'")

def main():
    """Main function - runs the whole model training process"""
    print("=== CREDIT CARD FRAUD DETECTION - MODEL TRAINING ===")
    
    # Load the data
    features_train, features_test, labels_train, labels_test = load_prepared_training_data()
    
    # Train different models
    trained_models = train_all_models(features_train, labels_train)
    
    # Evaluate how well they did
    model_performance = evaluate_all_models(trained_models, features_test, labels_test)
    
    # Create comparison plots
    create_comparison_plots(model_performance, labels_test, trained_models)
    
    # Save everything
    save_all_models(trained_models, model_performance)
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print("All models trained and evaluated")
    print("Best model selected and saved")
    print("Ready for fraud detection!")

if __name__ == "__main__":
    main() 