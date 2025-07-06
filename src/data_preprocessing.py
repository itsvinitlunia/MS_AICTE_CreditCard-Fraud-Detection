"""
Credit Card Fraud Detection - Data Preprocessing
My MS-AICTE internship project for fraud detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

def load_credit_card_data(file_path):
    """Load the credit card data from CSV"""
    print("Loading the credit card dataset...")
    data_frame = pd.read_csv(file_path)
    print(f"Got {len(data_frame)} transactions with {len(data_frame.columns)} features")
    return data_frame

def explore_credit_card_data(data_frame):
    """Take a look at what we're working with"""
    print("\n=== DATA EXPLORATION ===")
    print(f"Total transactions: {len(data_frame)}")
    print(f"Features available: {len(data_frame.columns)}")
    print(f"Missing data: {data_frame.isnull().sum().sum()}")
    
    # Check how many fraud vs normal transactions we have
    fraud_count = len(data_frame[data_frame['Class'] == 1])
    normal_count = len(data_frame[data_frame['Class'] == 0])
    fraud_ratio = (fraud_count/len(data_frame))*100
    
    print(f"Normal transactions: {normal_count}")
    print(f"Fraud transactions: {fraud_count}")
    print(f"Fraud percentage: {fraud_ratio:.2f}%")
    
    # This is pretty imbalanced data - typical for fraud detection

def create_data_visualizations(data_frame):
    """Make some plots to understand the data better"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: How many normal vs fraud transactions
    axes[0,0].hist(data_frame['Class'], bins=2, alpha=0.7)
    axes[0,0].set_title('Normal vs Fraud Transactions')
    axes[0,0].set_xlabel('Class (0=Normal, 1=Fraud)')
    axes[0,0].set_ylabel('Number of Transactions')
    
    # Plot 2: Distribution of transaction amounts
    axes[0,1].hist(data_frame['Amount'], bins=50, alpha=0.7, color='green')
    axes[0,1].set_title('Transaction Amounts Distribution')
    axes[0,1].set_xlabel('Amount ($)')
    axes[0,1].set_ylabel('Frequency')
    
    # Plot 3: Compare amounts between normal and fraud
    normal_amounts = data_frame[data_frame['Class'] == 0]['Amount']
    fraud_amounts = data_frame[data_frame['Class'] == 1]['Amount']
    
    axes[1,0].boxplot([normal_amounts, fraud_amounts], labels=['Normal', 'Fraud'])
    axes[1,0].set_title('Amount Comparison: Normal vs Fraud')
    axes[1,0].set_ylabel('Amount ($)')
    
    # Plot 4: When do transactions happen
    axes[1,1].hist(data_frame['Time'], bins=50, alpha=0.7, color='orange')
    axes[1,1].set_title('Transaction Timing')
    axes[1,1].set_xlabel('Time (seconds)')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data_plots.png')
    plt.show()
    print("Saved plots as 'data_plots.png'")

def prepare_data_for_training(data_frame):
    """Get the data ready for machine learning"""
    print("\n=== PREPARING DATA FOR TRAINING ===")
    
    # Split features from target variable
    features = data_frame.drop('Class', axis=1)
    labels = data_frame['Class']
    
    # Split into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Scale the features so they're all on the same scale
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)
    
    print(f"Training set size: {features_train.shape}")
    print(f"Test set size: {features_test.shape}")
    
    return features_train_scaled, features_test_scaled, labels_train, labels_test, scaler

def save_prepared_data(features_train, features_test, labels_train, labels_test, scaler):
    """Save all the processed data for later use"""
    print("\n=== SAVING PROCESSED DATA ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save as numpy arrays (much faster than CSV for large data)
    np.save('models/X_train.npy', features_train)
    np.save('models/X_test.npy', features_test)
    np.save('models/y_train.npy', labels_train.values)
    np.save('models/y_test.npy', labels_test.values)
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("All data saved to models folder (using numpy arrays for speed)")

def main():
    """Main function - runs the whole data preparation process"""
    print("=== CREDIT CARD FRAUD DETECTION - DATA PREP ===")
    
    # Load the data
    data_frame = load_credit_card_data("Dataset/archive/creditcard.csv")
    
    # Explore what we have
    explore_credit_card_data(data_frame)
    
    # Create some visualizations
    create_data_visualizations(data_frame)
    
    # Prepare data for training
    features_train, features_test, labels_train, labels_test, scaler = prepare_data_for_training(data_frame)
    
    # Save everything
    save_prepared_data(features_train, features_test, labels_train, labels_test, scaler)
    
    print("\n=== DATA PREPARATION COMPLETE ===")

if __name__ == "__main__":
    main() 