"""
Credit Card Fraud Detection - Main Script
My MS-AICTE internship project - runs everything
"""

import os
import sys

def run_credit_card_fraud_detection_project():
    """Main function - runs the whole project"""
    print("=== CREDIT CARD FRAUD DETECTION PROJECT ===")
    print("My MS-AICTE internship project")
    print()
    
    print("This project will:")
    print("1. Prepare the credit card data")
    print("2. Train different machine learning models")
    print("3. Compare model performance")
    print("4. Start the web app for fraud detection")
    print()
    
    # Check if data exists
    if not os.path.exists("Dataset/archive/creditcard.csv"):
        print("Error: Credit card data not found!")
        print("Please make sure the dataset is in Dataset/archive/creditcard.csv")
        return
    
    # Step 1: Data preprocessing
    print("Step 1: Preparing the data...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "src/data_preprocessing.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Data preparation completed successfully!")
        else:
            print("Error in data preparation:", result.stderr)
            return
    except Exception as error:
        print(f"Error running data preprocessing: {error}")
        return
    
    print()
    
    # Step 2: Model training
    print("Step 2: Training the models...")
    try:
        result = subprocess.run([sys.executable, "src/model_training.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Model training completed successfully!")
        else:
            print("Error in model training:", result.stderr)
            return
    except Exception as error:
        print(f"Error running model training: {error}")
        return
    
    print()
    
    # Step 3: Start web app
    print("Step 3: Starting the web app...")
    print("The web app will open in your browser.")
    print("You can use it to detect fraud in real-time!")
    print()
    
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/web_app.py"])
    except Exception as error:
        print(f"Error starting web app: {error}")
        print("You can manually start it with: streamlit run src/web_app.py")

if __name__ == "__main__":
    run_credit_card_fraud_detection_project() 