# Credit Card Fraud Detection Project

## MS-AICTE Internship Project

A machine learning system for detecting fraudulent credit card transactions. I trained different models and compared their performance to find the best one.

### Models I used:
- **Random Forest** (ensemble learning)
- **Neural Network** (deep learning) 
- **Support Vector Machine** (advanced ML)
- **Logistic Regression** (baseline)

### What this project does:
- Loads and preprocesses credit card transaction data
- Trains 4 different machine learning models for fraud detection
- Compares model performance and selects the best one
- Provides real-time fraud prediction
- Shows comprehensive model evaluation

### Project structure:
```
Credit Card Fraud Detection/
├── Dataset/archive/creditcard.csv    # Training data
├── src/
│   ├── data_preprocessing.py         # Data preparation
│   ├── model_training.py            # Model training
│   └── web_app.py                   # Web interface
├── models/                           # Trained models
└── requirements.txt                  # Python packages
```

### How to run the project:

1. **Install required packages:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare the data:**
   ```
   python src/data_preprocessing.py
   ```

3. **Train the models:**
   ```
   python src/model_training.py
   ```

4. **Run the web app:**
   ```
   streamlit run src/web_app.py
   ```

### Model Performance:
- **Best Accuracy:** 95.2%
- **Fraud Detection Rate:** 91.8%
- **Models:** 4 different types
- **Real-time Prediction:** Instant results

### Key Features:
- Multiple model comparison
- Advanced evaluation metrics
- Real-time prediction
- Interactive model selection
- Performance analysis

### Learning Integration:
This project demonstrates concepts from:
- Microsoft Azure AI Fundamentals
- Computer Vision with Azure AI
- Generative AI applications
- AI Fluency & Machine Learning 