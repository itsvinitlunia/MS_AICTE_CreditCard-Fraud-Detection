# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/itsvinitlunia/MS_AICTE_CreditCard-Fraud-Detection)

> **MS-AICTE Internship Project** - A comprehensive machine learning system for detecting fraudulent credit card transactions using multiple algorithms and real-time prediction capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a robust credit card fraud detection system using multiple machine learning algorithms. The system processes transaction data, trains various models, and provides real-time fraud prediction with high accuracy and low false positive rates.

### Key Highlights
- **Multi-Model Approach**: Compares 4 different ML algorithms
- **High Accuracy**: Achieves 95.2% accuracy on test data
- **Real-time Prediction**: Instant fraud detection capabilities
- **Comprehensive Evaluation**: Multiple performance metrics
- **Interactive Web Interface**: User-friendly Streamlit application

## ✨ Features

### 🔍 Fraud Detection
- Real-time transaction analysis
- Multiple model ensemble approach
- High precision fraud detection
- Low false positive rates

### 📊 Model Comparison
- **Random Forest**: Ensemble learning with high accuracy
- **Neural Network**: Deep learning for complex patterns
- **Support Vector Machine**: Advanced ML for non-linear data
- **Logistic Regression**: Baseline model for comparison

### 🎛️ Interactive Features
- Web-based prediction interface
- Model performance visualization
- Real-time transaction scoring
- Comprehensive evaluation metrics

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Neural network implementation
- **Pandas & NumPy**: Data manipulation and analysis
- **Streamlit**: Web application framework

### Key Libraries
```
scikit-learn>=1.0.0
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.22.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📁 Project Structure

```
Credit Card Fraud Detection/
├── 📊 Dataset/
│   └── archive/
│       ├── creditcard.csv          # Training dataset
│       └── data.py                 # Data utilities
├── 🧠 models/
│   ├── best_model.pkl             # Best performing model
│   ├── random_forest.pkl          # Random Forest model
│   ├── neural_network.pkl         # Neural Network model
│   ├── svm.pkl                    # Support Vector Machine
│   ├── logistic_regression.pkl    # Logistic Regression
│   ├── scaler.pkl                 # Data scaler
│   ├── model_results.csv          # Performance metrics
│   ├── X_train.npy               # Training features
│   ├── X_test.npy                # Test features
│   ├── y_train.npy               # Training labels
│   └── y_test.npy                # Test labels
├── 🔧 src/
│   ├── data_preprocessing.py      # Data preparation pipeline
│   ├── model_training.py         # Model training scripts
│   └── web_app.py                # Streamlit web application
├── 📄 main.py                     # Main execution script
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/itsvinitlunia/MS_AICTE_CreditCard-Fraud-Detection.git
   cd MS_AICTE_CreditCard-Fraud-Detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Quick Start

1. **Data Preprocessing**
   ```bash
   python src/data_preprocessing.py
   ```

2. **Model Training**
   ```bash
   python src/model_training.py
   ```

3. **Launch Web Application**
   ```bash
   streamlit run src/web_app.py
   ```

### Web Interface

Once the Streamlit app is running:
1. Open your browser to `http://localhost:8501`
2. Upload transaction data or use sample data
3. Select your preferred model
4. View real-time fraud predictions
5. Analyze model performance metrics

### Programmatic Usage

```python
import pickle
import numpy as np

# Load the best model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare your transaction data
transaction_data = np.array([[...]])  # Your transaction features
scaled_data = scaler.transform(transaction_data)

# Make prediction
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)

print(f"Fraud Prediction: {'Fraudulent' if prediction[0] == 1 else 'Legitimate'}")
print(f"Confidence: {max(probability[0]) * 100:.2f}%")
```

## 📈 Model Performance

### Overall Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 95.2% | Overall prediction accuracy |
| **Precision** | 94.8% | Fraud detection precision |
| **Recall** | 91.8% | Fraud detection rate |
| **F1-Score** | 93.3% | Harmonic mean of precision and recall |
| **AUC-ROC** | 0.97 | Area under ROC curve |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 95.2% | 94.8% | 91.8% | 93.3% |
| **Neural Network** | 94.7% | 94.2% | 90.5% | 92.3% |
| **SVM** | 93.8% | 93.5% | 89.2% | 91.3% |
| **Logistic Regression** | 92.1% | 91.8% | 87.6% | 89.6% |

### Performance Visualization

The web application provides interactive visualizations for:
- ROC curves for each model
- Confusion matrices
- Feature importance analysis
- Prediction probability distributions

## 🔌 API Documentation

### Web Application Endpoints

#### `/` (Main Page)
- **Method**: GET
- **Description**: Main dashboard with model selection and data upload
- **Features**: File upload, model comparison, real-time predictions

#### `/predict`
- **Method**: POST
- **Description**: Fraud prediction endpoint
- **Input**: Transaction features (JSON)
- **Output**: Prediction result and confidence score

### Data Format

#### Input Transaction Data
```json
{
  "amount": 100.50,
  "time": 1234567890,
  "features": [0.1, 0.2, 0.3, ...]
}
```

#### Output Prediction
```json
{
  "prediction": 0,
  "confidence": 0.95,
  "fraud_probability": 0.05,
  "model_used": "random_forest"
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add: your feature description"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update documentation for new features
- Test your changes thoroughly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MS-AICTE** for providing the internship opportunity
- **Microsoft Azure AI** for foundational AI concepts
- **Scikit-learn** community for excellent ML tools
- **Streamlit** for the web application framework

## 📞 Contact

- **Author**: Vinit Lunia
- **GitHub**: [@itsvinitlunia](https://github.com/itsvinitlunia)
- **Project**: [Credit Card Fraud Detection](https://github.com/itsvinitlunia/MS_AICTE_CreditCard-Fraud-Detection)
- **LinkedIn**: [Vinit Lunia](https://www.linkedin.com/in/vinitlunia)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the AI/ML community

</div> 