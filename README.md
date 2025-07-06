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

### Dataset Structure
The credit card dataset contains 30 features:
- **Time**: Time elapsed between transactions
- **V1-V28**: PCA-transformed features (anonymized for privacy)
- **Amount**: Transaction amount in dollars
- **Class**: Target variable (0=Normal, 1=Fraud)

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
├── 📖 README.md                   # Project documentation
├── 📊 Output/                     # Web app screenshots and outputs
│   ├── Home Page.png             # Main dashboard screenshot
│   ├── Home page 2.png           # Additional home page view
│   ├── Data Analysis.png         # Data analysis interface
│   ├── Model Performance.png     # Model performance dashboard
│   ├── Model Comparison.png      # Model comparison view
│   └── Fraud Detection System.png # Fraud detection interface
├── 📈 model_comparison.png       # Model comparison visualization
├── 📊 Figure_1.png              # Data visualization plots
└── 📊 data_plots.png            # Data exploration plots
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
2. Navigate to the "Fraud Detection" page
3. Choose between manual input or sample data testing
4. If using manual input, provide all 30 features:
   - **Time**: Transaction time in seconds
   - **Amount**: Transaction amount in dollars
   - **V1-V28**: PCA-transformed features (can be set to 0.0 for testing)
5. Select your preferred model
6. View real-time fraud predictions with confidence scores
7. Analyze model performance metrics

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

### 📸 Visual Outputs

The project includes comprehensive visual documentation:

#### Web Application Screenshots (`Output/` folder)
- **Home Page**: Main dashboard with navigation and overview
- **Data Analysis**: Interactive data exploration interface
- **Model Performance**: Detailed performance metrics and charts
- **Model Comparison**: Side-by-side model comparison
- **Fraud Detection**: Real-time prediction interface

#### Generated Visualizations
- **Model Comparison**: Performance comparison across all models
- **Data Plots**: Transaction distribution and fraud analysis
- **Figure_1**: Additional data exploration visualizations

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

## 🔧 Troubleshooting

### Common Issues

#### Feature Count Mismatch Error
If you encounter: `ValueError: X has 29 features, but StandardScaler is expecting 30 features`
- **Solution**: Ensure you provide all 30 features in the correct order
- **Order**: [Time, V1, V2, V3, ..., V28, Amount]
- **Use Sample Data**: Check "Use Sample Data for Testing" in the web interface

#### Model Loading Issues
- Ensure all model files are present in the `models/` directory
- Run `python src/model_training.py` to regenerate models if needed

#### Web App Issues
- Check that Streamlit is installed: `pip install streamlit`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## 🤝 Contributing

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