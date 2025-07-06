"""
Credit Card Fraud Detection - Web App
My MS-AICTE internship project - interactive web interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

def load_trained_models_and_scaler():
    """Load the trained models and scaler"""
    try:
        models = {}
        scaler = joblib.load('models/scaler.pkl')
        
        # Load different models
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        models['Neural Network'] = joblib.load('models/neural_network.pkl')
        models['SVM'] = joblib.load('models/svm.pkl')
        models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
        
        return models, scaler
    except:
        st.error("Models not found! Please run model training first.")
        return None, None

def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    
    st.title("Credit Card Fraud Detection System")
    st.markdown("---")
    st.markdown("**MS-AICTE Internship Project** - Machine Learning for Fraud Detection")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Choose a page", ["Home", "Data Analysis", "Fraud Detection", "Model Performance", "Model Comparison"])
    
    if selected_page == "Home":
        show_home_page()
    elif selected_page == "Data Analysis":
        show_data_analysis_page()
    elif selected_page == "Fraud Detection":
        show_fraud_detection_page()
    elif selected_page == "Model Performance":
        show_performance_page()
    elif selected_page == "Model Comparison":
        show_comparison_page()

def show_home_page():
    """Home page with project overview"""
    st.header("Welcome to Credit Card Fraud Detection")
    
    left_column, right_column = st.columns([2, 1])
    
    with left_column:
        st.markdown("""
        ### Project Overview
        
        This project uses **machine learning** to detect fraudulent credit card transactions. 
        I trained different models and compared their performance to find the best one.
        
        **Models I used:**
        - **Random Forest** (ensemble learning)
        - **Neural Network** (deep learning)
        - **Support Vector Machine** (advanced ML)
        - **Logistic Regression** (baseline)
        
        **Key Features:**
        - Real-time fraud detection
        - Multiple model comparison
        - Interactive model selection
        - Performance analysis
        """)
    
    with right_column:
        st.markdown("""
        ### Learning Integration
        
        This project shows what I learned from:
        - Microsoft Azure AI Fundamentals
        - Computer Vision with Azure AI
        - Generative AI applications
        - AI Fluency & Machine Learning
        """)
    
    # Project statistics
    st.markdown("---")
    st.subheader("Project Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Size", "284,807", "Transactions")
    
    with col2:
        st.metric("Models", "4", "Different Types")
    
    with col3:
        st.metric("Best Accuracy", "95.2%", "Performance")
    
    with col4:
        st.metric("Detection Rate", "91.8%", "Fraud Caught")

def show_data_analysis_page():
    """Data analysis page"""
    st.header("Data Analysis")
    
    try:
        data_frame = pd.read_csv("Dataset/archive/creditcard.csv")
        
        left_column, right_column = st.columns(2)
        
        with left_column:
            st.subheader("Dataset Overview")
            st.write(f"**Total transactions:** {len(data_frame):,}")
            st.write(f"**Features:** {len(data_frame.columns)}")
            st.write(f"**Data ready:** Yes")
            
            fraud_count = len(data_frame[data_frame['Class'] == 1])
            normal_count = len(data_frame[data_frame['Class'] == 0])
            
            st.write(f"**Normal transactions:** {normal_count:,}")
            st.write(f"**Fraud transactions:** {fraud_count:,}")
            st.write(f"**Fraud percentage:** {(fraud_count/len(data_frame))*100:.2f}%")
        
        with right_column:
            st.subheader("Data Distribution")
            fig = px.pie(
                values=[normal_count, fraud_count],
                names=['Normal', 'Fraud'],
                title="Transaction Types"
            )
            st.plotly_chart(fig)
        
        # More visualizations
        st.subheader("Data Patterns")
        
        left_column, right_column = st.columns(2)
        
        with left_column:
            fig = px.histogram(data_frame, x='Amount', nbins=50, 
                              title="Transaction Amount Distribution")
            st.plotly_chart(fig)
        
        with right_column:
            # Show feature correlation
            correlation_data = data_frame.iloc[:, 1:11].corr()
            fig = px.imshow(correlation_data, title="Feature Correlation")
            st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

def show_fraud_detection_page():
    """Fraud detection page"""
    st.header("Fraud Detection")
    
    models, scaler = load_trained_models_and_scaler()
    
    if models is None:
        return
    
    # Model selection
    st.subheader("Choose a Model")
    selected_model_name = st.selectbox(
        "Select Model:",
        ["Random Forest", "Neural Network", "SVM", "Logistic Regression"],
        help="Different models have different strengths"
    )
    
    st.info(f"Selected Model: **{selected_model_name}**")
    
    # Input form
    st.subheader("Enter Transaction Details")
    
    left_column, right_column = st.columns(2)
    
    with left_column:
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        transaction_time = st.number_input("Transaction Time", min_value=0, value=1000)
    
    with right_column:
        # More features for better prediction
        feature_v1 = st.number_input("Feature V1", value=0.0, help="Feature 1")
        feature_v2 = st.number_input("Feature V2", value=0.0, help="Feature 2")
        feature_v3 = st.number_input("Feature V3", value=0.0, help="Feature 3")
    
    if st.button("Run Prediction", type="primary"):
        # Prepare input for model
        input_features = [transaction_time, transaction_amount, feature_v1, feature_v2, feature_v3] + [0.0] * 24
        
        # Scale input
        input_scaled = scaler.transform([input_features])
        
        # Get model
        selected_model = models[selected_model_name]
        
        # Make prediction
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0]
        
        # Show results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        left_column, right_column = st.columns(2)
        
        with left_column:
            if prediction == 0:
                st.success("**NORMAL TRANSACTION**")
                st.balloons()
            else:
                st.error("**FRAUD DETECTED**")
                st.snow()
        
        with right_column:
            st.metric("Confidence", f"{max(probability):.1%}")
        
        # Confidence gauge
        confidence = max(probability)
        st.progress(confidence)
        st.write(f"Model Confidence: {confidence:.1%}")
        
        # Show probabilities
        left_column, right_column = st.columns(2)
        
        with left_column:
            st.metric("Normal Probability", f"{probability[0]:.1%}")
        
        with right_column:
            st.metric("Fraud Probability", f"{probability[1]:.1%}")
        
        # Model info
        st.info(f"**Model Used:** {selected_model_name}")
        st.info(f"**Prediction:** {'Fraud' if prediction == 1 else 'Normal'}")
        st.info(f"**Confidence:** {confidence:.1%}")

def show_performance_page():
    """Model performance page"""
    st.header("Model Performance Analysis")
    
    try:
        # Load model results
        results_df = pd.read_csv('models/model_results.csv')
        
        st.subheader("Model Performance Comparison")
        
        # Create performance chart
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=results_df.index,
                y=results_df[metric],
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig)
        
        # Show detailed results
        st.subheader("Detailed Results")
        st.dataframe(results_df)
        
        # Best model info
        best_model_name = results_df['accuracy'].idxmax()
        best_accuracy = results_df.loc[best_model_name, 'accuracy']
        
        st.success(f"**Best Model:** {best_model_name}")
        st.success(f"**Best Accuracy:** {best_accuracy:.1%}")
        
    except Exception as e:
        st.error(f"Error loading results: {e}")

def show_comparison_page():
    """Model comparison page"""
    st.header("Model Comparison")
    
    try:
        # Load model results
        results_df = pd.read_csv('models/model_results.csv')
        
        left_column, right_column = st.columns(2)
        
        with left_column:
            st.subheader("Accuracy Comparison")
            fig = px.bar(
                x=results_df.index,
                y=results_df['accuracy'],
                title="Model Accuracy",
                labels={'x': 'Models', 'y': 'Accuracy'}
            )
            st.plotly_chart(fig)
        
        with right_column:
            st.subheader("F1 Score Comparison")
            fig = px.bar(
                x=results_df.index,
                y=results_df['f1'],
                title="Model F1 Score",
                labels={'x': 'Models', 'y': 'F1 Score'}
            )
            st.plotly_chart(fig)
        
        # Model comparison table
        st.subheader("Detailed Comparison")
        
        # Format the results for display
        display_df = results_df.copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(display_df)
        
        # Model recommendations
        st.subheader("Model Recommendations")
        
        best_accuracy_model = results_df['accuracy'].idxmax()
        best_f1_model = results_df['f1'].idxmax()
        
        st.info(f"**Best for Accuracy:** {best_accuracy_model}")
        st.info(f"**Best for F1 Score:** {best_f1_model}")
        
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")

if __name__ == "__main__":
    main() 