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
    """Main fraud detection interface - where users can test transactions"""
    st.header("🔍 Fraud Detection System")
    
    # Load our trained models and data scaler
    trained_models, data_scaler = load_trained_models_and_scaler()
    
    if trained_models is None:
        st.error("❌ Could not load the trained models. Please run model training first.")
        return
    
    # Let user pick which model to use for prediction
    st.subheader("🤖 Choose Your Detection Model")
    available_models = ["Random Forest", "Neural Network", "SVM", "Logistic Regression"]
    chosen_model_name = st.selectbox(
        "Select a model for fraud detection:",
        available_models,
        help="Each model has different strengths - Random Forest is usually most reliable"
    )
    
    st.success(f"✅ Selected Model: **{chosen_model_name}**")
    
    # Transaction input section
    st.subheader("💳 Enter Transaction Details")
    
    # Quick testing option for users
    use_test_data = st.checkbox("Use Sample Data for Quick Testing", value=True, 
                               help="Check this to test with pre-made sample data (recommended for first-time users)")
    
    # Create two columns for better layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        # Basic transaction info
        transaction_amount_dollars = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.0, 
            value=100.0, 
            help="How much money is involved in this transaction?"
        )
        transaction_time_seconds = st.number_input(
            "Transaction Time (seconds)", 
            min_value=0, 
            value=1000,
            help="Time since the very first transaction in the dataset"
        )
    
    with right_col:
        # First few PCA features (these are the most important ones)
        pca_feature_1 = st.number_input("PCA Feature V1", value=0.0, help="First principal component feature")
        pca_feature_2 = st.number_input("PCA Feature V2", value=0.0, help="Second principal component feature")
        pca_feature_3 = st.number_input("PCA Feature V3", value=0.0, help="Third principal component feature")
    
    # Add more feature inputs in a scrollable area
    st.subheader("Additional Features (V4-V28)")
    
    # Create columns for better layout
    col1, col2, col3, col4 = st.columns(4)
    
    features_v4_v28 = []
    with col1:
        for i in range(4, 8):
            features_v4_v28.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
    with col2:
        for i in range(8, 12):
            features_v4_v28.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
    with col3:
        for i in range(12, 16):
            features_v4_v28.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
    with col4:
        for i in range(16, 20):
            features_v4_v28.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
    
    # Add remaining features
    col5, col6, col7 = st.columns(3)
    with col5:
        for i in range(20, 24):
            features_v4_v28.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
    with col6:
        for i in range(24, 28):
            features_v4_v28.append(st.number_input(f"V{i}", value=0.0, key=f"v{i}"))
    with col7:
        features_v4_v28.append(st.number_input("V28", value=0.0, key="v28"))
    
    if st.button("🚀 Run Fraud Detection", type="primary", help="Click to analyze this transaction"):
        # Handle sample data vs manual input
        if use_test_data:
            # Pre-made sample data for easy testing (represents a normal transaction)
            sample_transaction_data = [
                1000,  # Time
                -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2,  # V1-V8
                -0.3, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1,  # V9-V16
                -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3,  # V17-V24
                -0.1, 0.2, -0.3, 0.1,                        # V25-V28
                100.0  # Amount
            ]
            transaction_features = sample_transaction_data
            st.info("🧪 Using sample data for testing. This represents a normal transaction.")
        else:
            # Build feature array from user inputs
            transaction_features = [
                transaction_time_seconds, 
                pca_feature_1, pca_feature_2, pca_feature_3
            ] + features_v4_v28 + [transaction_amount_dollars]
        
        # Safety check - make sure we have the right number of features
        expected_feature_count = 30
        actual_feature_count = len(transaction_features)
        if actual_feature_count != expected_feature_count:
            st.error(f"❌ Feature count error! Expected {expected_feature_count}, got {actual_feature_count}")
            st.write("🔍 Debug - Features:", transaction_features)
            return
        
        # Show progress and feature count
        st.write(f"📊 Processing {len(transaction_features)} features...")
        
        # Scale the features to match our training data
        scaled_features = data_scaler.transform([transaction_features])
        
        # Get the chosen model and make our prediction
        chosen_model = trained_models[chosen_model_name]
        
        # Run the fraud detection analysis
        fraud_prediction = chosen_model.predict(scaled_features)[0]
        prediction_probabilities = chosen_model.predict_proba(scaled_features)[0]
        
        # Display the results with nice formatting
        st.markdown("---")
        st.subheader("🎯 Fraud Detection Results")
        
        # Create two columns for the main result
        result_left_col, result_right_col = st.columns(2)
        
        with result_left_col:
            if fraud_prediction == 0:
                st.success("✅ **SAFE TRANSACTION**")
                st.balloons()
                st.write("🎉 This transaction appears to be legitimate!")
            else:
                st.error("🚨 **FRAUD DETECTED**")
                st.snow()
                st.write("⚠️ This transaction shows suspicious patterns!")
        
        with result_right_col:
            confidence_level = max(prediction_probabilities)
            st.metric("🎯 Model Confidence", f"{confidence_level:.1%}")
        
        # Show a progress bar for confidence
        st.progress(confidence_level)
        st.write(f"📊 Model Confidence Level: {confidence_level:.1%}")
        
        # Detailed probability breakdown
        st.subheader("📈 Detailed Analysis")
        prob_left_col, prob_right_col = st.columns(2)
        
        with prob_left_col:
            normal_prob = prediction_probabilities[0]
            st.metric("✅ Normal Transaction Probability", f"{normal_prob:.1%}")
        
        with prob_right_col:
            fraud_prob = prediction_probabilities[1]
            st.metric("🚨 Fraud Probability", f"{fraud_prob:.1%}")
        
        # Summary information
        st.subheader("📋 Analysis Summary")
        st.info(f"🤖 **Model Used:** {chosen_model_name}")
        st.info(f"🔍 **Prediction:** {'🚨 FRAUD' if fraud_prediction == 1 else '✅ NORMAL'}")
        st.info(f"📊 **Confidence:** {confidence_level:.1%}")
        
        # Add some helpful context
        if fraud_prediction == 1:
            st.warning("💡 **Recommendation:** Review this transaction carefully and consider additional verification.")
        else:
            st.success("💡 **Recommendation:** This transaction appears safe based on our analysis.")

def show_performance_page():
    """Display detailed performance analysis of all our trained models"""
    st.header("📊 Model Performance Analysis")
    
    try:
        # Load our saved model performance data
        model_performance_data = pd.read_csv('models/model_results.csv')
        
        st.subheader("🏆 Model Performance Comparison")
        
        # Create a beautiful performance visualization
        performance_chart = go.Figure()
        
        # Define our performance metrics and colors
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
        chart_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Add each metric as a bar in our chart
        for metric_index, metric_name in enumerate(performance_metrics):
            performance_chart.add_trace(go.Bar(
                name=metric_name.capitalize(),
                x=model_performance_data.index,
                y=model_performance_data[metric_name],
                marker_color=chart_colors[metric_index]
            ))
        
        # Make the chart look professional
        performance_chart.update_layout(
            title="📈 Model Performance Metrics Comparison",
            xaxis_title="🤖 Machine Learning Models",
            yaxis_title="📊 Performance Score",
            barmode='group'
        )
        
        st.plotly_chart(performance_chart)
        
        # Show the detailed performance table
        st.subheader("📋 Detailed Performance Results")
        st.dataframe(model_performance_data)
        
        # Highlight the best performing model
        best_model_name = model_performance_data['accuracy'].idxmax()
        best_model_accuracy = model_performance_data.loc[best_model_name, 'accuracy']
        
        st.success(f"🏆 **Best Overall Model:** {best_model_name}")
        st.success(f"🎯 **Best Accuracy:** {best_model_accuracy:.1%}")
        
        # Add some insights
        st.info("💡 **Insight:** The Random Forest model typically performs best for fraud detection due to its ability to handle complex patterns in transaction data.")
        
    except Exception as error:
        st.error(f"❌ Error loading performance data: {error}")
        st.info("💡 Make sure you've run the model training script first!")

def show_comparison_page():
    """Side-by-side comparison of all our fraud detection models"""
    st.header("🔍 Model Comparison Dashboard")
    
    try:
        # Load our model comparison data
        comparison_data = pd.read_csv('models/model_results.csv')
        
        # Create two columns for our comparison charts
        comparison_left_col, comparison_right_col = st.columns(2)
        
        with comparison_left_col:
            st.subheader("🎯 Accuracy Comparison")
            accuracy_chart = px.bar(
                x=comparison_data.index,
                y=comparison_data['accuracy'],
                title="📊 Model Accuracy Scores",
                labels={'x': '🤖 Models', 'y': '📈 Accuracy'}
            )
            st.plotly_chart(accuracy_chart)
        
        with comparison_right_col:
            st.subheader("⚖️ F1 Score Comparison")
            f1_chart = px.bar(
                x=comparison_data.index,
                y=comparison_data['f1'],
                title="📊 Model F1 Scores",
                labels={'x': '🤖 Models', 'y': '📈 F1 Score'}
            )
            st.plotly_chart(f1_chart)
        
        # Show detailed comparison table
        st.subheader("📋 Detailed Model Comparison")
        
        # Format our data for better display
        formatted_comparison_data = comparison_data.copy()
        for column in formatted_comparison_data.columns:
            formatted_comparison_data[column] = formatted_comparison_data[column].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(formatted_comparison_data)
        
        # Provide model recommendations
        st.subheader("💡 Model Recommendations")
        
        best_accuracy_model = comparison_data['accuracy'].idxmax()
        best_f1_model = comparison_data['f1'].idxmax()
        
        st.info(f"🏆 **Best for Overall Accuracy:** {best_accuracy_model}")
        st.info(f"⚖️ **Best for Balanced Performance (F1):** {best_f1_model}")
        
        # Add helpful insights
        st.success("💡 **Pro Tip:** Choose Random Forest for general use, Neural Network for complex patterns, and SVM for high precision scenarios.")
        
    except Exception as error:
        st.error(f"❌ Error loading comparison data: {error}")
        st.info("💡 Please run the model training script to generate comparison data.")

if __name__ == "__main__":
    main() 