import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Algorithm Selector", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Algorithm Selector")

st.markdown("Upload your CSV file and select an algorithm to train with GridSearchCV")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar for file upload and configuration
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

if st.session_state.data_loaded:
    df = st.session_state.df
    
    # Display data preview
    st.subheader("📊 Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
    with col2:
        st.dataframe(df.head(), use_container_width=True)
    
    # Column information
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
    
    st.markdown("---")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox("Select Target Column", df.columns.tolist())
    
    with col2:
        algorithm = st.selectbox(
            "Select Algorithm",
            [
                "SVM (Support Vector Machine)",
                "Neural Network",
                "Decision Tree",
                "Bayesian (Naive Bayes)",
                "KNN (K-Nearest Neighbors)",
                "Linear Regression",
                "Logistic Regression"
            ]
        )
    
    # Prepare data function
    def prepare_data(df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            task_type = 'classification'
        else:
            unique_ratio = len(y.unique()) / len(y)
            task_type = 'classification' if unique_ratio < 0.05 else 'regression'
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, task_type
    