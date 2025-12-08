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
    
    #nbdaw nchwiya sahlin
    def run_decision_tree(X, y, task_type):
        param_grid = {
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'] if task_type == 'classification' else ['squared_error', 'absolute_error']
        }
        
        if task_type == 'classification':
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if task_type == 'classification' else 'r2', n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def run_bayesian(X, y, task_type):
        if task_type != 'classification':
            return None, None, None
        
        param_grid = {
            'var_smoothing': np.logspace(-10, -8, 10)
        }
        
        model = GaussianNB()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def run_knn(X, y, task_type):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        if task_type == 'classification':
            model = KNeighborsClassifier()
        else:
            model = KNeighborsRegressor()
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if task_type == 'classification' else 'r2', n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def run_linear_regression(X, y, task_type):
        if task_type != 'regression':
            return None, None, None
        
        param_grid = {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
        
        model = LinearRegression()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def run_logistic_regression(X, y, task_type):
        if task_type != 'classification':
            return None, None, None
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear', 'newton-cg'],
            'max_iter': [1000]
        }
        
        model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    # Train button
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model with GridSearchCV... This may take a few minutes."):
            try:
                # Prepare data
                X, y, task_type = prepare_data(df, target_column)
                
                st.info(f"📌 Task Type Detected: **{task_type.upper()}**")
                
                # Algorithm mapping
                algorithms = {
                    "Decision Tree": run_decision_tree,
                    "Bayesian (Naive Bayes)": run_bayesian,
                    "KNN (K-Nearest Neighbors)": run_knn,
                    "Linear Regression": run_linear_regression,
                    "Logistic Regression": run_logistic_regression
                }
                
                # Train model
                model, best_params, best_score = algorithms[algorithm](X, y, task_type)
                
                if model is None:
                    st.error(f"❌ {algorithm} is not compatible with {task_type} tasks!")
                else:
                    st.success("✅ Model trained successfully!")
                    
                    # Store results
                    st.session_state.model = model
                    st.session_state.best_params = best_params
                    st.session_state.best_score = best_score
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.task_type = task_type
                    st.session_state.model_trained = True
                    
            except Exception as e:
                st.error(f"❌ Error during training: {e}")
    
    # Display results
    if st.session_state.model_trained:
        st.markdown("---")
        st.subheader("📈 Results")
        
        model = st.session_state.model
        best_params = st.session_state.best_params
        best_score = st.session_state.best_score
        X = st.session_state.X
        y = st.session_state.y
        task_type = st.session_state.task_type
        
        # Best parameters
        st.markdown("### 🎯 Best Parameters from GridSearchCV")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Best CV Score", f"{best_score:.4f}")
        with col2:
            params_df = pd.DataFrame([best_params]).T
            params_df.columns = ['Value']
            st.dataframe(params_df, use_container_width=True)
        
        # Model performance
        st.markdown("### 📊 Model Performance on Full Dataset")
        predictions = model.predict(X)
        
        if task_type == 'classification':
            accuracy = accuracy_score(y, predictions)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.markdown("**Confusion Matrix:**")
                cm = confusion_matrix(y, predictions)
                st.dataframe(pd.DataFrame(cm), use_container_width=True)
            
            with col2:
                st.markdown("**Classification Report:**")
                report = classification_report(y, predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
        
        else:
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Predictions vs Actual
            results_df = pd.DataFrame({
                'Actual': y,
                'Predicted': predictions,
                'Difference': y - predictions
            })
            st.markdown("**Sample Predictions:**")
            st.dataframe(results_df.head(20), use_container_width=True)

else:
    st.info("👈 Please upload a CSV file from the sidebar to get started")
    
    # Show example
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload your CSV file** using the file uploader in the sidebar
        2. **Select your target column** (the variable you want to predict)
        3. **Choose an algorithm** from the dropdown menu
        4. **Click 'Train Model'** to start training with GridSearchCV
        5. **View the results** including best parameters and model performance
        
        **Supported Algorithms:**
        - SVM (Classification & Regression)
        - Neural Networks (Classification & Regression)
        - Decision Tree (Classification & Regression)
        - Naive Bayes (Classification only)
        - KNN (Classification & Regression)
        - Linear Regression (Regression only)
        - Logistic Regression (Classification only)
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | GridSearchCV with 5-Fold Cross-Validation")