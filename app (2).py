import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Insurance Analysis", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

df1 = load_data()

# Encoding Categorical Variables
def encode_categorical_data(df):
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])
    return df

df1 = encode_categorical_data(df1)

# Data Scaling
scaler = MinMaxScaler()
df1[['age', 'bmi', 'children']] = scaler.fit_transform(df1[['age', 'bmi', 'children']])

# Split Data for Training
X = df1.drop('expenses', axis=1)
y = df1['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_svm = SVR(kernel='linear')
model_svm.fit(X_train, y_train)

model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select a Section", ("Data Overview", "Make Predictions", "Model Evaluation"))

if section == "Data Overview":
    st.title("📊 Insurance Dataset Overview")
    st.write("This dataset contains details about individuals' medical expenses, which can be predicted based on their personal information.")
    
    # Column Selection for Data Display
    columns_to_show = st.multiselect("Select Columns to View", df1.columns, default=df1.columns.tolist())
    st.write(df1[columns_to_show])

    # Statistical Summary
    if st.checkbox("Show Statistical Summary"):
        st.subheader("Statistical Summary")
        st.write(df1.describe())

    # Missing Data
    if st.checkbox("Show Missing Data"):
        st.subheader("Missing Data")
        st.write(df1.isnull().sum())

    # Feature Distribution (User Can Choose Feature to Visualize)
    selected_feature = st.selectbox("Choose a feature to visualize distribution", df1.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df1[selected_feature], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

elif section == "Make Predictions":
    st.title("💡 Make Predictions")
    st.write("Enter the details below to predict yearly medical expenses.")

    # Input Fields
    age = st.slider("Age", 18, 100, 30)
    sex = st.radio("Sex", ["Male", "Female"])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    smoker = st.radio("Smoker Status", ["Yes", "No"])
    region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

    # Convert Inputs
    sex_encoded = 1 if sex == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0
    region_mapping = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}
    region_encoded = region_mapping[region]

    # Prepare Input Data
    input_data_raw = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_encoded],
        'region': [region_encoded]
    })

    # Scale only the numerical features
    scaled_features = ['age', 'bmi', 'children']
    input_data_scaled = input_data_raw.copy()
    input_data_scaled[scaled_features] = scaler.transform(input_data_raw[scaled_features])

    # Model Selection
    model_choice = st.selectbox("Choose a Model", ("Linear Regression", "Support Vector Machine", "Decision Tree"))

    if model_choice == "Linear Regression":
        if st.button("Predict Expenses with Linear Regression"):
            prediction = model_lr.predict(input_data_scaled)
            original_prediction = prediction[0]
            st.success(f"Predicted Yearly Expense: **${original_prediction:,.2f}**")

    elif model_choice == "Support Vector Machine":
        if st.button("Predict Expenses with SVM"):
            prediction = model_svm.predict(input_data_scaled)
            original_prediction = prediction[0]
            st.success(f"Predicted Yearly Expense: **${original_prediction:,.2f}**")

    elif model_choice == "Decision Tree":
        if st.button("Predict Expenses with Decision Tree"):
            prediction = model_dt.predict(input_data_scaled)
            original_prediction = prediction[0]
            st.success(f"Predicted Yearly Expense: **${original_prediction:,.2f}**")

elif section == "Model Evaluation":
    st.title("📊 Model Evaluation")
    st.write("This section evaluates the performance of different models.")
    
    # User Input for Test Size
    test_size = st.slider("Select Test Size for Data Split", 0.1, 0.9, 0.2)

    # Re-split data with user-defined test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model Evaluation Metrics
    evaluation_metric = st.selectbox("Choose Evaluation Metric", ("Mean Squared Error", "R-squared"))

    # Linear Regression Evaluation
    if evaluation_metric == "Mean Squared Error":
        lr_pred = model_lr.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        st.write(f"Linear Regression - Mean Squared Error (MSE): {lr_mse}")
    elif evaluation_metric == "R-squared":
        lr_pred = model_lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        st.write(f"Linear Regression - R-squared: {lr_r2}")
    
    # Support Vector Machine Evaluation
    if evaluation_metric == "Mean Squared Error":
        svm_pred = model_svm.predict(X_test)
        svm_mse = mean_squared_error(y_test, svm_pred)
        st.write(f"SVM - Mean Squared Error (MSE): {svm_mse}")
    elif evaluation_metric == "R-squared":
        svm_pred = model_svm.predict(X_test)
        svm_r2 = r2_score(y_test, svm_pred)
        st.write(f"SVM - R-squared: {svm_r2}")
    
    # Decision Tree Evaluation
    if evaluation_metric == "Mean Squared Error":
        dt_pred = model_dt.predict(X_test)
        dt_mse = mean_squared_error(y_test, dt_pred)
        st.write(f"Decision Tree - Mean Squared Error (MSE): {dt_mse}")
    elif evaluation_metric == "R-squared":
        dt_pred = model_dt.predict(X_test)
        dt_r2 = r2_score(y_test, dt_pred)
        st.write(f"Decision Tree - R-squared: {dt_r2}")

    # Model Evaluation Visualization
    st.subheader("Model Performance Comparison")
    comparison_data = {
        "Linear Regression": lr_r2 if evaluation_metric == "R-squared" else lr_mse,
        "SVM": svm_r2 if evaluation_metric == "R-squared" else svm_mse,
        "Decision Tree": dt_r2 if evaluation_metric == "R-squared" else dt_mse,
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(comparison_data.keys()), y=list(comparison_data.values()), ax=ax)
    st.pyplot(fig)

# Additional Styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .stButton button {
            background-color: #2d6a4f;
            color: white;
            font-weight: bold;
        }
        .stSlider div {
            background-color: #f0f0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

