import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page config
st.set_page_config(page_title="Customer Churn ANN Classifier", layout="wide")

# Title
st.title("💡 Customer Churn Prediction using ANN")
st.markdown("This app predicts whether a customer will **churn** using a pre-trained Artificial Neural Network.")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\divya\Downloads\Churn_Modelling.csv")
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'Churn_Modelling.csv' is in the specified path.")
        return pd.DataFrame()

df = load_data()

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_ann.joblib")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'churn_ann.joblib' is in the working directory.")
        return None

model = load_model()

# Navigation
page = st.sidebar.radio("Choose Page", ["📊 Data Exploration", "🧠 Model & Metrics", "📈 Predict Churn"])

# Page 1: Data Exploration
if page == "📊 Data Exploration":
    st.subheader("Dataset Overview")
    if df.empty:
        st.warning("Dataset not loaded.")
    else:
        if st.checkbox("Show raw data"):
            st.write(df)

        if st.checkbox("Show info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        if st.checkbox("Show descriptive statistics"):
            st.write(df.describe())

        # Select numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            selected_num = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num], kde=True, ax=ax)
            st.pyplot(fig)

        with col2:
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=selected_cat, ax=ax)
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))

        if numeric_cols:
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        else:
            st.write("No numeric columns available for correlation heatmap.")

        st.pyplot(fig)

# Page 2: Model Architecture and Performance
elif page == "🧠 Model & Metrics":
    st.subheader("ANN Architecture")
    st.code("""
model = keras.Sequential([
    keras.layers.Dense(6, activation='relu', input_shape=(10,)),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
    """)
    st.markdown("### 📊 Performance (example)")
    st.write("Accuracy: 86%")
    st.write("Precision: 0.78")
    st.write("Recall: 0.65")
    st.write("F1-score: 0.71")

    st.subheader("Confusion Matrix")
    cm = np.array([[1500, 100], [200, 300]])  # Example confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Page 3: Prediction
elif page == "📈 Predict Churn":
    st.subheader("Enter Customer Details")

    if model is not None and not df.empty:
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                credit_score = st.number_input("Credit Score", 300, 850, 600)
                geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.slider("Age", 18, 100, 35)
                tenure = st.slider("Tenure (years)", 0, 10, 5)

            with col2:
                balance = st.number_input("Balance", value=10000.0)
                num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
                has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
                is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
                estimated_salary = st.number_input("Estimated Salary", value=50000.0)

            submitted = st.form_submit_button("Predict")

        if submitted:
            # Map categorical inputs to numerical values
            geo_dict = {"France": 0, "Germany": 1, "Spain": 2}
            gender_dict = {"Male": 1, "Female": 0}

            input_data = np.array([[ 
                credit_score,
                geo_dict[geography],
                gender_dict[gender],
                age,
                tenure,
                balance,
                num_of_products,
                1 if has_cr_card == "Yes" else 0,
                1 if is_active_member == "Yes" else 0,
                estimated_salary
            ]])

            # Make prediction safely
            prediction = model.predict(input_data)

            # Handle different output shapes safely
            if np.isscalar(prediction):
                prediction_prob = prediction
            elif len(prediction.shape) == 1:
                prediction_prob = prediction[0]
            else:
                prediction_prob = prediction[0][0]

            result = "Customer is likely to churn" if prediction_prob > 0.5 else "Customer is likely to stay"
            st.success(f"🔍 Prediction: **{result}** (Confidence: {prediction_prob:.2f})")
    else:
        st.warning("Please ensure the model and dataset are properly loaded.")
