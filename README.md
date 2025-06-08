🧠 Customer Churn Prediction using ANN (Keras + TensorFlow)
📌 Project Summary:

This project implements a classification model using Artificial Neural Networks (ANN) to predict whether a customer will churn (exit) from a bank based on demographic and account-related features. The model is trained using Keras and TensorFlow, with preprocessing, training, evaluation, and visualization handled in a structured machine learning pipeline.

- The objective is to help banks identify customers likely to churn so they can take preemptive retention actions.

📂 Dataset Details:

- Dataset Name: Churn_Modelling.csv

- Records: 10,000 bank customers

- Target Variable: Exited (1 = churned, 0 = retained)

Features Used:

- CreditScore, Geography, Gender, Age, Tenure, Balance

- NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

- The dataset is cleaned to remove unnecessary columns like RowNumber, CustomerId, and Surname, and categorical features are encoded using one-hot encoding.

🛠️ Technologies Used:

Python Libraries:

- Pandas — for data loading and manipulation

- NumPy — for numerical operations

- Seaborn & Matplotlib — for data visualization

- Scikit-learn — for preprocessing and model evaluation

- TensorFlow / Keras — for building and training the ANN model

🔄 Project Workflow:

- Load and Explore Data

- Understand data shape, distribution, and missing values

- Data Preprocessing

- Drop unnecessary columns

- Encode categorical features

- Feature scaling using StandardScaler

- Split Data

- Train-test split (80/20)

- Model Building

- Keras Sequential model with dense layers and ReLU activation

- Output layer uses sigmoid (binary classification)

- Model Compilation

- Loss: binary_crossentropy

- Optimizer: adam

- Metrics: accuracy

- Training and Evaluation

- Monitor training/validation accuracy

- Evaluate performance using accuracy and confusion matrix

- Visualization

- Accuracy and loss over epochs

- Confusion matrix heatmap

📈 Key Features:

ANN-based model using Keras with TensorFlow backend

- Fully preprocessed and scaled inputs

- Visual insight into feature distributions and training performance

- Predictive accuracy and confusion matrix evaluation

- Suitable for binary classification tasks like churn prediction



![Screenshot 2025-06-07 173242](https://github.com/user-attachments/assets/458d17e7-f147-4711-b128-4920985f2730)

![Screenshot 2025-06-07 173305](https://github.com/user-attachments/assets/54709992-6282-4707-b742-25cef4e00fb5)

![Screenshot 2025-06-07 173321](https://github.com/user-attachments/assets/6c9f8cc1-57e7-4cee-8810-b994b92ecf07)

![Screenshot 2025-06-07 174552](https://github.com/user-attachments/assets/ffd5afa4-6b31-434f-af36-77b67cfad06f)

![Screenshot 2025-06-07 174611](https://github.com/user-attachments/assets/09df5eb6-816f-460d-8eea-2bee95e9fa05)

![Screenshot 2025-06-07 174627](https://github.com/user-attachments/assets/66e7294d-155b-4b6d-b12d-6c18ee4de9b1)
