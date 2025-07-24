**📊 Customer Churn Prediction Model**
This project builds a machine learning pipeline to predict customer churn using a structured dataset (e.g., Telco Customer Churn). The pipeline includes preprocessing, handling class imbalance with SMOTE, model training, evaluation, and saving the best model for future predictions.

**🧠 Model Overview**
This churn prediction model uses supervised learning to identify customers likely to stop using a service. 
Two ensemble models are evaluated:
Random Forest
Gradient Boosting (🏆 Best Model)

**⚙️ Features Used**
🔢 Numerical:
tenure

MonthlyCharges

TotalCharges

AvgChargePerTenure (engineered)

🔤 Categorical:
gender

SeniorCitizen

Partner

Dependents

PhoneService

MultipleLines

InternetService

OnlineSecurity

OnlineBackup

DeviceProtection

TechSupport

StreamingTV

StreamingMovies

Contract

PaperlessBilling

PaymentMethod

**🔄 Pipeline Workflow**
**Data Preprocessing:**

Convert TotalCharges to numeric.

Encode Churn as binary.

Engineer AvgChargePerTenure and HasInternet.

Drop irrelevant columns (customerID).

**Preprocessing Pipeline:**

StandardScaler for numerical features.

OneHotEncoder for categorical features.

**Model Training:**

Oversampling with SMOTE.

Hyperparameter tuning with GridSearchCV.

Evaluation using ROC AUC, precision, recall, F1 score, etc.

**Model Saving:**

Best model and preprocessor saved as best_churn_model.pkl.

📈 Performance
Best Model: Gradient Boosting
Test ROC AUC: 0.84

**Classification Report:**

               precision    recall  f1-score   support
     Not Churn     0.88      0.82      0.85      1552
         Churn     0.58      0.69      0.63       561
       Accuracy                         0.78      2113
       
**💾 How to Use**
🔧 Installation

pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib

**🚀 Training the Model**
python churn_model.py
This will:

Load and clean the data

Train multiple models

Evaluate and select the best one

Save the model as best_churn_model.pkl

**🔍 Making Predictions**
from churn_model import predict_churn

sample_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 89.10,
    'TotalCharges': 1050.25,
    'AvgChargePerTenure': 87.52,
    'HasInternet': 1
}

prediction = predict_churn(sample_customer)
print(prediction)

**🧠 Future Improvements**
Add support for automated feature selection

Train on additional models (e.g., XGBoost, LightGBM)

Convert into a REST API using Flask or FastAPI

Deploy with Docker or on a cloud platform

**📜 License**
This project is licensed under the MIT License — you are free to use, modify, and distribute it with proper attribution.
See LICENSE for more details.

**🤝 Contributing**
Feel free to fork, improve, and submit PRs.
