# Dependencies 
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib 

# -------- note------------------------------------------------
# The experimentation has been conducted in a notebook 
# Hence the decisions regarding cleaning, model choice, etc are
# all as a result of that experimentation (using MLflow).
# --------------------------------------------------------------


# Read data 
customers = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")



# Clean data 
customers.drop(customers[customers["TotalCharges"]== " "].index, inplace=True)
customers["TotalCharges"] = pd.to_numeric(customers["TotalCharges"]) 



# Pre-processing data 
y = customers["Churn"] 
X = customers.loc[:, ~customers.columns.isin(['customerID', 'Churn'])]
categorical_columns = X.select_dtypes(include='object').columns
numerical_columns = X.select_dtypes(exclude='object').columns
 
preprocessor = ColumnTransformer(
    transformers = [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns), 
        ("numerical", StandardScaler(), numerical_columns)
    ])

# random state 42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Training 
# training pipeline with logistic regression model 
model = Pipeline(
    steps = [
        ("preprocessor", preprocessor), 
        ("classifier", LogisticRegression()) 
    ]
)

# model training and prediction 
model.fit(X_train, y_train) 

joblib.dump(model, filename="../models/lr-churn-model.joblib")
