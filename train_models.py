import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json

def load_and_preprocess_data():
    df = pd.read_csv('data/real_augmented_parvaah_x_data.csv')
    
    # Feature Engineering
    date_cols = ['Start_Date', 'Planned_Completion_Date', 'Current_Completion_Date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        
    df['Planned_Duration_Days'] = (df['Planned_Completion_Date'] - df['Start_Date']).dt.days
    df['Current_Duration_Days'] = (df['Current_Completion_Date'] - df['Start_Date']).dt.days
    
    df['Expenditure_Progress_Ratio'] = (df['Cumulative_Expenditure_Cr'] / df['Original_Approved_Cost_Cr']) / (df['Physical_Progress_Pct'] / 100 + 0.001)
    
    features_to_drop = ['Project_ID', 'Project_Name', 'Start_Date', 'Planned_Completion_Date', 
                        'Current_Completion_Date', 'Revised_Cost_Cr', 'Target_Cost_Overrun', 'Target_Time_Overrun']
    
    X = df.drop(columns=features_to_drop)
    y_cost = df['Target_Cost_Overrun']
    y_time = df['Target_Time_Overrun']
    
    # Label Encoding for categorical
    categorical_cols = ['Sector', 'Implementing_Agency'] 
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    return X, y_cost, y_time, label_encoders

def train_and_evaluate(X, y, target_name):
    print(f"\n--- Training for {target_name} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline: Logistic Regression
    # Scale data for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    try:
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        lr_preds = lr_model.predict(X_test_scaled)
        
        lr_acc = accuracy_score(y_test, lr_preds)
        lr_f1 = f1_score(y_test, lr_preds)
    except ValueError as e:
        print(f"Logistic Regression failed: {e}")
        lr_acc = 0.0
        lr_f1 = 0.0
    
    print(f"Baseline (Logistic Regression):")
    print(f"  Accuracy: {lr_acc:.4f}")
    print(f"  F1-Score: {lr_f1:.4f}")
    
    try:
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_f1 = f1_score(y_test, xgb_preds)
    except ValueError as e:
        print(f"XGBoost failed: {e}")
        # If model training fails due to only one class, we will just return a dummy or predict all 0
        from sklearn.dummy import DummyClassifier
        xgb_model = DummyClassifier(strategy="constant", constant=y_train.iloc[0] if len(y_train) > 0 else 0)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_f1 = f1_score(y_test, xgb_preds)
        
    print(f"Advanced (XGBoost):")
    print(f"  Accuracy: {xgb_acc:.4f}")
    print(f"  F1-Score: {xgb_f1:.4f}")
    
    print(f"Improvement using AI/ML: +{(xgb_acc - lr_acc)*100:.2f}% Accuracy")
    
    return xgb_model

def main():
    os.makedirs('models', exist_ok=True)
    
    print("Loading and preprocessing data...")
    X, y_cost, y_time, label_encoders = load_and_preprocess_data()
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
        
    # Save label encoders mapping
    le_mapping = {}
    for col, le in label_encoders.items():
        le_mapping[col] = list(le.classes_)
    with open('models/label_encoders.json', 'w') as f:
        json.dump(le_mapping, f)
        
    cost_model = train_and_evaluate(X, y_cost, "Cost Overrun")
    joblib.dump(cost_model, 'models/cost_model.joblib')
    
    time_model = train_and_evaluate(X, y_time, "Time Overrun")
    joblib.dump(time_model, 'models/time_model.joblib')
    
    print("\nModels trained and saved in 'models/' directory.")

if __name__ == "__main__":
    main()
