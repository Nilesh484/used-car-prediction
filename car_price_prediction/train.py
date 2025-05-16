import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import datetime

def load_data(filepath):
    # Load dataset
    if filepath.endswith('.xlsx'):
        data = pd.read_excel(filepath)
    else:
        data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Derive Car_Age feature
    current_year = datetime.datetime.now().year
    data['Car_Age'] = current_year - data['Year']
    # Drop columns not needed or redundant
    data = data.drop(['Year', 'car_ID', 'CarName'], axis=1, errors='ignore')

    # Separate features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Define categorical features for OneHotEncoding
    categorical_features = ['fueltype', 'doornumber', 'carbody', 'enginetype', 'seller_type', 'transmission']

    # Check if seller_type and transmission columns exist, if not fallback to existing columns
    if 'seller_type' not in X.columns:
        if 'Seller_Type' in X.columns:
            X.rename(columns={'Seller_Type': 'seller_type'}, inplace=True)
        else:
            categorical_features.remove('seller_type')
    if 'transmission' not in X.columns:
        if 'Transmission' in X.columns:
            X.rename(columns={'Transmission': 'transmission'}, inplace=True)
        else:
            categorical_features.remove('transmission')

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    return X, y, preprocessor

def train_model(X, y, preprocessor):
    # Create pipeline with preprocessing and Linear Regression model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score on test set: {score:.4f}")

    return model

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    # Path to dataset - user should update this path as needed
    dataset_path = 'traindata.csv'  # Updated to existing CSV file

    data = load_data(dataset_path)
    X, y, preprocessor = preprocess_data(data)
    model = train_model(X, y, preprocessor)
    save_model(model, 'car_price_model.pkl')
