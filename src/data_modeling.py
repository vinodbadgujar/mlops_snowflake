# src/data_modeling.py

from sklearn.tree import DecisionTreeClassifier
from src.snowflake_client import fetch_table_as_dataframe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    #encoding target variable
    df["FS"] = df["FS"].map({1: 1, 2: 0})
    return df


def split_data(df: pd.DataFrame, target_column: str = "FS", test_size: float = 0.2):
    print(f"Splitting data into training and testing sets with test_size={test_size}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    print("Scaling features using StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def train_decision_tree_classifier(X_train, y_train):
    print("Training Decision tree Classifier model")
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20, 
        min_samples_leaf=10,
        criterion='gini',       
        random_state=42         
    )
    model.fit(X_train, y_train)
    print("Model training complete")
    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, report, cm


def get_feature_importance(model, X_train):

    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    print("Top 10 important features:")
    print(importance_df.head(10))
    
    return importance_df


def save_model(model, scaler):

    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join(models_dir, 'food_stamp_model.pkl')
    scaler_path = os.path.join(models_dir, 'food_stamp_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def main():
    # Load and prepare data
    table_name = "GOLD"
    print(f"Loading data from table: {table_name}")
    df = fetch_table_as_dataframe(table_name)
    df = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_decision_tree_classifier(X_train_scaled, y_train)
    
    # Evaluate model
    accuracy, report, confusion_matrix = evaluate_model(model, X_test_scaled, y_test)
    
    # Get feature importance
    importance_df = get_feature_importance(model, X_train)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print("Food stamp classification modeling complete!")

if __name__ == "__main__":
    main()