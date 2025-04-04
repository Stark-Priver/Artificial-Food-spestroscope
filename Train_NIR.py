# train_nir_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def load_nir_data(filename="nir_dataset.csv"):
    """Loads NIR spectra and labels."""
    df = pd.read_csv(filename)
    X = df.iloc[:, :-3].values  # Spectra (100 wavelengths)
    y = df.iloc[:, -3:].values  # Labels: [protein, fat, carbs]
    return X, y

def train_and_export_nir_model():
    """Trains and exports an NIR prediction model."""
    X, y = load_nir_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Protein MAE: {mean_absolute_error(y_test[:, 0], y_pred[:, 0]):.2f} g/100g")
    print(f"Fat MAE: {mean_absolute_error(y_test[:, 1], y_pred[:, 1]):.2f} g/100g")
    print(f"Carbs MAE: {mean_absolute_error(y_test[:, 2], y_pred[:, 2]):.2f} g/100g")
    
    # Export to ONNX (for MATLAB)
    initial_type = [('float_input', FloatTensorType([None, 100]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open("nir_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # Export to joblib (for Python)
    joblib.dump(model, "nir_model.pkl")
    print("NIR model exported to ONNX and joblib")

if __name__ == "__main__":
    train_and_export_nir_model()