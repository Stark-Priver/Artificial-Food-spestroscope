# test_nir_model.py
import pandas as pd
import numpy as np
import onnxruntime as rt
import matplotlib.pyplot as plt

def load_and_predict(csv_path="nir_test_data.csv"):
    """Loads test CSV and predicts composition."""
    # Load data
    df = pd.read_csv(csv_path)
    X_test = df.iloc[:, :100].values  # Spectra
    y_true = df.iloc[:, -3:].values   # True compositions
    
    # Load ONNX model
    sess = rt.InferenceSession("nir_model.onnx")
    input_name = sess.get_inputs()[0].name
    
    # Predict
    y_pred = sess.run(None, {input_name: X_test.astype(np.float32)})[0]
    
    # Print results
    for i in range(len(X_test)):
        print(f"\nSample {i+1}:")
        print(f"True  -> Protein: {y_true[i, 0]:.2f}, Fat: {y_true[i, 1]:.2f}, Carbs: {y_true[i, 2]:.2f}")
        print(f"Pred  -> Protein: {y_pred[i, 0]:.2f}, Fat: {y_pred[i, 1]:.2f}, Carbs: {y_pred[i, 2]:.2f}")
    
    # Plot first sample's spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(800, 2500, 100), X_test[0], label="NIR Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title("Sample 1: NIR Spectrum")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    load_and_predict()