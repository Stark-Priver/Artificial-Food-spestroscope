# generate_test_data.py
import numpy as np
import pandas as pd

def create_test_samples(num_samples=5):
    """Creates realistic NIR test data with known compositions."""
    np.random.seed(42)
    data = []
    for _ in range(num_samples):
        # Realistic composition ranges (g/100g)
        protein = np.round(np.random.uniform(1, 20), 2)
        fat = np.round(np.random.uniform(1, 30), 2)
        carbs = np.round(np.random.uniform(10, 80), 2)
        
        # Simulate NIR spectrum (100 wavelengths)
        wavelengths = np.linspace(800, 2500, 100)
        spectrum = (
            0.5 * protein * np.exp(-0.002 * (wavelengths - 1650)**2) +  # Protein peak
            0.3 * fat * np.exp(-0.0015 * (wavelengths - 2300)**2) +     # Fat peak
            0.4 * carbs * np.exp(-0.0018 * (wavelengths - 2100)**2) +    # Carbs peak
            0.02 * np.random.randn(len(wavelengths))                     # Noise
        )
        data.append(np.append(spectrum, [protein, fat, carbs]))
    
    columns = [f"wl_{i}" for i in range(100)] + ["true_protein", "true_fat", "true_carbs"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("nir_test_data.csv", index=False)
    print(f"Generated {num_samples} test samples in nir_test_data.csv")

if __name__ == "__main__":
    create_test_samples()