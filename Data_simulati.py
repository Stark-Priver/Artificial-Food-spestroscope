# generate_nir_data.py
import numpy as np
import pandas as pd

def simulate_nir_spectrum(protein, fat, carbs):
    """Generates synthetic NIR spectra based on macronutrient composition."""
    wavelengths = np.linspace(800, 2500, 100)  # NIR range (800-2500 nm)
    
    # Simulate absorption peaks (real NIR peaks are more complex)
    protein_peak = 0.5 * np.exp(-0.002 * (wavelengths - 1650)**2)  # Amide band
    fat_peak = 0.3 * np.exp(-0.0015 * (wavelengths - 2300)**2)     # C-H stretch
    carbs_peak = 0.4 * np.exp(-0.0018 * (wavelengths - 2100)**2)   # O-H stretch
    
    # Combine with noise
    spectrum = (protein * protein_peak + 
               fat * fat_peak + 
               carbs * carbs_peak + 
               0.02 * np.random.randn(len(wavelengths)))
    return wavelengths, spectrum

def save_nir_csv(filename="nir_dataset.csv", num_samples=500):
    """Saves synthetic NIR data to CSV."""
    np.random.seed(42)
    data = []
    for _ in range(num_samples):
        protein = np.random.uniform(1, 20)   # g/100g
        fat = np.random.uniform(1, 30)       # g/100g
        carbs = np.random.uniform(10, 80)    # g/100g
        
        wavelengths, spectrum = simulate_nir_spectrum(protein, fat, carbs)
        row = np.append(spectrum, [protein, fat, carbs])
        data.append(row)
    
    columns = [f"wl_{i}" for i in range(100)] + ["protein", "fat", "carbs"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"NIR dataset saved to {filename}")

if __name__ == "__main__":
    save_nir_csv()