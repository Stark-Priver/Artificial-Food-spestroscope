import pandas as pd
import numpy as np
import onnxruntime as rt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class NIRAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NIR Composition Analyzer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.X_test = None
        self.y_true = None
        self.y_pred = None
        self.current_sample = 0
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # File input
        ttk.Label(control_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        self.file_entry = ttk.Entry(control_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5)
        self.file_entry.insert(0, "nir_test_data.csv")
        ttk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Sample selector
        ttk.Label(control_frame, text="Sample:").grid(row=1, column=0, sticky=tk.W)
        self.sample_slider = ttk.Scale(control_frame, from_=0, to=0, command=self.update_sample)
        self.sample_slider.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.sample_var = tk.StringVar()
        ttk.Label(control_frame, textvariable=self.sample_var).grid(row=1, column=2)
        
        # Action buttons
        ttk.Button(control_frame, text="Load & Predict", command=self.load_and_predict).grid(row=2, column=0, columnspan=3, pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        results_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization frame
        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create 6 subplots (2 rows, 3 columns)
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def browse_file(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
    
    def load_and_predict(self):
        try:
            # Load data
            csv_path = self.file_entry.get()
            df = pd.read_csv(csv_path)
            self.X_test = df.iloc[:, :100].values  # Spectra
            self.y_true = df.iloc[:, -3:].values   # True compositions
            
            # Load ONNX model
            sess = rt.InferenceSession("nir_model.onnx")
            input_name = sess.get_inputs()[0].name
            
            # Predict
            self.y_pred = sess.run(None, {input_name: self.X_test.astype(np.float32)})[0]
            
            # Update sample slider
            self.sample_slider.config(to=len(self.X_test)-1)
            self.current_sample = 0
            self.sample_slider.set(0)
            self.sample_var.set(f"1 / {len(self.X_test)}")
            
            # Update display
            self.update_plots()
            self.update_results()
            
            messagebox.showinfo("Success", "Data loaded and predictions made successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    
    def update_sample(self, value):
        self.current_sample = int(float(value))
        self.sample_var.set(f"{self.current_sample + 1} / {len(self.X_test) if self.X_test is not None else 0}")
        self.update_plots()
        self.update_results()
    
    def update_results(self):
        if self.y_true is None or self.y_pred is None:
            return
            
        self.results_text.delete(1.0, tk.END)
        i = self.current_sample
        
        # Basic results
        self.results_text.insert(tk.END, f"Sample {i+1} Results:\n")
        self.results_text.insert(tk.END, "-"*40 + "\n")
        self.results_text.insert(tk.END, f"True  -> Protein: {self.y_true[i, 0]:.2f}, Fat: {self.y_true[i, 1]:.2f}, Carbs: {self.y_true[i, 2]:.2f}\n")
        self.results_text.insert(tk.END, f"Pred  -> Protein: {self.y_pred[i, 0]:.2f}, Fat: {self.y_pred[i, 1]:.2f}, Carbs: {self.y_pred[i, 2]:.2f}\n")
        
        # Calculate errors
        errors = self.y_pred[i] - self.y_true[i]
        self.results_text.insert(tk.END, f"Error -> Protein: {errors[0]:.2f}, Fat: {errors[1]:.2f}, Carbs: {errors[2]:.2f}\n")
        
        # Formatting
        self.results_text.tag_configure("header", font=('Arial', 10, 'bold'))
        self.results_text.tag_add("header", "1.0", "1.end")
    
    def update_plots(self):
        if self.X_test is None or self.y_true is None or self.y_pred is None:
            return
            
        i = self.current_sample
        wavelengths = np.linspace(800, 2500, 100)
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: NIR Spectrum
        ax = self.axes[0, 0]
        ax.plot(wavelengths, self.X_test[i], label="NIR Spectrum", color='blue')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title(f"Sample {i+1}: NIR Spectrum")
        ax.grid(True)
        
        # Plot 2: Composition Bar Chart (True vs Predicted)
        ax = self.axes[0, 1]
        components = ['Protein', 'Fat', 'Carbs']
        width = 0.35
        x = np.arange(len(components))
        ax.bar(x - width/2, self.y_true[i], width, label='True', color='green')
        ax.bar(x + width/2, self.y_pred[i], width, label='Predicted', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylabel("Composition (%)")
        ax.set_title("True vs Predicted Composition")
        ax.legend()
        ax.grid(True, axis='y')
        
        # Plot 3: Prediction Error
        ax = self.axes[0, 2]
        errors = self.y_pred[i] - self.y_true[i]
        ax.bar(components, errors, color=['red' if e < 0 else 'blue' for e in errors])
        ax.axhline(0, color='black', linestyle='--')
        ax.set_ylabel("Prediction Error")
        ax.set_title("Prediction Errors")
        ax.grid(True, axis='y')
        
        # Plot 4: Feature Importance (example - could be replaced with actual feature importance if available)
        ax = self.axes[1, 0]
        # This is a placeholder - in a real scenario, you would use actual feature importance
        importance = np.abs(np.gradient(self.X_test[i]))
        ax.plot(wavelengths, importance, color='purple')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance (Gradient)")
        ax.grid(True)
        
        # Plot 5: All Samples Composition (Scatter)
        ax = self.axes[1, 1]
        for j, comp in enumerate(components):
            ax.scatter(self.y_true[:, j], self.y_pred[:, j], label=comp, alpha=0.6)
        ax.plot([0, 100], [0, 100], 'k--')  # Perfect prediction line
        ax.set_xlabel("True Composition (%)")
        ax.set_ylabel("Predicted Composition (%)")
        ax.set_title("All Samples: True vs Predicted")
        ax.legend()
        ax.grid(True)
        
        # Plot 6: Residuals Plot (Percentage)
        ax = self.axes[1, 2]
        # Calculate percentage residuals: (predicted - true)/true * 100
        percentage_residuals = (self.y_pred - self.y_true) / self.y_true * 100
        for j, comp in enumerate(components):
            ax.scatter(self.y_pred[:, j], percentage_residuals[:, j], label=comp, alpha=0.6)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel("Predicted Composition (%)")
        ax.set_ylabel("Residuals (% of true value)")
        ax.set_title("Percentage Residuals Analysis")
        ax.legend()
        ax.grid(True)
        
        self.canvas.draw()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = NIRAnalyzerApp(root)
    app.run()