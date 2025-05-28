import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Baca file
df = pd.read_csv("svm_results.csv")

# Plot data dengan label asli dan prediksi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="area", y="perimeter", hue="label", style="predicted_label", s=100)

w0 = df.iloc[0]["w0"]
w1 = df.iloc[0]["w1"]
bias = df.iloc[0]["bias"]

# Buat garis decision boundary
x_vals = np.linspace(df["area"].min(), df["area"].max(), 100)
# Selesaikan untuk y: w0*x + w1*y + bias = 0 ⇒ y = -(w0*x + bias)/w1
y_vals = -(w0 * x_vals + bias) / w1

# Gambar hyperplane
plt.plot(x_vals, y_vals, 'k--', label='SVM Hyperplane')

# Format plot
plt.title("SVM Result: Area vs Perimeter")
plt.xlabel("Area")
plt.ylabel("Perimeter")
plt.legend(title="Label")
plt.grid(True)
plt.tight_layout()
plt.show()
