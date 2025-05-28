import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Baca CSV dari file
# df = pd.read_csv("fitur_biskuit.csv")
df = pd.read_csv("svm_results.csv")

# Plot Area vs Perimeter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="area", y="perimeter", hue="label", style="label", s=100)
plt.title("Plot Area vs Perimeter berdasarkan Label")
plt.xlabel("Area")
plt.ylabel("Perimeter")
plt.grid(True)
plt.legend(title="Label")
plt.tight_layout()
plt.show()
