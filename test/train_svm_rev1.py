import csv
import random
import matplotlib.pyplot as plt

# Fungsi untuk membaca data dari CSV
def load_data(csv_file):
    features = []
    labels = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Lewati header
        for row in reader:
            filename, area, perimeter, aspect_ratio, extent, circularity, label = row
            feature = [float(area), float(perimeter), float(aspect_ratio), float(extent), float(circularity)]
            label_num = -1 if label == 'bulat' else 1
            features.append(feature)
            labels.append(label_num)
    return features, labels

# Fungsi untuk menghitung dot product secara manual
def dot_product(w, x):
    result = 0
    for i in range(len(w)):
        result += w[i] * x[i]
    return result

# Fungsi untuk melatih SVM dengan SGD
def train_svm(features, labels, learning_rate=0.01, epochs=100):
    n_samples = len(features)
    n_features = len(features[0])
    
    w = [random.uniform(-1, 1) for _ in range(n_features)]
    b = random.uniform(-1, 1)
    
    accuracies = []
    
    for epoch in range(epochs):
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        correct = 0
        for i in indices:
            x = features[i]
            y = labels[i]
            
            score = dot_product(w, x) + b
            prediction = 1 if score >= 0 else -1
            
            loss = max(0, 1 - y * score)
            
            if loss > 0:
                for j in range(n_features):
                    w[j] += learning_rate * (y * x[j])
                b += learning_rate * y
            
            if prediction == y:
                correct += 1
        
        accuracy = correct / n_samples
        accuracies.append(accuracy)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Akurasi: {accuracy:.2f}")
    
    return w, b, accuracies

# Fungsi untuk mengidentifikasi support vector
def find_support_vectors(features, labels, w, b):
    support_vectors = []
    for x, y in zip(features, labels):
        score = dot_product(w, x) + b
        if abs(y * score) <= 1:  # Titik dalam margin
            support_vectors.append((x[2], x[4], y))  # Ambil aspect_ratio dan circularity
    return support_vectors

# Fungsi untuk memvisualisasikan persebaran data sebelum pelatihan
def plot_initial_distribution(features, labels):
    x_bulat = [f[2] for f, l in zip(features, labels) if l == -1]  # aspect_ratio
    y_bulat = [f[4] for f, l in zip(features, labels) if l == -1]  # circularity
    x_persegi = [f[2] for f, l in zip(features, labels) if l == 1]  # aspect_ratio
    y_persegi = [f[4] for f, l in zip(features, labels) if l == 1]  # circularity
    
    plt.figure(figsize=(6, 5))
    plt.scatter(x_bulat, y_bulat, color='blue', label='Bulat', alpha=0.6)
    plt.scatter(x_persegi, y_persegi, color='red', label='Persegi', alpha=0.6)
    
    plt.title('Persebaran Data Sebelum Pelatihan')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Circularity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Fungsi untuk memvisualisasikan hasil pelatihan
def plot_final_distribution(features, labels, w, b, support_vectors):
    x_bulat = [f[2] for f, l in zip(features, labels) if l == -1]  # aspect_ratio
    y_bulat = [f[4] for f, l in zip(features, labels) if l == -1]  # circularity
    x_persegi = [f[2] for f, l in zip(features, labels) if l == 1]  # aspect_ratio
    y_persegi = [f[4] for f, l in zip(features, labels) if l == 1]  # circularity
    
    plt.figure(figsize=(6, 5))
    plt.scatter(x_bulat, y_bulat, color='blue', label='Bulat', alpha=0.6)
    plt.scatter(x_persegi, y_persegi, color='red', label='Persegi', alpha=0.6)
    
    # Plot support vectors
    x_sv = [sv[0] for sv in support_vectors]
    y_sv = [sv[1] for sv in support_vectors]
    colors_sv = ['green' if sv[2] == -1 else 'orange' for sv in support_vectors]
    plt.scatter(x_sv, y_sv, color=colors_sv, edgecolor='black', s=100, label='Support Vectors', zorder=5)
    
    # Plot hyperplane dan margin (hanya untuk 2D aspect_ratio vs circularity)
    x_range = [min(x_bulat + x_persegi), max(x_bulat + x_persegi)]
    if w[2] != 0:  # Hindari pembagian dengan nol
        m = -w[4] / w[2]  # Gradien hyperplane
        c = -b / w[2]  # Intersep
        y_hyper = [m * x + c for x in x_range]
        plt.plot(x_range, y_hyper, 'k-', label='Decision Boundary')
        
        # Margin (w^T x + b = ±1)
        y_margin_pos = [m * x + (c + 1/w[2]) for x in x_range]
        y_margin_neg = [m * x + (c - 1/w[2]) for x in x_range]
        plt.plot(x_range, y_margin_pos, 'k--', alpha=0.5, label='Margin')
        plt.plot(x_range, y_margin_neg, 'k--', alpha=0.5)
    
    plt.title('Persebaran Data Setelah Pelatihan')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Circularity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Fungsi untuk memvisualisasikan akurasi
def plot_accuracy(accuracies):
    plt.figure(figsize=(6, 5))
    plt.plot(range(len(accuracies)), accuracies)
    plt.title('Akurasi Pelatihan per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.grid(True)
    plt.show()

# Fungsi utama
if __name__ == "__main__":
    csv_file = "fitur_biskuit.csv"
    features, labels = load_data(csv_file)
    
    # Visualisasi sebelum pelatihan
    print("Menampilkan persebaran data sebelum pelatihan...")
    plot_initial_distribution(features, labels)
    
    # Latih model
    print("Melatih model SVM...")
    w, b, accuracies = train_svm(features, labels)
    
    # Identifikasi support vectors
    support_vectors = find_support_vectors(features, labels, w, b)
    
    # Visualisasi setelah pelatihan
    print("Menampilkan persebaran data setelah pelatihan...")
    plot_final_distribution(features, labels, w, b, support_vectors)
    
    # Visualisasi akurasi
    print("Menampilkan grafik akurasi...")
    plot_accuracy(accuracies)
    
    print(f"Bobot akhir (w): {w}")
    print(f"Bias akhir (b): {b}")