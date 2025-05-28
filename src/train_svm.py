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
            # Konversi fitur ke float
            feature = [float(area), float(perimeter), float(aspect_ratio), float(extent), float(circularity)]
            # Konversi label ke -1 atau 1
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
    
    # Inisialisasi bobot dan bias secara acak
    w = [random.uniform(-1, 1) for _ in range(n_features)]
    b = random.uniform(-1, 1)
    
    # Daftar untuk melacak akurasi
    accuracies = []
    
    for epoch in range(epochs):
        # Acak urutan sampel untuk SGD
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        correct = 0
        for i in indices:
            x = features[i]
            y = labels[i]
            
            # Hitung prediksi
            score = dot_product(w, x) + b
            prediction = 1 if score >= 0 else -1
            
            # Hitung loss hinge
            loss = max(0, 1 - y * score)
            
            # Perbarui bobot dan bias jika ada loss
            if loss > 0:
                for j in range(n_features):
                    w[j] += learning_rate * (y * x[j])
                b += learning_rate * y
            
            # Hitung akurasi untuk epoch ini
            if prediction == y:
                correct += 1
        
        accuracy = correct / n_samples
        accuracies.append(accuracy)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Akurasi: {accuracy:.2f}")
    
    return w, b, accuracies

# Fungsi untuk memvisualisasikan akurasi
def plot_accuracy(accuracies):
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
    
    # Latih model
    w, b, accuracies = train_svm(features, labels)
    
    # Visualisasikan akurasi
    plot_accuracy(accuracies)
    
    print(f"Bobot akhir (w): {w}")
    print(f"Bias akhir (b): {b}")