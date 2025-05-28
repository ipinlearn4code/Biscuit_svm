import csv
import random
import matplotlib.pyplot as plt

# Fungsi untuk membaca data dari CSV
def load_data(csv_file):
    features = []
    labels = []
    filenames = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Lewati header
        for row in reader:
            filename, area, perimeter, aspect_ratio, extent, circularity, label = row
            feature = [float(area), float(perimeter), float(aspect_ratio), float(extent), float(circularity)]
            label_num = -1 if label == 'bulat' else 1
            features.append(feature)
            labels.append(label_num)
            filenames.append(filename)
    return features, labels, filenames

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

# Fungsi untuk membuat prediksi
def predict(features, w, b):
    predictions = []
    for x in features:
        score = dot_product(w, x) + b
        prediction = 1 if score >= 0 else -1
        predictions.append(prediction)
    return predictions

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
    output_csv = "svm_results.csv"
    
    # Muat data
    features, labels, filenames = load_data(csv_file)
    
    # Latih model
    w, b, accuracies = train_svm(features, labels)
    
    # Buat prediksi
    predictions = predict(features, w, b)
    
    # Simpan ke CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['filename', 'area', 'perimeter', 'aspect_ratio', 'extent', 'circularity', 
                  'label', 'predicted_label', 'w0', 'w1', 'w2', 'w3', 'w4', 'bias']
        writer.writerow(header)
        
        # Data
        with open(csv_file, 'r') as input_f:
            reader = csv.reader(input_f)
            next(reader)  # Lewati header
            for i, row in enumerate(reader):
                filename, area, perimeter, aspect_ratio, extent, circularity, label = row
                pred_label = 'persegi' if predictions[i] == 1 else 'bulat'
                writer.writerow([filename, area, perimeter, aspect_ratio, extent, circularity, 
                               label, pred_label] + w + [b])
    
    # Visualisasikan akurasi (opsional)
    plot_accuracy(accuracies)
    
    print(f"Hasil pelatihan disimpan di {output_csv}")
    print(f"Bobot akhir (w): {w}")
    print(f"Bias akhir (b): {b}")