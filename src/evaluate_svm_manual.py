import csv
import random
import matplotlib.pyplot as plt

# Fungsi untuk membaca bobot dan bias dari svm_results.csv
def load_svm_results(svm_results_file):
    with open(svm_results_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Lewati header
        for row in reader:
            w = [float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])]  # w0, w1, w2, w3, w4
            b = float(row[13])  # bias
    return w, b

# Fungsi untuk memuat data uji dari CSV terpisah
def load_test_data(test_file):
    test_features = []
    test_labels = []
    test_filenames = []
    with open(test_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Lewati header
        for row in reader:
            filename, area, perimeter, aspect_ratio, extent, circularity, label = row
            feature = [float(area), float(perimeter), float(aspect_ratio), float(extent), float(circularity)]
            label_num = -1 if label == 'bulat' else 1
            test_features.append(feature)
            test_labels.append(label_num)
            test_filenames.append(filename)
    return test_features, test_labels, test_filenames

# Fungsi untuk menghitung dot product secara manual
def dot_product(w, x):
    result = 0
    for i in range(len(w)):
        result += w[i] * x[i]
    return result

# Fungsi untuk prediksi manual
def predict(test_features, w, b):
    predictions = []
    for x in test_features:
        score = dot_product(w, x) + b
        prediction = 1 if score >= 0 else -1
        predictions.append(prediction)
    return predictions

# Fungsi untuk menghitung confusion matrix secara manual
def calculate_confusion_matrix(test_labels, predictions):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(test_labels, predictions):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == -1:
            fn += 1
        elif true == -1 and pred == 1:
            fp += 1
        elif true == -1 and pred == -1:
            tn += 1
    return tp, tn, fp, fn

# Fungsi untuk menghitung metrik evaluasi secara manual
def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

# Fungsi untuk visualisasi
def plot_results(test_features, test_labels, predictions, test_filenames):
    x1 = [f[2] for f in test_features]  # aspect_ratio (index 2)
    x2 = [f[4] for f in test_features]  # circularity (index 4)
    colors = ['green' if pred == true else 'red' for pred, true in zip(predictions, test_labels)]
    
    plt.scatter(x1, x2, c=colors)
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Circularity')
    plt.title('Hasil Klasifikasi Data Uji')
    plt.legend(['Benar', 'Salah'])
    plt.grid(True)
    plt.show()

# Fungsi utama
if __name__ == "__main__":
    svm_results_file = "svm_results.csv"  # File hasil pelatihan
    test_file = "uji_fitur_biskuit.csv"  # File data uji terpisah
    
    # Muat bobot dan bias dari svm_results.csv
    w, b = load_svm_results(svm_results_file)
    print(f"Bobot (w): {w}")
    print(f"Bias (b): {b}")
    
    # Muat data uji dari file terpisah
    test_features, test_labels, test_filenames = load_test_data(test_file)
    
    # Prediksi
    predictions = predict(test_features, w, b)
    
    # Hitung confusion matrix
    tp, tn, fp, fn = calculate_confusion_matrix(test_labels, predictions)
    print(f"Confusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Hitung metrik
    accuracy, precision, recall, f1 = calculate_metrics(tp, tn, fp, fn)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Visualisasi
    plot_results(test_features, test_labels, predictions, test_filenames)