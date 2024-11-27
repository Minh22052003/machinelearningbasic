import pandas as pd
import numpy as np
import ThongKeDuLieu as tk
import joblib
from keras.api.models import load_model

model_svm = joblib.load('svm_model.pkl')
model_neural = load_model("heart_disease_model.h5")

data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
data = tk.Randomundersampler(data)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
nhan = np.unique(y)
Py = {}
Pxy = {}
for cls in nhan:
    X_cls = X[y == cls]
    Py[cls] = len(X_cls) / len(y)
    Pxy[cls] = {
        "mean": np.mean(X_cls, axis=0),
        "std": np.std(X_cls, axis=0)
    }

def gaussian_prob(x, mean, std):
    eps = 1e-6
    std = std + eps
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def dudoanvsbayes(X):
    predictions = []
    for x in X:
        probs = {}
        for cls in nhan:
            prior = np.log(Py[cls])
            likelihood = np.sum(
                np.log(gaussian_prob(x, Pxy[cls]["mean"], Pxy[cls]["std"]))
            )
            probs[cls] = prior + likelihood
        predictions.append(max(probs, key=probs.get))
    return np.array(predictions)















with open("input.txt", "r") as f:
    raw_data = f.read().strip()
data = [float(x) for x in raw_data.split(",")]
print("Du lieu:", data)

while True:
    print("-----------------------------")
    print("Chon phuong phap du doan: ")
    print("1. Naive Bayes")
    print("2. SVM")
    print("3. Neural Network")
    print("0. Thoat")
    choice = int(input("Chon pp : "))
    if(choice == 0):
        break
    elif(choice == 1):
        nhan = dudoanvsbayes(data)
        print("Nhan du doan voi Naive Bayes:", nhan[0])
    elif(choice == 2):
        data = np.array(data).reshape(1, -1)
        predicted_label = model_svm.predict(data)
        print("Nhan du doan voi SVMs:", int(predicted_label[0]))
    elif(choice == 3):
        data = np.array(data).reshape(1, -1)
        predicted_probability  = model_neural.predict(data)
        predicted_label = (predicted_probability > 0.5).astype(int)
        print("Nhan du doan vs Neural Network:", predicted_label[0][0])
