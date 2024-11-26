import numpy as np
import pandas as pd
import ThongKeDuLieu as tk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    """Tính xác suất Gaussian P(x | Y)"""
    eps = 1e-6
    std = std + eps
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def predict(X):
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for cls in nhan:
    X_cls = X_train[y_train == cls]
    Py[cls] = len(X_cls) / len(y_train)
    Pxy[cls] = {
        "mean": np.mean(X_cls, axis=0),
        "std": np.std(X_cls, axis=0)
    }

y_pred = predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))