import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import ThongKeDuLieu as tk
data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

data = tk.Randomundersampler(data)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
unique, counts = np.unique(y, return_counts=True)
print("Phân phối nhãn trong tập huấn luyện:", dict(zip(unique, counts)))

nhan = np.unique(y)
Py = {}
Pxy = {}
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alpha = 1
K = len(nhan)


for cls in nhan:
    X_cls = X_train[y_train == cls]
    Py[cls] = len(X_cls) / len(y_train)
    Pxy[cls] = {
        "mean": np.mean(X_cls, axis=0),
        "std": np.std(X_cls, axis=0)
    }


y_du_doan = dudoanvsbayes(X_test)

print("Độ chính xác:", accuracy_score(y_test, y_du_doan))






du_lieu_moi = np.array([[1.0,0.0,1.0,23.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,3.0,0.0,0.0,0.0,1.0,7.0,5.0,1.0]])
nhan_du_doan = dudoanvsbayes(du_lieu_moi)
print("Dữ liệu đầu vào:", du_lieu_moi)
print("Nhãn dự đoán:", nhan_du_doan[0])