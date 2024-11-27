import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ThongKeDuLieu as tk

du_lieu = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

du_lieu = tk.Randomundersampler(du_lieu)

X = du_lieu.iloc[:, 1:].values
y = du_lieu.iloc[:, 0].values

nhan = np.unique(y)
PY = {}
Pxy = {}

def xac_suat_da_thuc(x, so_luong, tong_so_luong):
    return (so_luong.get(x, 0) + 1) / (tong_so_luong + len(so_luong))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alpha = 1
K = len(nhan)

for cls in nhan:
    X_cls = X_train[y_train == cls]
    PY[cls] = (len(X_cls) + alpha) / (len(y_train) + alpha * K)
    
    Pxy[cls] = {}
    for i in range(X_train.shape[1]):
        gia_tri_dac_trung, so_luong = np.unique(X_cls[:, i], return_counts=True)
        Pxy[cls][i] = dict(zip(gia_tri_dac_trung, so_luong + alpha))


def du_doan(X):
    du_doan = []
    for x in X:
        xac_suat = {}
        for cls in nhan:
            xac_suat_truoc = PY[cls]
            xac_suat_dieu_kien = 0
            for i, gia_tri_dac_trung in enumerate(x):
                xac_suat_categoric = Pxy[cls][i]
                xac_suat_dieu_kien *= xac_suat_da_thuc(gia_tri_dac_trung, xac_suat_categoric, len(X_cls))
            
            xac_suat[cls] = xac_suat_truoc * xac_suat_dieu_kien
        du_doan.append(max(xac_suat, key=xac_suat.get))
    return np.array(du_doan)


y_du_doan = du_doan(X_test)

print("Độ chính xác:", accuracy_score(y_test, y_du_doan))
