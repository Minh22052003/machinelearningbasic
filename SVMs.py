import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import ThongKeDuLieu as tk


du_lieu = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

du_lieu = tk.Randomundersampler(du_lieu)

X = du_lieu.iloc[:, 1:].values
y = du_lieu.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

dump(svm_model, "svm_model.pkl")

y_pred = svm_model.predict(X_test)

print("Độ chính xác trên tập kiểm tra:", accuracy_score(y_test, y_pred))
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

