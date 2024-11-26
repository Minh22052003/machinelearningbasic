import pandas as pd
import ThongKeDuLieu as tk

du_lieu = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")

du_lieu_resampled = tk.Smote(du_lieu)

print(du_lieu_resampled['HeartDiseaseorAttack'].value_counts())
print(du_lieu_resampled.head())
