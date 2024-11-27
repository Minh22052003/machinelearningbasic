from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pandas as pd

def select_10features(data):
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)

    selected_columns = X.columns[selector.get_support()]
    data = data.drop(columns=[col for col in data.columns if col not in selected_columns])
    return data

def Smote(data):
    X = data.drop('HeartDiseaseorAttack', axis=1)
    y = data['HeartDiseaseorAttack']

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    data_resampled.insert(0, 'HeartDiseaseorAttack', y_resampled)

    return data_resampled

def Randomundersampler(data):
    X = data.drop('HeartDiseaseorAttack', axis=1)
    y = data['HeartDiseaseorAttack']

    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    data_resampled.insert(0, 'HeartDiseaseorAttack', y_resampled)

    return data_resampled