import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def normalize(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return (X - X_min) / (X_max - X_min)


def load_wdbc(filepath):
    df = pd.read_csv(filepath)

    labels = df['label'].values
    features = df.drop(columns=['label']).values

    features = normalize(features)

    x_list = [features[i] for i in range(len(features))]
    y_list = [np.array([labels[i]]) for i in range(len(labels))]

    return x_list, y_list


def load_loan(filepath):
    df = pd.read_csv(filepath)

    labels = df['label'].values
    df = df.drop(columns=['label'])

    # Identify categorical vs numerical columns by name
    cat_cols = [c for c in df.columns if 'cat' in c]
    num_cols = [c for c in df.columns if 'num' in c]

    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False)
    cat_encoded = encoder.fit_transform(df[cat_cols])

    # Normalize numerical columns
    num_data = normalize(df[num_cols].values)

    # Combine numerical and one-hot encoded features
    features = np.hstack([num_data, cat_encoded])

    x_list = [features[i] for i in range(len(features))]
    y_list = [np.array([labels[i]]) for i in range(len(labels))]

    return x_list, y_list
