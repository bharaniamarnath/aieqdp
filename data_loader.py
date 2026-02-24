import pandas as pd

def load_data(labels_path, values_path):
    data_labels = pd.read_csv(labels_path)
    data_values = pd.read_csv(values_path)
    return data_labels, data_values