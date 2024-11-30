import json

from preprocessing.feature_generating import generate_worker_features, extract_salaries
from preprocessing.ratings import company_rates
import pandas as pd


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    with open("data/client_dataset.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    data = generate_worker_features(pd.DataFrame(data))
    data = extract_salaries(data, 'salary')
    data = company_rates(pd.DataFrame(data))
    print(data.columns)
    print(data.head())