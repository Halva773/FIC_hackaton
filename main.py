import json

from preprocessing.feature_generating import generate_worker_features, extract_salaries, read_features, \
    add_features_to_dataframe
from preprocessing.ratings import company_rates
import pandas as pd

from preprocessing.vectorizing import process_skills, cosine_distance


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
    new_columns = data.apply(lambda row: process_skills(row, cosine_distance), axis=1)
    data = pd.concat([data, new_columns], axis=1)

    features = read_features(file_path='../data/skills.txt')
    data = add_features_to_dataframe(data, features)

    data.drop(columns=['key_skills', 'position', 'salary', 'work_experience'])

    print(data.columns)
    print(data.head())