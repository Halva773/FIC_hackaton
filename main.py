import json
from preprocessing.formating import split_work_experience
import pandas as pd


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    data_dict = {'position': [],
                 'age': [],
                 'country': [],
                 'city': [],
                 'key_skills': [],
                 'client_name': [],
                 'grade_proof': [],
                 'salary': []}
    data = read_json('data/client_dataset.json')
    for datum in data:
        for key, value in datum.items():
            if key != 'work_experience':
                data_dict[key].append(value)

    dataframe = pd.DataFrame(data_dict)

    dataframe.to_csv('data/client_dataset_csv.csv')