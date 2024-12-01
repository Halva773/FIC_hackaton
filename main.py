import json

from preprocessing.feature_generating import generate_worker_features, extract_salaries, read_features, \
    add_features_to_dataframe
from preprocessing.ratings import company_rates
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matri
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing.vectorizing import process_skills, cosine_distance


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def main(file_path="data/client_dataset.json"):
    data = read_json(file_path)

    data = generate_worker_features(pd.DataFrame(data))
    data = extract_salaries(data, 'salary')
    data = company_rates(pd.DataFrame(data))
    new_columns = data.apply(lambda row: process_skills(row, cosine_distance), axis=1)
    data = pd.concat([data, new_columns], axis=1)
    features = read_features(file_path='../data/skills.txt')
    data = add_features_to_dataframe(data, features)


    data = data.drop(columns=['position', 'key_skills', 'salary', 'unique_work', 'work_experience'])
    categorial_features = ['country', 'city', 'client_name', 'company_type']
    for feature in categorial_features:
        data[feature] = data[feature].astype('category').cat.codes

    if 'grade_proof' in data.columns:
        data = data.drop(columns=['grade_proof'])

    def normalize_column(col):
        # Если стандартное отклонение колонки 0, не нормализуем её
        if col.std() == 0:
            return col
        return (col - col.min()) / (col.max() - col.min())
    data = data.apply(normalize_column)

    # Загрузка модели из файла
    model_loaded = CatBoostClassifier()
    model_loaded.load_model('models/catboost_model.bin')

    # Прогнозирование с загруженной модели
    y_pred = model_loaded.predict(data)
    return y_pred


if __name__ == '__main__':
    main()