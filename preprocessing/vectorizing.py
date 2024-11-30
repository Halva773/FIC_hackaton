import torch
from transformers import AutoTokenizer, AutoModel, logging
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np

logging.set_verbosity_error()  # Отключает все сообщения, кроме ошибок



def cosine_distance(text1, text2, model_name="DeepPavlov/rubert-base-cased"):
    """
    Вычисляет косинусное расстояние между двумя текстами с использованием BERT.

    Аргументы:
        text1 (str): Первый текст.
        text2 (str): Второй текст.
        model_name (str): Название модели BERT (по умолчанию ruBERT).

    Возвращает:
        float: Косинусное расстояние между двумя текстами.
    """
    # Загрузка модели и токенайзера
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Токенизация текстов
    tokens1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    tokens2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # Получение эмбеддингов
    with torch.no_grad():
        embedding1 = model(**tokens1).last_hidden_state.mean(dim=1).squeeze()
        embedding2 = model(**tokens2).last_hidden_state.mean(dim=1).squeeze()

    # Перевод в numpy для расчёта косинусного расстояния
    embedding1 = embedding1.cpu().numpy()
    embedding2 = embedding2.cpu().numpy()

    # Вычисление косинусного расстояния
    distance = cosine(embedding1, embedding2)
    return distance


# Функция для обработки одной строки
def process_skills(row, cosine_distance):
    skills = [skill.strip() for skill in row['key_skills'].split(',') if skill.strip()]
    position = row['position']

    # Считаем косинусное расстояние для каждого навыка
    distances = [cosine_distance(skill, position) for skill in skills]

    # Вычисляем параметры распределения
    if distances:
        mean_distance = np.mean(distances)
        count_above_05 = sum(1 for d in distances if d > 0.5)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
    else:
        mean_distance = 0
        count_above_05 = 0
        min_distance = 0
        max_distance = 0
        std_distance = 0

    return pd.Series({
        'mean_distance': mean_distance,
        'count_above_05': count_above_05,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'std_distance': std_distance
    })


# Пример использования
# Датасет data должен содержать колонки 'key_skills' и 'position'
data = pd.DataFrame({
    'key_skills': ['Python, SQL, Machine Learning', 'Java, Spring, Hibernate', 'Data Analysis, Excel'],
    'position': ['Data Scientist', 'Backend Developer', 'Data Analyst']
})




if __name__ == '__main__':
#     distance = cosine_distance("SQL", '"программист" (по факту аналитик), отдел проектирование информационных систем')
#     print(f"Косинусное расстояние: {distance}")

    data = pd.read_csv('../data/client_dataset_csv.csv', index_col=0)[:10]
    print(data.shape)
    # Применяем функцию к каждому ряду
    new_columns = data.apply(lambda row: process_skills(row, cosine_distance), axis=1)

    # Объединяем результаты с оригинальным датасетом
    data = pd.concat([data, new_columns], axis=1)

    print(data)
