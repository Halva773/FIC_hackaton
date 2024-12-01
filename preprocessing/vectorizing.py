import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()
model_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).cuda()  # Переносим модель на GPU


# Функция для вычисления косинусного расстояния с использованием CUDA
def cosine_distance(text1, text2, model=model, tokenizer=tokenizer):
    # Токенизация и создание эмбеддингов
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to('cuda')
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to('cuda')

    # Получение эмбеддингов
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)  # Берём среднее по всем токенам
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

    # Вычисление косинусного расстояния
    sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return sim.item()  # Возвращаем скалярное значение


# Функция для обработки строки
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


if __name__ == '__main__':
#     distance = cosine_distance("SQL", '"программист" (по факту аналитик), отдел проектирование информационных систем')
#     print(f"Косинусное расстояние: {distance}")

    data = pd.read_csv('../data/client_dataset_csv.csv', index_col=0)[:3]
    print(data.shape)
    # Применяем функцию к каждому ряду
    new_columns = data.apply(lambda row: process_skills(row, cosine_distance), axis=1)

    # Объединяем результаты с оригинальным датасетом
    data = pd.concat([data, new_columns], axis=1)

    print(data[['mean_distance', 'count_above_05', 'min_distance', 'max_distance', 'std_distance']])
