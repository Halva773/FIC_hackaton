import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from formating import split_work_experience as unique_work


def calculate_experience_months(data):
    """
    Функция подсчёта количества месяцев, которые работал человек

    :param data: DataFrame with data
    :return:
    """
    intervals = []
    current_date = datetime.now()

    # Извлечение временных интервалов
    for entry in data:
        match = re.match(r'(\d{4}-\d{2}-\d{2}) - (\d{4}-\d{2}-\d{2}|:)', entry)
        if match:
            start_date = datetime.strptime(match.group(1), '%Y-%m-%d')
            end_date = current_date if match.group(2) == ':' else datetime.strptime(match.group(2), '%Y-%m-%d')
            intervals.append((start_date, end_date))

    # Объединение пересекающихся интервалов
    intervals.sort()  # Сортируем по началу интервалов
    merged_intervals = []
    for start, end in intervals:
        if not merged_intervals or merged_intervals[-1][1] < start:
            merged_intervals.append((start, end))
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))

    # Подсчёт суммарной продолжительности в месяцах
    total_months = sum((relativedelta(end, start).years * 12 + relativedelta(end, start).months) for start, end in merged_intervals)
    return total_months

def generate_worker_features(df):
    df['unique_work'] = df['work_experience'].apply(unique_work)
    df['work_experience_months'] = df['unique_work'].apply(calculate_experience_months)
    df['count_works'] = df['unique_work'].apply(len)
    df['avg_time_per_work'] = df['work_experience_months'] / df['count_works']
    return df

if __name__ == '__main__':
    with open('../data/client_dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    df = generate_worker_features(pd.DataFrame(data))
    print(df[['unique_work', 'count_works', 'work_experience_months', 'avg_time_per_work']].head(10))
