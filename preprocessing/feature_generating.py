import json
import re
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


def split_work_experience(text):
    works = re.split(r"(?=\d{4}-\d{2}-\d{2}\s-\s(?:\d{4}-\d{2}-\d{2}|:))", text)
    return set(entry.strip() for entry in works if entry.strip())

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
    total_months = sum(
        (relativedelta(end, start).years * 12 + relativedelta(end, start).months) for start, end in merged_intervals)
    return total_months


def generate_worker_features(df):
    df['unique_work'] = df['work_experience'].apply(split_work_experience)
    df['work_experience_months'] = df['unique_work'].apply(calculate_experience_months)
    df['count_works'] = df['unique_work'].apply(len)
    df['avg_time_per_work'] = df['work_experience_months'] / df['count_works']
    return df


def extract_salaries(df, salary_column):
    """
    Извлекает минимальную и комфортную зарплату из текстовой колонки, а также грейд.

    Args:
        df (pd.DataFrame): Исходный датафрейм.
        salary_column (str): Имя текстовой колонки с информацией о зарплате.

    Returns:
        pd.DataFrame: Датафрейм с добавленными колонками `min_salary`, `comfort_salary` и `grade`.
    """

    def process_salary(salary_text):
        if len(salary_text) < 2:
            return 0, 0, 0, 0
        # Проверка на наличие грейда
        grade_match = re.search(r'(?:\bгрейд\s*\d+|\d+\s*-?[й-]?\s*грейд)', salary_text, flags=re.IGNORECASE)
        if grade_match:
            # Если найден грейд, извлекаем его и очищаем текст
            grade = int(re.search(r'\d+', grade_match.group(0)).group())
            # Удаляем текст с грейдом из salary
            salary_text = re.sub(r'(?:\bгрейд\s*\d+|\d+\s*-?[й-]?\s*грейд)', '', salary_text, flags=re.IGNORECASE)
            return 0, 0, grade, salary_text.strip()

        matches = re.findall(r'\b\d{1,3}(?:[\.,]\d{1,3})*(?:\s\d{3})?\s?(?:евро)\b', salary_text)

        numbers = [int(match.split(' ')[0].replace(" ", "").replace(".", "").replace(',', '')) for match in matches]
        numbers = [i * 100 * 90 if i < 100 else i * 90 for i in numbers]
        if numbers:
            return min(numbers), max(numbers), 0, salary_text

        # Находим числа из 2 до 6 цифр, поддерживая разделители пробелом или точкой
        matches = re.findall(r'\b\d{2,6}(?:[\s.,]\d{3})?', salary_text)
        numbers = [int(match.replace(" ", "").replace(".", "").replace(',', '')) for match in matches]

        numbers = [i * 1000 if i < 1000 else i for i in numbers]
        # Возвращаем минимальную и комфортную зарплату
        if numbers:
            return min(numbers), max(numbers), 0, salary_text
        return 0, 0, 0, salary_text

    # Применяем функцию к каждой строке в колонке salary_column
    df[['min_salary', 'comfort_salary', 'grade', 'updated_salary']] = df[salary_column].apply(
        lambda x: pd.Series(process_salary(x) if isinstance(x, str) else (0, 0, 0, x))
    )
    df.loc[(df.min_salary == 0) & (df.comfort_salary != 0), 'min_salary'] = df.loc[
        (df.min_salary == 0) & (df.comfort_salary != 0), 'comfort_salary']
    df.loc[(df.min_salary != 0) & (df.comfort_salary == 0), 'comfort_salary'] = df.loc[
        (df.min_salary != 0) & (df.comfort_salary == 0), 'min_salary']
    # Обновляем колонку salary с учетом удаленного текста про грейд
    #     df[salary_column] = df['updated_salary']
    df.drop(columns=['updated_salary'], inplace=True)

    return df


if __name__ == '__main__':
    with open('../data/client_dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    data = generate_worker_features(pd.DataFrame(data))
    data = extract_salaries(data, 'salary')
    print(data[['min_salary', 'comfort_salary', 'grade', 'unique_work', 'count_works', 'work_experience_months',
                'avg_time_per_work']].head(10))
