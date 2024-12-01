import json
import re
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from rapidfuzz import process, fuzz


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


def normalize_russian_words(words):
    """
    Приводит русский текст к нормальной форме.
    """
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    return [morph.parse(word)[0].normal_form for word in words]


def clean_and_reduce_skills(data, column_name='key_skills', threshold=80):
    """
    Очищает, нормализует и сокращает количество уникальных навыков в указанном столбце.

    Параметры:
        data (pd.DataFrame): Исходный DataFrame.
        column_name (str): Название столбца с навыками.
        threshold (int): Порог для объединения похожих навыков (0-100).

    Возвращает:
        list: Отсортированный и уменьшенный список уникальных навыков.
    """
    # Шаг 1: Удаляем пропущенные значения и разделяем навыки
    all_skills = data[column_name].dropna().str.split(',').sum()

    # Шаг 2: Приводим к нижнему регистру и удаляем лишние пробелы
    cleaned_skills = {skill.strip().lower() for skill in all_skills if skill.strip()}

    # Шаг 3: Объединяем похожие навыки
    reduced_skills = []
    while cleaned_skills:
        skill = cleaned_skills.pop()
        similar_skills = process.extract(skill, cleaned_skills, scorer=fuzz.ratio, score_cutoff=threshold)
        reduced_skills.append(skill)
        # Удаляем все найденные похожие навыки
        for match, _, _ in similar_skills:
            cleaned_skills.discard(match)

    # Шаг 4: Сортируем результат
    return sorted(reduced_skills)


def add_features_to_dataframe(data, features, skills_column='key_skills'):
    """
    Добавляет столбцы для признаков и отмечает их наличие в ключевых навыках.

    Параметры:
        data (pd.DataFrame): Исходный DataFrame.
        features (list): Список признаков.
        skills_column (str): Название столбца с ключевыми навыками.

    Возвращает:
        pd.DataFrame: DataFrame с добавленными столбцами признаков.
    """
    # Удаляем дубликаты из списка признаков
    features = list(set(features))

    # Отделяем русские слова от остальных
    russian_words = [feature for feature in features if re.search(r'[а-яА-Я]', feature)]
    other_words = [feature for feature in features if feature not in russian_words]

    # Нормализуем русские слова
    normalized_russian_words = normalize_russian_words(russian_words)

    # Объединяем нормализованные русские слова с другими
    final_features = normalized_russian_words + other_words

    # Создаем DataFrame со столбцами из final_features, заполненными нулями
    new_columns = pd.DataFrame(0, index=data.index, columns=final_features)
    # Объединяем существующий DataFrame с новым
    data = pd.concat([data, new_columns], axis=1)

    # Проверяем навыки в key_skills построчно
    for index, row in data.iterrows():
        key_skills = str(row[skills_column]).lower().split(', ')
        for feature in final_features:
            if feature in key_skills:
                data.loc[index, feature] = 1  # Помечаем 1, если навык найден

    return data


def read_features(file_path='data/skills.txt'):
    with open(file_path, 'r') as file:
        features = file.read().split(', ')
    return features

if __name__ == '__main__':
    with open('../data/client_dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    # data = generate_worker_features(data)
    # data = extract_salaries(data, 'salary')
    # print(data[['min_salary', 'comfort_salary', 'grade', 'unique_work', 'count_works', 'work_experience_months',
    #             'avg_time_per_work']].head(10))
    # Чтение JSON-файла

    features = read_features(file_path='../data/skills.txt')

    processed_data = add_features_to_dataframe(data, features)
    print(processed_data.columns)
    print(processed_data.shape)
    print(processed_data['c++'].sum())
