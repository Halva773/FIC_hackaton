import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import json
import os
from catboost import CatBoostClassifier
from preprocessing.feature_generating import generate_worker_features, extract_salaries, read_features, add_features_to_dataframe
from preprocessing.ratings import company_rates
from preprocessing.vectorizing import process_skills, cosine_distance
from PIL import Image, ImageTk  # Для работы с изображениями
# Загрузка модели и признаков
model = CatBoostClassifier()
model.load_model('models/catboost_model.bin')

FEATURES_PATH = 'data/skills.txt'
features = read_features(file_path=FEATURES_PATH)


# Функция предобработки данных
def preprocess_data(data):
    df = pd.DataFrame(data)
    df = generate_worker_features(df)
    df = extract_salaries(df, 'salary')
    df = company_rates(df)
    new_columns = df.apply(lambda row: process_skills(row, cosine_distance), axis=1)
    df = pd.concat([df, new_columns], axis=1)
    df = add_features_to_dataframe(df, features)
    df = df.drop(columns=['key_skills', 'position', 'salary', 'work_experience'], errors='ignore')
    return df


# Функция обработки файла
def process_file(input_path, output_path):
    try:
        # Чтение JSON файла
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Предобработка данных
        preprocessed_data = preprocess_data(data)

        # Применение модели
        predictions = model.predict(preprocessed_data)

        # Создание результата
        result_df = pd.DataFrame({
            'id': preprocessed_data.index,
            'grade': predictions
        })

        # Сохранение в файл
        result_df.to_json(output_path, orient='records', lines=False)

        return True
    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
        return False


# Создание основного окна
def create_app():
    root = tk.Tk()
    root.title("Анализ экспертности по резюме. ФИЦ: Хакатон")
    root.geometry("800x600")

    def select_file():
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")], title="Выберите JSON файл"
        )
        if filepath:
            input_file_var.set(filepath)

    def select_output():
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Сохранить результат как"
        )
        if filepath:
            output_file_var.set(filepath)

    def process_and_save():
        input_path = input_file_var.get()
        output_path = output_file_var.get()

        if not input_path:
            messagebox.showwarning("Предупреждение", "Выберите входной файл!")
            return

        if not output_path:
            messagebox.showwarning("Предупреждение", "Укажите путь для сохранения результата!")
            return

        success = process_file(input_path, output_path)
        if success:
            messagebox.showinfo("Успех", f"Результат успешно сохранён в {output_path}")

     # Функция для отображения второй картинки
    def display_result_image():
        result_img_path = "images/image2.png"  # Укажите путь ко второй картинке
        result_image = Image.open(result_img_path)
        result_image = result_image.resize((200, 200), Image.ANTIALIAS)  # Уменьшаем размер картинки
        result_photo = ImageTk.PhotoImage(result_image)

        result_label = tk.Label(root, image=result_photo)
        result_label.image = result_photo  # Сохраняем ссылку на изображение, чтобы оно отображалось
        result_label.place(relx=0.5, rely=0.6, anchor="center")  # Размещаем картинку в центре

    # Интерфейс
    input_file_var = tk.StringVar()
    output_file_var = tk.StringVar()

    # Заголовок
    tk.Label(root, text="Оценка уровня экспертности по резюме", font=("Arial", 20), fg="blue").pack(pady=10)

    # Добавление первого изображения
    img_path = "images/image1.png"  # Укажите путь к первому изображению
    image = Image.open(img_path)
    image = image.resize((600, 200), Image.Resampling.LANCZOS)  # Изменение размера изображения
    photo = ImageTk.PhotoImage(image)
    tk.Label(root, image=photo).pack(pady=10)

    # Поля для выбора файлов
    tk.Label(root, text="Входной JSON файл:").pack(anchor="w", padx=20, pady=5)
    tk.Entry(root, textvariable=input_file_var, width=70).pack(padx=20, pady=5)
    tk.Button(root, text="Выбрать файл", command=select_file).pack(pady=5)

    tk.Label(root, text="Сохранить результат как:").pack(anchor="w", padx=20, pady=5)
    tk.Entry(root, textvariable=output_file_var, width=70).pack(padx=20, pady=5)
    tk.Button(root, text="Выбрать путь", command=select_output).pack(pady=5)

    tk.Button(root, text="Обработать", command=process_and_save, bg="green", fg="white").pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    create_app()