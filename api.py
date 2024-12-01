from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import joblib
import json
import os
from preprocessing.feature_generating import generate_worker_features, extract_salaries, read_features, add_features_to_dataframe
from preprocessing.ratings import company_rates
from preprocessing.vectorizing import process_skills, cosine_distance

# Загрузка модели
MODEL_PATH = "models/catboost_model.bin"
model = joblib.load(MODEL_PATH)

# Чтение признаков
FEATURES_PATH = 'data/skills.txt'
features = read_features(file_path=FEATURES_PATH)

# Создание FastAPI приложения
app = FastAPI()

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

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        # Проверка формата файла
        if not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a JSON file.")

        # Чтение и парсинг файла
        contents = await file.read()
        data = json.loads(contents)

        # Предобработка данных
        preprocessed_data = preprocess_data(data)

        # Применение модели
        predictions = model.predict(preprocessed_data)

        # Создание результата
        result_df = pd.DataFrame({
            'id': preprocessed_data.index,  # Убедитесь, что id есть в данных
            'grade': predictions
        })

        # Сохранение в файл result.json
        result_file = "result.json"
        result_df.to_json(result_file, orient='records', lines=False)

        # Возврат результата
        return FileResponse(result_file, media_type="application/json", filename="result.json")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
